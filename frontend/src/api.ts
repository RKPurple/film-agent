/**
 * Typed client for the cinemagent API (server/app.py). Always relative
 * /api/... URLs: in development Vite proxies them to the FastAPI server.
 *
 * Nothing here retries. Re-sending a question would start a duplicate turn,
 * so every sendMessage() is exactly one fetch; the caller decides what to do
 * with a failure.
 */

import { parseSse, type SseMessage } from './sse'
import type {
  AgentEvent,
  AgentEventType,
  CreateConversationResponse,
  DisplayRecord,
  ValidationErrorDetail,
} from './types'

/** GET /api/health. 200 with status "ok" (db "ok", or "busy" when every
 * pooled connection is in use by a running turn), or 503 with status
 * "degraded" and db describing the failure. */
export interface HealthResponse {
  status: 'ok' | 'degraded'
  model: string
  films: number
  tmdb: boolean
  db: string
}

/** A health check either reached the API and got a health body, or didn't:
 * connection refused, the dev proxy's error while uvicorn is still loading
 * models, or an unexpected response. */
export type HealthCheck =
  | { reachable: true; health: HealthResponse }
  | { reachable: false; error: string }

function isHealthResponse(body: unknown): body is HealthResponse {
  if (typeof body !== 'object' || body === null) return false
  const b = body as Record<string, unknown>
  return (
    (b.status === 'ok' || b.status === 'degraded') &&
    typeof b.model === 'string' &&
    typeof b.films === 'number' &&
    typeof b.tmdb === 'boolean' &&
    typeof b.db === 'string'
  )
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthCheck> {
  let response: Response
  try {
    response = await fetch('/api/health', { signal, headers: { Accept: 'application/json' } })
  } catch (error) {
    if (signal?.aborted) throw error
    return { reachable: false, error: String(error) }
  }
  if (response.status !== 200 && response.status !== 503) {
    return { reachable: false, error: `HTTP ${response.status}` }
  }
  try {
    const body: unknown = await response.json()
    if (isHealthResponse(body)) return { reachable: true, health: body }
    return { reachable: false, error: 'unexpected health response' }
  } catch {
    return { reachable: false, error: 'health response was not JSON' }
  }
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

/** Any failed API call. status is the HTTP status, or null when there was
 * no usable HTTP response (e.g. a 200 that isn't an event stream). */
export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** 404: the server doesn't know this conversation (e.g. it restarted, and
 * conversations live in memory). */
export class NotFoundError extends ApiError {
  constructor(message = 'conversation not found') {
    super(message, 404)
    this.name = 'NotFoundError'
  }
}

/** 409: a turn is already running in this conversation. */
export class TurnInProgressError extends ApiError {
  constructor(message = 'a turn is already running in this conversation') {
    super(message, 409)
    this.name = 'TurnInProgressError'
  }
}

/** 422: the request failed validation (e.g. an empty or too-long question). */
export class ValidationError extends ApiError {
  readonly details: ValidationErrorDetail[]

  constructor(details: ValidationErrorDetail[], message?: string) {
    super(message ?? (details.map((d) => d.msg).join('; ') || 'invalid request'), 422)
    this.name = 'ValidationError'
    this.details = details
  }
}

async function readDetail(response: Response): Promise<unknown> {
  try {
    const body: unknown = await response.json()
    return typeof body === 'object' && body !== null && 'detail' in body ? body.detail : undefined
  } catch {
    return undefined
  }
}

/** Map a non-2xx response to its typed error, reading the body as JSON only. */
async function errorFor(response: Response): Promise<ApiError> {
  const detail = await readDetail(response)
  const message = typeof detail === 'string' ? detail : undefined
  switch (response.status) {
    case 404:
      return new NotFoundError(message)
    case 409:
      return new TurnInProgressError(message)
    case 422:
      return Array.isArray(detail)
        ? new ValidationError(detail as ValidationErrorDetail[])
        : new ValidationError([], message)
    default:
      return new ApiError(message ?? `HTTP ${response.status}`, response.status)
  }
}

const EVENT_TYPES: ReadonlySet<string> = new Set<AgentEventType>(['tool_call', 'tool_result', 'answer', 'error', 'done'])

/** Narrow one parsed SSE message to an agent event. Unknown event names are
 * skipped (null), for forward compatibility; a known name whose data isn't
 * an object with the matching type is a protocol error. */
export function toAgentEvent(message: SseMessage): AgentEvent | null {
  if (!EVENT_TYPES.has(message.event)) return null
  const data = message.data
  if (typeof data !== 'object' || data === null || (data as { type?: unknown }).type !== message.event) {
    throw new ApiError(`SSE event "${message.event}" doesn't carry a matching "type"`, null)
  }
  return data as AgentEvent
}

async function* agentEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<AgentEvent> {
  for await (const message of parseSse(stream)) {
    const event = toAgentEvent(message)
    if (event) yield event
  }
}

const conversationPath = (id: string) => `/api/conversations/${encodeURIComponent(id)}`

/** POST /api/conversations */
export async function createConversation(signal?: AbortSignal): Promise<CreateConversationResponse> {
  const response = await fetch('/api/conversations', { method: 'POST', signal, headers: { Accept: 'application/json' } })
  if (!response.ok) throw await errorFor(response)
  return (await response.json()) as CreateConversationResponse
}

/** GET /api/conversations/{id}: the display record. 404 -> NotFoundError. */
export async function getConversation(id: string, signal?: AbortSignal): Promise<DisplayRecord> {
  const response = await fetch(conversationPath(id), { signal, headers: { Accept: 'application/json' } })
  if (!response.ok) throw await errorFor(response)
  return (await response.json()) as DisplayRecord
}

/**
 * POST a question and stream the turn's events. The HTTP status is checked
 * before any of the body is read: 404 -> NotFoundError, 409 ->
 * TurnInProgressError, 422 -> ValidationError, anything else non-2xx ->
 * ApiError. Those reject the returned promise; failures mid-stream (network,
 * malformed data) are thrown from the iterator.
 *
 * Exactly one fetch per call, never retried. Aborting `signal` cancels the
 * request or the stream (the iterator then throws an AbortError), which is
 * how the UI stops a turn: the server sees the disconnect and aborts the
 * turn at its next step, unless the answer already exists.
 */
export async function sendMessage(id: string, question: string, signal?: AbortSignal): Promise<AsyncGenerator<AgentEvent>> {
  const response = await fetch(`${conversationPath(id)}/messages`, {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ question }),
  })
  if (!response.ok) throw await errorFor(response)
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.startsWith('text/event-stream') || response.body === null) {
    throw new ApiError(`expected an event stream, got ${contentType || 'no content type'}`, response.status)
  }
  return agentEvents(response.body)
}
