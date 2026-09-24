/**
 * Conversation state for the UI: a pure reducer over the turn's SSE events,
 * plus hydrate() from the server's display record (GET after a refresh).
 *
 * A turn's status uses the display record's vocabulary, so a streamed turn
 * and a hydrated one look the same:
 * - done arrives: "complete", or "error" when exit_reason is "error" (the
 *   same rule as server/app.py).
 * - streamEnded with no done: "crashed". The server's crash path ends the
 *   stream without a done.
 * - streamFailed (the client stopped reading):
 *     AbortError -> "aborted" (the user pressed stop; the server also sees
 *                   a disconnect and aborts the turn at its next step).
 *     anything else (HTTP refusal, network failure, malformed stream) -> "error", with
 *                   clientError saying what happened. The turn failed from
 *                   the user's point of view, but it wasn't a server-side
 *                   agent error, so exit_reason stays null.
 *   A later GET shows the server's own view of such a turn: usually
 *   "aborted", or "complete" if the answer already existed.
 *
 * `running` stays true from turnStarted until streamEnded or streamFailed,
 * even after done. The server only unlocks the conversation (no more 409s)
 * once the stream closes.
 *
 * Events arriving when no turn is running (e.g. after done) are ignored.
 */

import { ApiError } from './api'
import type {
  AgentEvent,
  DisplayRecord,
  DisplayStep,
  DisplayTurn,
  ErrorEvent,
  FilmCard,
  JsonObject,
  TurnStatus,
  Usage,
} from './types'

export interface Step {
  /** Call id within the turn ("c1", ...). Steps are kept in arrival order. */
  id: string
  name: string
  arguments: JsonObject
  iteration: number
  /** Full result: only for live turns (display records omit results). */
  result?: JsonObject
  /** Same rule as the server's display record; null until the result arrives. */
  result_count: number | null
  /** null until the result arrives. */
  duration_ms: number | null
  blocked_duplicate: boolean
  /** result.error for a failed or blocked call, else null. */
  error: string | null
  /** The tool_call arrived but its tool_result hasn't (yet). */
  pending: boolean
}

/** Why the client stopped reading a turn's stream. */
export interface ClientError {
  /** aborted: the AbortSignal fired. http: the POST was refused (404, 409,
   * 422, ...). network: fetch failed. protocol: the stream was malformed. */
  kind: 'aborted' | 'http' | 'network' | 'protocol'
  message: string
}

export interface Turn {
  question: string
  status: TurnStatus
  steps: Step[]
  answer: string | null
  films: FilmCard[]
  exit_reason: DisplayTurn['exit_reason']
  usage: Usage | null
  elapsed_ms: number | null
  /** The server's error event for this turn (an agent or server error). */
  error: ErrorEvent | null
  /** Set by streamFailed. */
  clientError: ClientError | null
}

export interface ConversationState {
  conversationId: string | null
  turns: Turn[]
  /** Index into turns, or -1 when there are none. */
  selectedTurn: number
  /** A turn's stream is open (see the module docstring). */
  running: boolean
}

export type ConversationAction =
  | { type: 'turnStarted'; question: string }
  | { type: 'event'; event: AgentEvent }
  | { type: 'streamEnded' }
  | { type: 'streamFailed'; error: unknown }
  | { type: 'hydrate'; record: DisplayRecord }
  | { type: 'selectTurn'; index: number }
  | { type: 'reset'; conversationId: string | null }

export const actions = {
  turnStarted: (question: string): ConversationAction => ({ type: 'turnStarted', question }),
  event: (event: AgentEvent): ConversationAction => ({ type: 'event', event }),
  streamEnded: (): ConversationAction => ({ type: 'streamEnded' }),
  streamFailed: (error: unknown): ConversationAction => ({ type: 'streamFailed', error }),
  hydrate: (record: DisplayRecord): ConversationAction => ({ type: 'hydrate', record }),
  selectTurn: (index: number): ConversationAction => ({ type: 'selectTurn', index }),
  /** A fresh, empty conversation: the new one's id from createConversation(), or none yet. */
  reset: (conversationId: string | null = null): ConversationAction => ({ type: 'reset', conversationId }),
}

export const initialState: ConversationState = { conversationId: null, turns: [], selectedTurn: -1, running: false }

// Result keys holding the list a tool returns (mirrors server/app.py _result_count).
const RESULT_LIST_KEYS = ['films', 'matches', 'credits', 'genres', 'recommendations']

export function resultCount(result: JsonObject): number | null {
  if ('error' in result) return null
  if ('tmdb_id' in result) return 1 // search_tmdb: one film
  for (const key of RESULT_LIST_KEYS) {
    const value = result[key]
    if (Array.isArray(value)) return value.length
  }
  return null
}

function resultError(result: JsonObject): string | null {
  const error = result.error
  return typeof error === 'string' ? error : null
}

function newTurn(question: string): Turn {
  return {
    question,
    status: 'running',
    steps: [],
    answer: null,
    films: [],
    exit_reason: null,
    usage: null,
    elapsed_ms: null,
    error: null,
    clientError: null,
  }
}

function applyEvent(turn: Turn, event: AgentEvent): Turn {
  switch (event.type) {
    case 'tool_call': {
      if (turn.steps.some((s) => s.id === event.id)) return turn
      const step: Step = {
        id: event.id,
        name: event.name,
        arguments: event.arguments,
        iteration: event.iteration,
        result_count: null,
        duration_ms: null,
        blocked_duplicate: false,
        error: null,
        pending: true,
      }
      return { ...turn, steps: [...turn.steps, step] }
    }
    case 'tool_result': {
      const completed = {
        result: event.result,
        result_count: resultCount(event.result),
        duration_ms: event.duration_ms,
        blocked_duplicate: event.blocked_duplicate,
        error: resultError(event.result),
        pending: false,
      }
      if (!turn.steps.some((s) => s.id === event.id)) {
        // No tool_call seen for it (shouldn't happen): keep the result anyway.
        const step: Step = { id: event.id, name: event.name, arguments: {}, iteration: event.iteration, ...completed }
        return { ...turn, steps: [...turn.steps, step] }
      }
      return { ...turn, steps: turn.steps.map((s) => (s.id === event.id ? { ...s, ...completed } : s)) }
    }
    case 'answer':
      return { ...turn, answer: event.text, films: event.films ?? [] }
    case 'error':
      return { ...turn, error: event }
    case 'done':
      return {
        ...turn,
        status: event.exit_reason === 'error' ? 'error' : 'complete',
        exit_reason: event.exit_reason,
        usage: event.usage,
        elapsed_ms: event.elapsed_ms,
        error: turn.error ?? event.error,
      }
  }
}

function clientErrorFor(error: unknown): ClientError {
  const message = error instanceof Error ? error.message : String(error)
  // By name: fetch rejects with a DOMException, which isn't always an Error subclass.
  if ((error as { name?: unknown } | null)?.name === 'AbortError') return { kind: 'aborted', message }
  if (error instanceof ApiError && error.status !== null) return { kind: 'http', message }
  if (error instanceof TypeError) return { kind: 'network', message } // fetch's network failures
  return { kind: 'protocol', message }
}

function stepFromDisplay(step: DisplayStep): Step {
  return {
    id: step.id,
    name: step.tool,
    arguments: step.arguments,
    iteration: step.iteration,
    result_count: step.result_count,
    duration_ms: step.duration_ms,
    blocked_duplicate: step.blocked_duplicate,
    error: step.error,
    pending: false,
  }
}

function turnFromDisplay(turn: DisplayTurn): Turn {
  return {
    question: turn.question,
    status: turn.status,
    steps: turn.steps.map(stepFromDisplay),
    answer: turn.answer,
    films: turn.films,
    exit_reason: turn.exit_reason,
    usage: turn.usage,
    elapsed_ms: turn.elapsed_ms,
    error: turn.error,
    clientError: null,
  }
}

/** Replace the running (last) turn, if there is one. */
function updateRunningTurn(state: ConversationState, update: (turn: Turn) => Turn): ConversationState {
  const last = state.turns.length - 1
  if (last < 0 || state.turns[last].status !== 'running') return state
  const turns = state.turns.slice()
  turns[last] = update(turns[last])
  return { ...state, turns }
}

export function conversationReducer(state: ConversationState, action: ConversationAction): ConversationState {
  switch (action.type) {
    case 'turnStarted': {
      const turns = [...state.turns, newTurn(action.question)]
      return { ...state, turns, selectedTurn: turns.length - 1, running: true }
    }
    case 'event':
      if (!state.running) return state
      return updateRunningTurn(state, (turn) => applyEvent(turn, action.event))
    case 'streamEnded': {
      if (!state.running) return state
      const next = updateRunningTurn(state, (turn) => ({ ...turn, status: 'crashed', exit_reason: 'crashed' }))
      return { ...next, running: false }
    }
    case 'streamFailed': {
      if (!state.running) return state
      const clientError = clientErrorFor(action.error)
      const next = updateRunningTurn(state, (turn) => ({
        ...turn,
        status: clientError.kind === 'aborted' ? 'aborted' : 'error',
        exit_reason: clientError.kind === 'aborted' ? 'aborted' : null,
        clientError,
      }))
      return { ...next, running: false }
    }
    case 'hydrate': {
      const turns = action.record.turns.map(turnFromDisplay)
      return {
        conversationId: action.record.conversation_id,
        turns,
        selectedTurn: turns.length - 1,
        running: action.record.running,
      }
    }
    case 'selectTurn':
      if (!Number.isInteger(action.index) || action.index < 0 || action.index >= state.turns.length) return state
      return { ...state, selectedTurn: action.index }
    case 'reset':
      return { ...initialState, conversationId: action.conversationId }
  }
}
