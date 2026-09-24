import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  ApiError,
  createConversation,
  getConversation,
  NotFoundError,
  sendMessage,
  TurnInProgressError,
  ValidationError,
} from './api'
import { bytesOf, FIXTURES, fixtureEvents, streamOf } from './test/sseFixtures'
import type { AgentEvent } from './types'

afterEach(() => {
  vi.unstubAllGlobals()
})

function stubFetch(impl: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
  const fetchMock = vi.fn(impl)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** An error response whose body can only be read as JSON: touching .body
 * (the stream) is recorded, so a test can prove it never happened. */
function errorResponse(status: number, detail: unknown) {
  const bodyRead = vi.fn()
  const response = {
    ok: false,
    status,
    headers: new Headers({ 'content-type': 'application/json' }),
    json: vi.fn(async () => ({ detail })),
    get body() {
      bodyRead()
      throw new Error('the body stream must not be read for an error status')
    },
  } as unknown as Response
  return { response, bodyRead }
}

function sseResponse(stream: ReadableStream<Uint8Array>) {
  return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8' } })
}

async function drain(events: AsyncIterable<AgentEvent>): Promise<AgentEvent[]> {
  const out: AgentEvent[] = []
  for await (const event of events) out.push(event)
  return out
}

describe('sendMessage', () => {
  it('POSTs once to the relative URL and streams typed events', async () => {
    const fetchMock = stubFetch(async () => sseResponse(streamOf([bytesOf(FIXTURES.structured)])))
    const controller = new AbortController()
    const events = await drain(await sendMessage('conv/1', 'Idris Elba?', controller.signal))

    expect(events).toEqual(await fixtureEvents('structured'))
    expect(events.map((e) => e.type)).toEqual(['tool_call', 'tool_result', 'answer', 'done'])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/conversations/conv%2F1/messages')
    expect(init).toMatchObject({ method: 'POST', signal: controller.signal, body: JSON.stringify({ question: 'Idris Elba?' }) })
  })

  it.each([
    [404, 'conversation not found', NotFoundError],
    [409, 'a turn is already running in this conversation', TurnInProgressError],
    [500, 'Internal Server Error', ApiError],
  ] as const)('maps HTTP %i to its typed error without reading the stream', async (status, detail, ErrorClass) => {
    const { response, bodyRead } = errorResponse(status, detail)
    const fetchMock = stubFetch(async () => response)

    const error = await sendMessage('c', 'q').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ErrorClass)
    expect(error).toMatchObject({ status, message: detail })
    expect(bodyRead).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('maps 422 to ValidationError carrying the server detail', async () => {
    const detail = [{ type: 'string_too_short', loc: ['body', 'question'], msg: 'String should have at least 1 character', input: '' }]
    const { response, bodyRead } = errorResponse(422, detail)
    const fetchMock = stubFetch(async () => response)

    const error = await sendMessage('c', '   ').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ValidationError)
    expect((error as ValidationError).details).toEqual(detail)
    expect((error as ValidationError).message).toBe('String should have at least 1 character')
    expect(bodyRead).not.toHaveBeenCalled()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('never retries: a network failure rejects after one fetch', async () => {
    const fetchMock = stubFetch(async () => {
      throw new TypeError('Failed to fetch')
    })
    await expect(sendMessage('c', 'q')).rejects.toThrow(TypeError)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('never retries: a stream failing mid-turn throws from the iterator after one fetch', async () => {
    // First read: the tool events. Second read: the connection drops.
    const head = bytesOf(FIXTURES.structured.slice(0, FIXTURES.structured.indexOf('event: answer')))
    let reads = 0
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (reads++ === 0) controller.enqueue(head)
        else controller.error(new TypeError('network error'))
      },
    })
    const fetchMock = stubFetch(async () => sseResponse(stream))
    const seen: string[] = []
    const error = await (async () => {
      for await (const event of await sendMessage('c', 'q')) seen.push(event.type)
    })().catch((e: unknown) => e)
    expect(error).toBeInstanceOf(TypeError)
    expect(seen).toEqual(['tool_call', 'tool_result'])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('an abort rejects with the AbortError, after one fetch', async () => {
    const fetchMock = stubFetch(async (_input, init) => {
      init?.signal?.throwIfAborted()
      return sseResponse(streamOf([]))
    })
    const controller = new AbortController()
    controller.abort()
    await expect(sendMessage('c', 'q', controller.signal)).rejects.toMatchObject({ name: 'AbortError' })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('rejects a 200 that is not an event stream', async () => {
    stubFetch(async () => Response.json({ hello: 'world' }))
    await expect(sendMessage('c', 'q')).rejects.toThrow(/expected an event stream/)
  })
})

describe('createConversation and getConversation', () => {
  it('createConversation POSTs and returns the id', async () => {
    const body = { conversation_id: 'abc', created_at: '2026-09-24T00:00:00+00:00' }
    const fetchMock = stubFetch(async () => Response.json(body, { status: 201 }))
    expect(await createConversation()).toEqual(body)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/conversations')
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST' })
  })

  it('getConversation throws NotFoundError on 404', async () => {
    stubFetch(async () => Response.json({ detail: 'conversation not found' }, { status: 404 }))
    await expect(getConversation('gone')).rejects.toBeInstanceOf(NotFoundError)
  })
})
