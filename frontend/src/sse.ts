/**
 * Server-Sent Events parsing, following the WHATWG event-stream rules for
 * the fields we use (event, data):
 * - Lines end in CRLF, LF or CR. A CR at the very end of a chunk is held
 *   back until the next chunk shows whether an LF follows it.
 * - "field: value" drops one leading space from the value. A line starting
 *   with ":" is a comment (the server's ": ping" keep-alive). Other fields
 *   (id, retry) are ignored.
 * - Several data lines in one event are joined with "\n".
 * - A blank line dispatches the event, but only if it had data. The event
 *   name defaults to "message".
 * - An event still incomplete when the stream ends (no blank line after it)
 *   is discarded.
 *
 * Bytes are decoded with one streaming TextDecoder, so a multi-byte UTF-8
 * character split across chunks decodes correctly. Every data payload is
 * parsed as JSON; malformed JSON throws SseParseError naming the event.
 */

export interface SseMessage {
  event: string
  data: unknown
}

export class SseParseError extends Error {
  readonly event: string

  constructor(event: string, cause: unknown) {
    super(`Malformed JSON in the data of SSE event "${event}": ${cause instanceof Error ? cause.message : String(cause)}`)
    this.name = 'SseParseError'
    this.event = event
  }
}

const LINE_END = /\r\n|\r|\n/g

/** Incremental event-stream decoder over text: push() each decoded chunk,
 * end() once the stream is done. Both return the events completed so far. */
export function createSseDecoder() {
  let buffer = ''
  let eventName = ''
  let dataLines: string[] = []

  function dispatch(): SseMessage | null {
    const name = eventName || 'message'
    const lines = dataLines
    eventName = ''
    dataLines = []
    if (lines.length === 0) return null
    const raw = lines.join('\n')
    try {
      return { event: name, data: JSON.parse(raw) }
    } catch (error) {
      throw new SseParseError(name, error)
    }
  }

  function processLine(line: string): SseMessage | null {
    if (line === '') return dispatch()
    if (line.startsWith(':')) return null
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') eventName = value
    else if (field === 'data') dataLines.push(value)
    return null
  }

  function drain(final: boolean): SseMessage[] {
    const messages: SseMessage[] = []
    let start = 0
    LINE_END.lastIndex = 0
    for (let match = LINE_END.exec(buffer); match !== null; match = LINE_END.exec(buffer)) {
      // A lone CR at the end of the buffer may be the first half of a CRLF.
      if (!final && match[0] === '\r' && match.index === buffer.length - 1) break
      const message = processLine(buffer.slice(start, match.index))
      start = match.index + match[0].length
      if (message) messages.push(message)
    }
    buffer = buffer.slice(start)
    return messages
  }

  return {
    push(text: string): SseMessage[] {
      buffer += text
      return drain(false)
    },
    end(): SseMessage[] {
      const messages = drain(true)
      // Whatever is left is an unterminated line or event: discarded.
      buffer = ''
      eventName = ''
      dataLines = []
      return messages
    },
  }
}

/** Parse an event stream into {event, data} messages as bytes arrive.
 * Stopping early (break, or an error) cancels the underlying stream. */
export async function* parseSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseMessage> {
  const reader = stream.getReader()
  const decoder = new TextDecoder('utf-8')
  const sse = createSseDecoder()
  let finished = false
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) {
        finished = true
        yield* sse.push(decoder.decode())
        yield* sse.end()
        return
      }
      yield* sse.push(decoder.decode(value, { stream: true }))
    }
  } finally {
    if (finished) reader.releaseLock()
    else await reader.cancel().catch(() => {})
  }
}
