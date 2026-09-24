/** Raw SSE fixtures captured from the live backend (see fixtures/), and
 * helpers to replay bytes as a ReadableStream. */
import followup from './fixtures/followup.sse?raw'
import recs from './fixtures/recs.sse?raw'
import structured from './fixtures/structured.sse?raw'
import vibe from './fixtures/vibe.sse?raw'

import { toAgentEvent } from '../api'
import { parseSse, type SseMessage } from '../sse'
import type { AgentEvent } from '../types'

export const FIXTURES = { structured, vibe, followup, recs }
export type FixtureName = keyof typeof FIXTURES

const encoder = new TextEncoder()

export function bytesOf(text: string): Uint8Array {
  return encoder.encode(text)
}

/** A stream that delivers `chunks` one per read. */
export function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk)
      controller.close()
    },
  })
}

/** Split bytes at the given (sorted, unique) offsets. */
export function splitAt(bytes: Uint8Array, offsets: number[]): Uint8Array[] {
  const chunks: Uint8Array[] = []
  let start = 0
  for (const offset of offsets) {
    chunks.push(bytes.subarray(start, offset))
    start = offset
  }
  chunks.push(bytes.subarray(start))
  return chunks
}

export async function collect(stream: ReadableStream<Uint8Array>): Promise<SseMessage[]> {
  const messages: SseMessage[] = []
  for await (const message of parseSse(stream)) messages.push(message)
  return messages
}

export async function fixtureEvents(name: FixtureName): Promise<AgentEvent[]> {
  const messages = await collect(streamOf([bytesOf(FIXTURES[name])]))
  return messages.map(toAgentEvent).filter((e): e is AgentEvent => e !== null)
}
