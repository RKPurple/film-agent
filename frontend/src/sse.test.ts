import { describe, expect, it } from 'vitest'

import { SseParseError } from './sse'
import { bytesOf, collect, FIXTURES, splitAt, streamOf, type FixtureName } from './test/sseFixtures'

const FIXTURE_NAMES = Object.keys(FIXTURES) as FixtureName[]

/** Deterministic PRNG (mulberry32), so random chunkings are reproducible. */
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function randomOffsets(random: () => number, length: number, count: number): number[] {
  const offsets = new Set<number>()
  while (offsets.size < Math.min(count, length - 1)) offsets.add(1 + Math.floor(random() * (length - 1)))
  return [...offsets].sort((a, b) => a - b)
}

const parseText = (text: string, chunks = 1) => {
  const bytes = bytesOf(text)
  const offsets = chunks > 1 ? Array.from({ length: chunks - 1 }, (_, i) => Math.floor(((i + 1) * bytes.length) / chunks)) : []
  return collect(streamOf(splitAt(bytes, offsets)))
}

describe('parseSse on the captured fixtures', () => {
  it.each(FIXTURE_NAMES)('%s parses into named events with object data', async (name) => {
    const messages = await parseText(FIXTURES[name])
    expect(messages.length).toBeGreaterThan(1)
    expect(messages.at(-1)?.event).toBe('done')
    for (const { event, data } of messages) expect(data).toMatchObject({ type: event })
  })

  it.each(FIXTURE_NAMES)('%s: every single split point gives the unsplit parse', async (name) => {
    const bytes = bytesOf(FIXTURES[name])
    const expected = JSON.stringify(await collect(streamOf([bytes])))
    for (let offset = 1; offset < bytes.length; offset++) {
      const got = JSON.stringify(await collect(streamOf(splitAt(bytes, [offset]))))
      if (got !== expected) expect.fail(`${name}: split at byte ${offset} changed the parse`)
    }
  })

  it.each(FIXTURE_NAMES)('%s: random multi-split chunkings (fixed seed) and byte-at-a-time', async (name) => {
    const bytes = bytesOf(FIXTURES[name])
    const expected = JSON.stringify(await collect(streamOf([bytes])))
    const random = mulberry32(0xc1_4e_3a)
    for (let round = 0; round < 150; round++) {
      const offsets = randomOffsets(random, bytes.length, 2 + Math.floor(random() * 60))
      const got = JSON.stringify(await collect(streamOf(splitAt(bytes, offsets))))
      if (got !== expected) expect.fail(`${name}: chunking at [${offsets.join(', ')}] changed the parse`)
    }
    const everyByte = Array.from({ length: bytes.length - 1 }, (_, i) => i + 1)
    expect(JSON.stringify(await collect(streamOf(splitAt(bytes, everyByte))))).toBe(expected)
  })
})

describe('parseSse edge cases', () => {
  it('decodes a multi-byte character split across chunks at every offset', async () => {
    const text = 'event: answer\ndata: {"text":"🎬 Amélie — ok"}\n\n'
    const bytes = bytesOf(text)
    for (let offset = 1; offset < bytes.length; offset++) {
      const [message] = await collect(streamOf(splitAt(bytes, [offset])))
      expect(message).toEqual({ event: 'answer', data: { text: '🎬 Amélie — ok' } })
    }
  })

  it('accepts CRLF and bare CR line endings, including a CRLF split between chunks', async () => {
    const lf = await parseText(FIXTURES.structured)
    const crlfText = FIXTURES.structured.replaceAll('\n', '\r\n')
    expect(await parseText(crlfText)).toEqual(lf)
    expect(await parseText(FIXTURES.structured.replaceAll('\n', '\r'))).toEqual(lf)
    const bytes = bytesOf(crlfText)
    for (let offset = 1; offset < bytes.length; offset++) {
      if (bytes[offset - 1] === 0x0d) expect(await collect(streamOf(splitAt(bytes, [offset])))).toEqual(lf)
    }
  })

  it('ignores comment lines, between and inside events', async () => {
    const text = ': ping\n\nevent: done\n: a comment mid-event\ndata: {"type":"done"}\n\n: ping\n\n'
    expect(await parseText(text, 5)).toEqual([{ event: 'done', data: { type: 'done' } }])
  })

  it('joins multiple data lines with "\\n"', async () => {
    const text = 'event: answer\ndata: {"type":"answer",\ndata:"text":\ndata: "a"}\n\n'
    expect(await parseText(text)).toEqual([{ event: 'answer', data: { type: 'answer', text: 'a' } }])
    const multiLineString = 'data: "x\ndata: y"\n\n' // the joined JSON contains a raw newline: invalid
    await expect(parseText(multiLineString)).rejects.toThrow(SseParseError)
  })

  it('discards an incomplete event at end of stream', async () => {
    const complete = 'event: tool_call\ndata: {"n":1}\n\n'
    expect(await parseText(complete + 'event: done\ndata: {"n":2}')).toEqual([{ event: 'tool_call', data: { n: 1 } }])
    expect(await parseText(complete + 'event: done\ndata: {"n":2}\n')).toEqual([{ event: 'tool_call', data: { n: 1 } }])
  })

  it('defaults the event name to "message" and skips events without data', async () => {
    expect(await parseText('event: nothing\n\ndata: {"a":1}\n\n')).toEqual([{ event: 'message', data: { a: 1 } }])
  })

  it('throws a clear error naming the event for malformed JSON', async () => {
    const text = 'event: tool_call\ndata: {"ok":true}\n\nevent: answer\ndata: {"text": oops}\n\n'
    const error = await parseText(text).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(SseParseError)
    expect((error as SseParseError).event).toBe('answer')
    expect((error as SseParseError).message).toMatch(/^Malformed JSON in the data of SSE event "answer": /)
  })
})
