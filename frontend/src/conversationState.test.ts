import { describe, expect, it } from 'vitest'

import { ApiError, TurnInProgressError } from './api'
import {
  actions,
  conversationReducer,
  initialState,
  type ConversationAction,
  type ConversationState,
  type Turn,
} from './conversationState'
import displayRecord from './test/fixtures/display_record.json'
import { fixtureEvents, type FixtureName } from './test/sseFixtures'
import type { AgentEvent, DisplayRecord, DoneEvent, ErrorEvent } from './types'

const record = displayRecord as DisplayRecord

const QUESTIONS: Record<FixtureName, string> = {
  structured: 'What movies with Idris Elba have I watched?',
  vibe: 'What have I watched that felt uneasy or unsettling?',
  followup: 'Which of those did I rate highest?',
  recs: "Recommend something like Parasite that I haven't seen",
}

function reduce(state: ConversationState, ...list: ConversationAction[]): ConversationState {
  return list.reduce(conversationReducer, state)
}

/** turnStarted, every event, streamEnded: one full streamed turn. */
function streamTurn(state: ConversationState, question: string, events: AgentEvent[]): ConversationState {
  return reduce(state, actions.turnStarted(question), ...events.map(actions.event), actions.streamEnded())
}

async function streamFixture(state: ConversationState, name: FixtureName) {
  return streamTurn(state, QUESTIONS[name], await fixtureEvents(name))
}

const last = (state: ConversationState): Turn => state.turns[state.turns.length - 1]

const modelError: ErrorEvent = {
  type: 'error', where: 'model_call', iteration: 1, error_type: 'ServerError', message: '503 overloaded', retryable: true,
}
const doneWith = (overrides: Partial<DoneEvent>): DoneEvent => ({
  type: 'done', exit_reason: 'final_answer', iterations: 1, interaction_id: 'i1', resumable: true, model_calls: 1,
  usage: { input_tokens: 1, output_tokens: 1, thought_tokens: null, cached_tokens: null, tool_use_tokens: null, total_tokens: 2 },
  elapsed_ms: 5, responses: [], error: null, ...overrides,
})

describe('streaming the captured fixtures', () => {
  it('structured: one complete turn with a filter_by_actor step and seven cards', async () => {
    const state = await streamFixture(initialState, 'structured')
    const turn = last(state)
    expect(state.running).toBe(false)
    expect(state.selectedTurn).toBe(0)
    expect(turn.status).toBe('complete')
    expect(turn.exit_reason).toBe('final_answer')
    expect(turn.steps).toHaveLength(1)
    expect(turn.steps[0]).toMatchObject({
      id: 'c1', name: 'filter_by_actor', arguments: { name: 'Idris Elba' }, result_count: 7,
      blocked_duplicate: false, error: null, pending: false,
    })
    expect(turn.steps[0].result).toHaveProperty('films')
    expect(turn.answer).toContain('Thor')
    expect(turn.films.map((f) => f.title)).toEqual([
      'Thor', 'Thor: The Dark World', 'Zootopia', 'Thor: Ragnarok', 'The Suicide Squad', 'Thor: Love and Thunder', 'Zootopia 2',
    ])
    expect(turn.films.every((f) => f.watched && f.poster_url?.startsWith('https://image.tmdb.org/t/p/w342/'))).toBe(true)
    expect(turn.usage?.input_tokens).toBeGreaterThan(0)
    expect(turn.error).toBeNull()
  })

  it('vibe: a search_my_history step; a watched card can have no poster', async () => {
    const turn = last(await streamFixture(initialState, 'vibe'))
    expect(turn.status).toBe('complete')
    // As many matches as the result carries (RERANK_TOP_N, which .env can override).
    const matches = turn.steps[0].result?.matches
    expect(Array.isArray(matches) && matches.length > 0).toBe(true)
    expect(turn.steps.map((s) => [s.name, s.result_count])).toEqual([['search_my_history', (matches as unknown[]).length]])
    expect(turn.films.length).toBeGreaterThan(0)
    expect(turn.films.every((f) => f.watched)).toBe(true)
    expect(turn.films.find((f) => f.title === 'Obsession')).toMatchObject({ poster_url: null, vote_average: 0 })
  })

  it('recs: five TMDB steps and unwatched cards with posters and (where known) vote_average', async () => {
    const turn = last(await streamFixture(initialState, 'recs'))
    expect(turn.status).toBe('complete')
    expect(turn.exit_reason).toBe('final_answer')
    expect(turn.steps.map((s) => s.name)).toEqual([
      'search_tmdb', 'tmdb_recommendations', 'search_tmdb', 'search_tmdb', 'search_tmdb',
    ])
    expect(turn.steps.map((s) => s.result_count)).toEqual([1, 10, 1, 1, 1])
    const unwatched = turn.films.filter((f) => !f.watched)
    expect(unwatched.length).toBeGreaterThan(0)
    expect(unwatched.every((f) => f.poster_url !== null && f.my_rating === null)).toBe(true)
    expect(unwatched.some((f) => typeof f.vote_average === 'number')).toBe(true)
    expect(turn.films.find((f) => f.title === 'Parasite')?.watched).toBe(true)
  })

  it('structured then followup: two complete turns, the follow-up with no steps', async () => {
    const state = await streamFixture(await streamFixture(initialState, 'structured'), 'followup')
    expect(state.turns.map((t) => [t.question, t.status, t.steps.length])).toEqual([
      [QUESTIONS.structured, 'complete', 1],
      [QUESTIONS.followup, 'complete', 0],
    ])
    expect(state.selectedTurn).toBe(1)
    expect(last(state).films.map((f) => f.title)).toEqual(['Thor', 'Zootopia', 'The Suicide Squad'])
  })
})

describe('turn endings', () => {
  it('a stream that ends without done is crashed', async () => {
    const events = (await fixtureEvents('structured')).filter((e) => e.type === 'tool_call' || e.type === 'tool_result')
    const turn = last(streamTurn(initialState, 'q', events))
    expect(turn.status).toBe('crashed')
    expect(turn.exit_reason).toBe('crashed')
    expect(turn.steps).toHaveLength(1)
  })

  it('the server crash path (error where "server", no done) is crashed with that error', () => {
    const serverError: ErrorEvent = {
      type: 'error', where: 'server', iteration: null, error_type: 'RuntimeError', message: 'bug', retryable: false,
    }
    const turn = last(streamTurn(initialState, 'q', [serverError]))
    expect(turn.status).toBe('crashed')
    expect(turn.error).toEqual(serverError)
  })

  it('an error event followed by done is error', () => {
    const turn = last(streamTurn(initialState, 'q', [modelError, doneWith({ exit_reason: 'error', resumable: false, error: modelError })]))
    expect(turn.status).toBe('error')
    expect(turn.exit_reason).toBe('error')
    expect(turn.error).toEqual(modelError)
  })

  it('max_iterations_reached still completes (with the fallback answer)', () => {
    const answer: AgentEvent = { type: 'answer', text: 'fallback', iteration: 6, films: [] }
    const turn = last(streamTurn(initialState, 'q', [answer, doneWith({ exit_reason: 'max_iterations_reached', resumable: false })]))
    expect([turn.status, turn.exit_reason, turn.answer]).toEqual(['complete', 'max_iterations_reached', 'fallback'])
  })

  it('stays running after done until the stream closes', () => {
    const state = reduce(initialState, actions.turnStarted('q'), actions.event(doneWith({})))
    expect(last(state).status).toBe('complete')
    expect(state.running).toBe(true)
    expect(reduce(state, actions.streamEnded()).running).toBe(false)
    expect(last(reduce(state, actions.streamEnded())).status).toBe('complete') // not crashed: done arrived
  })

  it('streamFailed: an abort is aborted, other failures are error with a clientError kind', () => {
    const started = reduce(initialState, actions.turnStarted('q'))
    const aborted = reduce(started, actions.streamFailed(new DOMException('The operation was aborted.', 'AbortError')))
    expect([last(aborted).status, last(aborted).exit_reason, last(aborted).clientError?.kind]).toEqual(['aborted', 'aborted', 'aborted'])
    expect(aborted.running).toBe(false)

    const network = last(reduce(started, actions.streamFailed(new TypeError('Failed to fetch'))))
    expect([network.status, network.exit_reason, network.clientError?.kind]).toEqual(['error', null, 'network'])
    const http = last(reduce(started, actions.streamFailed(new TurnInProgressError())))
    expect([http.status, http.clientError?.kind]).toEqual(['error', 'http'])
    const protocol = last(reduce(started, actions.streamFailed(new ApiError('bad stream', null))))
    expect(protocol.clientError?.kind).toBe('protocol')
  })

  it('streamFailed after done leaves the completed turn alone', () => {
    const state = reduce(initialState, actions.turnStarted('q'), actions.event(doneWith({})), actions.streamFailed(new TypeError('x')))
    expect([last(state).status, last(state).clientError, state.running]).toEqual(['complete', null, false])
  })
})

describe('events for a turn that is not running are ignored', () => {
  it('before any turn, and after the turn has finished', async () => {
    const [toolCall] = await fixtureEvents('structured')
    expect(reduce(initialState, actions.event(toolCall))).toEqual(initialState)
    const finished = await streamFixture(initialState, 'followup')
    expect(reduce(finished, actions.event(toolCall), actions.streamEnded())).toEqual(finished)
    const afterDone = reduce(initialState, actions.turnStarted('q'), actions.event(doneWith({})))
    expect(reduce(afterDone, actions.event(toolCall))).toEqual(afterDone)
  })
})

describe('selectTurn, reset and hydrate', () => {
  it('selectTurn selects valid indexes only', async () => {
    const state = await streamFixture(await streamFixture(initialState, 'structured'), 'followup')
    expect(reduce(state, actions.selectTurn(0)).selectedTurn).toBe(0)
    for (const bad of [-1, 2, 0.5]) expect(reduce(state, actions.selectTurn(bad))).toBe(state)
  })

  it('reset starts an empty conversation, optionally with a new id', async () => {
    const state = await streamFixture(initialState, 'structured')
    expect(reduce(state, actions.reset())).toEqual(initialState)
    expect(reduce(state, actions.reset('abc'))).toEqual({ ...initialState, conversationId: 'abc' })
  })

  it('hydrate replaces the state from a display record', () => {
    const state = reduce(initialState, actions.turnStarted('stale'), actions.hydrate(record))
    expect(state.conversationId).toBe(record.conversation_id)
    expect(state.turns.map((t) => t.question)).toEqual([QUESTIONS.structured, QUESTIONS.followup])
    expect([state.selectedTurn, state.running]).toEqual([1, false])
  })
})

describe('hydrate consistency with the live stream', () => {
  it('hydrate(display_record.json) matches streaming structured.sse + followup.sse, except full results', async () => {
    const live = await streamFixture(
      await streamFixture(reduce(initialState, actions.reset(record.conversation_id)), 'structured'),
      'followup',
    )
    const hydrated = reduce(initialState, actions.hydrate(record))

    // The one field that legitimately differs: full tool results exist only live.
    expect(live.turns[0].steps[0].result).toBeDefined()
    expect(hydrated.turns[0].steps[0].result).toBeUndefined()
    const withoutResults = (state: ConversationState) => ({
      ...state,
      turns: state.turns.map((t) => ({ ...t, steps: t.steps.map(({ result: _result, ...step }) => step) })),
    })
    expect(withoutResults(hydrated)).toEqual(withoutResults(live))

    // Spelled out for the fields the display record carries:
    for (const [i, turn] of hydrated.turns.entries()) {
      const liveTurn = live.turns[i]
      for (const key of ['question', 'status', 'answer', 'films', 'exit_reason', 'usage', 'elapsed_ms', 'error'] as const) {
        expect(turn[key], `turn ${i + 1} ${key}`).toEqual(liveTurn[key])
      }
      expect(turn.steps.map((s) => [s.id, s.name, s.arguments, s.iteration, s.duration_ms, s.result_count, s.blocked_duplicate, s.error]))
        .toEqual(liveTurn.steps.map((s) => [s.id, s.name, s.arguments, s.iteration, s.duration_ms, s.result_count, s.blocked_duplicate, s.error]))
    }
  })
})
