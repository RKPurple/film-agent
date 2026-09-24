import { describe, expect, it } from 'vitest'

import { describeHealth } from './health'

describe('describeHealth', () => {
  it('reports an unreachable server as starting until it has been ready, then offline', () => {
    const down = { reachable: false as const, error: 'HTTP 502' }
    expect(describeHealth(down, false).label).toBe('starting…')
    expect(describeHealth(down, true).label).toBe('offline')
  })
})
