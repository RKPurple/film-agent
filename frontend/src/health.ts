import { useEffect, useState } from 'react'

import { fetchHealth, type HealthCheck } from './api'

export type HealthLabel = 'starting…' | 'ready' | 'busy' | 'offline'

export interface HealthState {
  label: HealthLabel
  /** Longer description, for a tooltip. */
  detail: string
}

/** Poll every second until the API is up, then every 15 seconds. */
export const FAST_POLL_MS = 1_000
export const SLOW_POLL_MS = 15_000

export const INITIAL_HEALTH: HealthState = { label: 'starting…', detail: 'Connecting to the server…' }

/**
 * What the indicator shows for one check:
 * - unreachable: "starting…" until the server has been ready once in this
 *   page session (uvicorn doesn't accept connections while it loads the
 *   models, about 10s), "offline" after that.
 * - status ok: "ready", or "busy" when every DB connection is serving a turn.
 * - status degraded (503): "offline", since the database is failing.
 */
export function describeHealth(check: HealthCheck, wasReady: boolean): HealthState {
  if (!check.reachable) {
    return wasReady
      ? { label: 'offline', detail: `Can't reach the server (${check.error})` }
      : { label: 'starting…', detail: 'Waiting for the server to finish loading…' }
  }
  const { health } = check
  if (health.status === 'degraded') {
    return { label: 'offline', detail: `Database unavailable: ${health.db}` }
  }
  const summary = `${health.films} films · ${health.model}${health.tmdb ? '' : ' · TMDB off'}`
  if (health.db === 'busy') return { label: 'busy', detail: `Busy with other questions · ${summary}` }
  return { label: 'ready', detail: summary }
}

export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>(INITIAL_HEALTH)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    let wasReady = false

    async function poll() {
      let next: HealthState
      try {
        next = describeHealth(await fetchHealth(controller.signal), wasReady)
      } catch {
        return // aborted: the component unmounted
      }
      const up = next.label === 'ready' || next.label === 'busy'
      wasReady ||= up
      setState(next)
      timer = setTimeout(poll, up ? SLOW_POLL_MS : FAST_POLL_MS)
    }

    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])

  return state
}
