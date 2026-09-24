import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HealthResponse } from '../api'
import { HealthIndicator } from './HealthIndicator'

const OK: HealthResponse = { status: 'ok', model: 'gemini-3.6-flash', films: 174, tmdb: true, db: 'ok' }

function mockFetch(impl: () => Promise<Response>) {
  const fetchMock = vi.fn(impl)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('HealthIndicator', () => {
  it('shows "ready" for an ok health response', async () => {
    const fetchMock = mockFetch(async () => Response.json(OK))
    render(<HealthIndicator />)

    expect(screen.getByRole('status')).toHaveTextContent('starting…')
    expect(await screen.findByText('ready')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveAttribute('title', '174 films · gemini-3.6-flash')
    expect(fetchMock).toHaveBeenCalledWith('/api/health', expect.anything())
  })

  it('stays on "starting…" while the server is unreachable', async () => {
    const fetchMock = mockFetch(async () => {
      throw new TypeError('Failed to fetch')
    })
    render(<HealthIndicator />)

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await vi.waitFor(() =>
      expect(screen.getByRole('status')).toHaveAttribute('title', 'Waiting for the server to finish loading…'),
    )
    expect(screen.getByRole('status')).toHaveTextContent('starting…')
  })

  it('shows "offline" when the database is failing', async () => {
    mockFetch(async () => Response.json({ ...OK, status: 'degraded', db: 'OperationalError: down' }, { status: 503 }))
    render(<HealthIndicator />)

    expect(await screen.findByText('offline')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveAttribute('title', 'Database unavailable: OperationalError: down')
  })

  it('shows "busy" when every connection is serving a turn', async () => {
    mockFetch(async () => Response.json({ ...OK, db: 'busy' }))
    render(<HealthIndicator />)

    expect(await screen.findByText('busy')).toBeInTheDocument()
  })
})
