import { useHealth, type HealthLabel } from '../health'

const DOT: Record<HealthLabel, string> = {
  'starting…': 'bg-faint animate-pulse',
  ready: 'bg-marquee shadow-[0_0_10px_2px_rgb(240_180_76/0.55)]',
  busy: 'bg-marquee animate-pulse',
  offline: 'bg-danger',
}

/** A marquee bulb: dim while the server starts, lit when it's ready. */
export function HealthIndicator() {
  const { label, detail } = useHealth()
  return (
    <div
      role="status"
      aria-live="polite"
      title={detail}
      className="flex items-center gap-2 rounded-full border border-line bg-surface-1/80 px-3 py-1 text-xs text-muted"
    >
      <span aria-hidden="true" className={`size-2 rounded-full ${DOT[label]}`} />
      <span>{label}</span>
    </div>
  )
}
