import { HealthIndicator } from './components/HealthIndicator'

// Placeholder content only -- the chat logic comes in a later step.
const PLACEHOLDER_TURNS = [
  'What movies with Idris Elba have I watched?',
  'Which of those did I rate highest?',
  'Recommend something like Sicario I haven’t seen',
]
const PLACEHOLDER_POSTERS = 6

function Header() {
  return (
    <header
      style={{ gridArea: 'header' }}
      className="flex items-center justify-between gap-4 border-b border-line px-6 py-3"
    >
      <div className="flex items-baseline gap-3">
        <h1 className="font-display text-2xl font-semibold tracking-tight text-text">
          cinem<span className="text-marquee">Agent</span>
        </h1>
        <span className="hidden text-xs tracking-wide text-faint sm:inline">your watch history, on screen</span>
      </div>
      <div className="flex items-center gap-3">
        <HealthIndicator />
        <button
          type="button"
          className="rounded-lg border border-line-strong px-3 py-1.5 text-sm text-text transition-colors hover:border-marquee/60 hover:bg-surface-2"
        >
          New chat
        </button>
      </div>
    </header>
  )
}

function TurnsRail() {
  return (
    <nav
      data-region="rail"
      aria-label="Turns"
      style={{ gridArea: 'rail' }}
      className="flex min-h-0 flex-col gap-3 overflow-hidden border-r border-line px-4 py-5"
    >
      <h2 className="px-2 text-[0.6875rem] font-semibold tracking-[0.18em] text-faint uppercase">Turns</h2>
      <ol className="flex flex-col gap-1">
        {PLACEHOLDER_TURNS.map((question, i) => {
          const active = i === PLACEHOLDER_TURNS.length - 1
          return (
            <li key={question}>
              <button
                type="button"
                aria-current={active ? 'true' : undefined}
                className={`flex w-full items-start gap-3 rounded-lg border-l-2 px-2 py-2 text-left text-sm transition-colors ${
                  active
                    ? 'border-marquee bg-surface-2 text-text'
                    : 'border-transparent text-muted hover:bg-surface-1 hover:text-text'
                }`}
              >
                <span className="font-display text-xs text-faint tabular-nums">{String(i + 1).padStart(2, '0')}</span>
                <span className="line-clamp-2">{question}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

function AnswerPanel() {
  return (
    <section
      aria-label="Answer"
      className="scroll-panel min-h-0 rounded-panel border border-line bg-surface-1/85 px-7 py-6 leading-relaxed text-text/90"
    >
      <p className="mb-4 text-muted">
        The answer will appear here, with the films it cites set as cards alongside. This panel is the only part of
        the page that scrolls.
      </p>
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} className="mb-6 space-y-2.5" aria-hidden="true">
          <div className="h-3 w-11/12 rounded-full bg-surface-3" />
          <div className="h-3 w-full rounded-full bg-surface-3" />
          <div className="h-3 w-4/5 rounded-full bg-surface-3" />
          <div className="h-3 w-2/3 rounded-full bg-surface-3" />
        </div>
      ))}
    </section>
  )
}

function PosterGrid() {
  return (
    <section aria-label="Films" className="flex min-h-0 flex-col overflow-hidden">
      <h3 className="flex h-(--poster-label-height) shrink-0 items-start text-[0.6875rem] font-semibold tracking-[0.18em] text-faint uppercase">
        Films
      </h3>
      <ul className="grid grid-cols-3 content-start gap-3">
        {Array.from({ length: PLACEHOLDER_POSTERS }, (_, i) => (
          <li
            key={i}
            className="relative aspect-2/3 overflow-hidden rounded-md border border-line bg-linear-to-b from-surface-3 to-surface-1"
          >
            <span className="absolute inset-x-0 bottom-0 h-1/3 bg-linear-to-t from-black/50 to-transparent" />
            <span className="absolute bottom-2 left-2 h-1.5 w-1/2 rounded-full bg-line-strong" />
          </li>
        ))}
      </ul>
    </section>
  )
}

function TraceStrip() {
  return (
    <button
      type="button"
      aria-expanded="false"
      className="flex w-full shrink-0 items-center gap-3 truncate rounded-lg border border-line bg-surface-1/70 px-4 py-2 text-left text-xs text-muted transition-colors hover:border-line-strong hover:text-text"
    >
      <span aria-hidden="true" className="text-marquee">▸</span>
      <span className="font-medium tracking-wide">How I got this</span>
      <span className="truncate text-faint">the tools the agent used will appear here</span>
    </button>
  )
}

function Stage() {
  return (
    <main style={{ gridArea: 'stage' }} className="flex min-h-0 min-w-0 flex-col gap-4 px-8 pt-6 pb-4">
      <h2 className="shrink-0 font-display text-3xl leading-tight font-medium text-balance text-text">
        Recommend something like <em className="text-marquee-strong">Sicario</em> I haven’t seen
      </h2>
      <div className="stage-body flex-1">
        <div className="stage-columns">
          <AnswerPanel />
          <PosterGrid />
        </div>
      </div>
      <TraceStrip />
    </main>
  )
}

function InputBar() {
  return (
    <form
      style={{ gridArea: 'input' }}
      onSubmit={(e) => e.preventDefault()}
      className="flex items-center gap-3 border-t border-line bg-bg/80 px-6 py-4"
    >
      <label htmlFor="question" className="sr-only">
        Ask a question
      </label>
      <input
        id="question"
        type="text"
        autoComplete="off"
        placeholder="Ask about your watch history…"
        className="min-w-0 flex-1 rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm text-text placeholder:text-faint focus:border-marquee/60 focus:outline-none"
      />
      <button
        type="submit"
        className="rounded-xl bg-marquee px-5 py-3 text-sm font-semibold text-marquee-ink transition-colors hover:bg-marquee-strong"
      >
        Send
      </button>
    </form>
  )
}

export default function App() {
  return (
    <div className="app-shell">
      <Header />
      <TurnsRail />
      <Stage />
      <InputBar />
    </div>
  )
}
