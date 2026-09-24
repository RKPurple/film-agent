/**
 * Wire types for the cinemagent API, derived from the backend source:
 * - SSE events: cinemagent/agent.py run_agent() (tool_call, tool_result,
 *   answer, error, done), plus server/app.py, which adds `films` to answer
 *   and emits the crash-path error (where "server").
 * - FilmCard: cinemagent/film_cards.py watched_card() / unwatched_card().
 * - Create and display records: server/app.py (create_conversation,
 *   ConversationState.display(), Turn.record / Turn._apply()).
 *
 * Nullability follows what the backend actually produces; each `| null`
 * notes when the value is null.
 */

/** Any JSON value, e.g. tool arguments and results. */
export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type JsonObject = { [key: string]: Json }

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

/** done.exit_reason (agent.py). Only final_answer and max_iterations_answered
 * are resumable. */
export type ExitReason = 'final_answer' | 'max_iterations_answered' | 'max_iterations_reached' | 'error'

/** error.where: "model_call" (create() raised), "model_response" (unusable
 * response), "server" (server/app.py crash path: an exception propagated
 * out of the agent). */
export type ErrorWhere = 'model_call' | 'model_response' | 'server'

/** done.usage: token totals summed over the turn's model calls. A key stays
 * null until the SDK reports it at least once (dict.fromkeys(..., None));
 * all six are null if no model call succeeded. */
export interface Usage {
  input_tokens: number | null
  output_tokens: number | null
  thought_tokens: number | null
  cached_tokens: number | null
  tool_use_tokens: number | null
  total_tokens: number | null
}

/** One entry of done.responses: one per model response in the turn. */
export interface ModelResponseSummary {
  iteration: number
  interaction_id: string
  /** str(interaction.status), e.g. "requires_action", "completed"; null if the SDK gave none. */
  status: string | null
  function_calls: number
}

/** A cited film (film_cards.py). Watched films come from the corpus,
 * unwatched ones from search_tmdb / tmdb_recommendations results. */
export interface FilmCard {
  tmdb_id: number
  title: string
  /** null only if the corpus year isn't an integer (never cited for unwatched films without a year). */
  year: number | null
  watched: boolean
  /** Letterboxd rating 0-5: watched films only (null if unrated); always null for unwatched. */
  my_rating: number | null
  /** TMDB audience score 0-10. null for an unwatched film only seen via search_tmdb.
   * Can be 0 for a watched film TMDB has no votes for. */
  vote_average: number | null
  /** Full TMDB image URL, or null when there's no poster_path. */
  poster_url: string | null
  overview: string | null
}

// ---------------------------------------------------------------------------
// SSE events (POST /api/conversations/{id}/messages)
// ---------------------------------------------------------------------------

export interface ToolCallEvent {
  type: 'tool_call'
  /** Call id within the turn: "c1", "c2", ... */
  id: string
  iteration: number
  /** Position within its model response (parallel calls share an iteration). */
  index: number
  name: string
  arguments: JsonObject
  /** Gemini's own function-call id. */
  model_call_id: string
}

export interface ToolResultEvent {
  type: 'tool_result'
  id: string
  iteration: number
  name: string
  /** Always an object: the tool's result, or {"error": "..."} for a failed or blocked call. */
  result: JsonObject
  blocked_duplicate: boolean
  duration_ms: number
}

export interface AnswerEvent {
  type: 'answer'
  text: string
  iteration: number
  /** Added by server/app.py: the films the answer cites, in order of first appearance. */
  films: FilmCard[]
}

export interface ErrorEvent {
  type: 'error'
  where: ErrorWhere
  /** null for where "server". */
  iteration: number | null
  error_type: string
  message: string
  retryable: boolean
}

export interface DoneEvent {
  type: 'done'
  exit_reason: ExitReason
  iterations: number
  /** This turn's latest response id; null if no model call succeeded. */
  interaction_id: string | null
  resumable: boolean
  model_calls: number
  usage: Usage
  elapsed_ms: number
  responses: ModelResponseSummary[]
  /** The turn's error event when exit_reason is "error" (a model_call or
   * model_response error); null otherwise, including max_iterations_reached. */
  error: ErrorEvent | null
}

export type AgentEvent = ToolCallEvent | ToolResultEvent | AnswerEvent | ErrorEvent | DoneEvent
export type AgentEventType = AgentEvent['type']

// ---------------------------------------------------------------------------
// REST responses
// ---------------------------------------------------------------------------

/** POST /api/conversations -> 201 */
export interface CreateConversationResponse {
  conversation_id: string
  /** ISO 8601, UTC. */
  created_at: string
}

/** Display-record status. "running" until the turn ends; then "complete"
 * (done arrived, exit_reason not "error"), "error" (done with exit_reason
 * "error"), "crashed" (server crash path) or "aborted" (the client
 * disconnected before the answer existed, or the server shut down). */
export type TurnStatus = 'running' | 'complete' | 'error' | 'aborted' | 'crashed'

/** One compact trace step in a display record (full results are omitted). */
export interface DisplayStep {
  id: string
  iteration: number
  tool: string
  arguments: JsonObject
  duration_ms: number
  blocked_duplicate: boolean
  /** result.error for a failed or blocked call, else null. */
  error: string | null
  /** Length of the result's list (1 for search_tmdb); null for an error or an unrecognized shape. */
  result_count: number | null
}

export interface DisplayTurn {
  /** 1-based turn number within the conversation. */
  turn: number
  question: string
  started_at: string
  status: TurnStatus
  /** done.exit_reason; "aborted"/"crashed" for those statuses; null while running. */
  exit_reason: ExitReason | 'aborted' | 'crashed' | null
  /** null until (and unless) an answer event arrives. */
  answer: string | null
  films: FilmCard[]
  steps: DisplayStep[]
  /** The turn's last error event, or null. */
  error: ErrorEvent | null
  /** null until done arrives. */
  usage: Usage | null
  /** null until done arrives. */
  elapsed_ms: number | null
}

/** GET /api/conversations/{id} -> 200 */
export interface DisplayRecord {
  conversation_id: string
  created_at: string
  /** A turn is in progress (it will be the last one, with status "running"). */
  running: boolean
  turns: DisplayTurn[]
}

/** FastAPI's 422 body: {"detail": [...]} with one entry per validation error. */
export interface ValidationErrorDetail {
  type: string
  loc: (string | number)[]
  msg: string
  input?: Json
  ctx?: JsonObject
}
