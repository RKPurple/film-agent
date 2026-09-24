"""
Agent loop with tool routing, multi-step chaining,
duplicate-call protection, and persisted trace logging.

How it works:
- Stateful google-genai Interactions API loop: each turn calls
  client.interactions.create() with the system prompt, current input, and
  TOOL_SCHEMAS; store=True + previous_interaction_id chains turns so Google
  holds conversation state server-side. Execute any function_call steps via
  cinemagent.tools' call_tool(), feed results back as the next input, repeat until
  the model returns a plain answer or AGENT_MAX_ITERATIONS (a safety cap,
  from cinemagent.config) hits.
  SYSTEM_PROMPT carries the which-tool-for-which-question guidance.
- run_agent() is a generator that only YIELDS plain-JSON events -- it
  prints nothing and logs nothing, so the same stream can feed the CLI,
  the eval harness, and a server streaming it over SSE:
    tool_call    {id, iteration, index, name, arguments, model_call_id}
    tool_result  {id, iteration, name, result, blocked_duplicate, duration_ms}
    answer       {text, iteration}
    error        {where, iteration, error_type, message, retryable}
    done         {exit_reason, iterations, interaction_id, resumable,
                  model_calls, usage, elapsed_ms, responses, error}
  done is always the last event; exit_reason is final_answer,
  max_iterations_reached or error. Consumers: collect() rebuilds
  (answer, trace_steps, done); with_trace_log() passes events through and
  appends the JSONL trace record when the stream ends.
- Citations are title + year (no bracket-number scheme, since one turn may
  combine results from several tools).

Design notes:
- Exact-duplicate tool calls (same name + arguments) are
  short-circuited in the loop with a synthetic error -- a code backstop
  against a model that mis-parses a failure and repeats the same call.
  SYSTEM_PROMPT distinguishes an {"error": ...} result (call failed, don't
  repeat) from an empty list (valid "nothing matches").
- Errors: tool exceptions are already caught by call_tool() and returned
  to the model as {"error": ...} results. Gemini API errors
  (google.genai.errors.APIError) and network/timeout errors, plus a
  model response with no function calls that is unusable (failed /
  cancelled / incomplete status, or no text), yield an error event and
  then done(exit_reason="error") instead of raising. Anything else is a
  programming error and propagates -- turning a code bug into a polite
  error event would hide it.
- resumable is True only for final_answer: after the iteration cap, an
  error or an abort, the last stored interaction requested tool calls whose
  results were never sent back, so chaining a new turn from it isn't safe.
- Every with_trace_log()-wrapped turn appends one untruncated JSON record
  to logs/agent_traces.jsonl, including early closes (exit_reason
  "aborted") and propagated exceptions ("crashed").
- Tuning: SYSTEM_PROMPT forbids passing a film title as
  search_my_history query text -- a literal title match dominates the
  hybrid retrieval's BM25 component and crowds out genuine thematic matches
  (fuzzy_found_family degenerated into a "Guardians of the Galaxy" query).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from google.genai import errors as genai_errors

from cinemagent.config import AGENT_MAX_ITERATIONS, GEMINI_MODEL, LOGS_DIR
from cinemagent.tools import TOOL_SCHEMAS, call_tool

FALLBACK_MESSAGE = (
    "I wasn't able to resolve this in a reasonable number of steps -- "
    "something about the question may need to be broken down differently."
)

# Step 15: one JSON record per turn, appended (never overwritten) so
# history accumulates across runs -- see module docstring.
TRACE_LOG_PATH = LOGS_DIR / "agent_traces.jsonl"

SYSTEM_PROMPT = (
    "You answer questions about Rohan's personal movie-watching history using ONLY "
    "the tools provided -- never answer from outside knowledge about what he has or "
    "hasn't watched, rated, or reviewed; outside general film knowledge is fine only "
    "when a tool (like search_tmdb) has explicitly supplied it.\n\n"
    "Tool choice: prefer the structured tools (filter_by_actor, filter_by_director, "
    "filter_by_genre, filter_by_rating, cross_reference) for exact, categorical "
    "questions -- a specific actor, director, genre, or rating range. For a "
    "person-based question, default to the single-role tool that matches how the "
    "question names their involvement: 'movies with X', 'X starring in', or asking "
    "about X as a cast member means filter_by_actor; 'X movies', 'directed by X', or "
    "asking about X as director means filter_by_director. Reserve cross_reference for "
    "when the question genuinely doesn't specify a role (e.g. 'what's my history with "
    "X') or explicitly asks about combined acting-and-directing involvement -- most "
    "person-based questions specify a role directly or by convention, and should use "
    "the matching single-role tool even if X might also work in the other role on some "
    "other film. filter_by_genre matches a genre name exactly, not on loose themes -- if you are "
    "unsure whether a genre exists under the name you have in mind, call get_all_genres "
    "first to see the real vocabulary (and per-genre film counts), then filter. "
    "Use search_my_history for vibe/theme/mood questions that don't map to "
    "a rigid field (e.g. 'something hopeful', 'movies with a twist'). When calling "
    "search_my_history, phrase the query in thematic/mood language only -- never "
    "include a specific film title as part of the query text. A literal title match "
    "scores far higher than genuine thematic similarity (the underlying retrieval "
    "rewards lexical overlap), so a title-anchored query crowds out real vibe matches "
    "and narrows the results down to that one title's near-duplicates instead of "
    "surfacing the broader set of films that actually fit the theme. Use search_tmdb "
    "only when the question is plausibly about a film not yet watched -- "
    "recommendations, or asking about a film by name that might not be in the "
    "history. You may call more than one tool in a single turn if the question "
    "genuinely needs it.\n\n"
    "For 'recommend something like X that I haven't seen' style questions specifically: "
    "first call search_tmdb with X's title to resolve it and get its tmdb_id (this also "
    "tells you whether X itself is already_watched, and if so its rating/review are "
    "useful taste signal), then call tmdb_recommendations with that tmdb_id to get "
    "candidate films. Those candidates are NOT pre-filtered -- you must exclude any "
    "tagged already_watched yourself, then rank what's left by fit (genre/vibe from "
    "the overviews, vote_average, and how well it matches what made X appealing) "
    "before presenting a recommendation.\n\n"
    "Tool results come back in two different 'nothing useful' shapes, and they mean "
    "different things. A result containing an 'error' key means the call itself "
    "failed (bad id, missing configuration, malformed input) -- do not repeat that "
    "exact call, since it will fail the same way again; instead fix your arguments, "
    "use a different tool, or tell the user honestly that this couldn't be resolved. "
    "A result with an empty list (e.g. no films found) is a valid, successful answer "
    "meaning nothing matches -- a different tool or slightly different phrasing may be "
    "worth one more attempt, but do not keep repeating the same call hoping for a "
    "different result, and do not fabricate one either way.\n\n"
    "Always cite films by title and year. search_my_history always returns the "
    "closest films in the watch history, ranked -- not confirmed matches, and even "
    "the top result can be the best of a bad set. Decide whether each one actually "
    "fits from its returned text (overview, tone summary, and Rohan's review), not "
    "from its rank, its confidence, or the fact that it was returned; confidence only "
    "compares results within that one search. Cite only the films that genuinely "
    "fit, and if none do, say plainly that nothing in the watch history fits. If a "
    "tool returns no results, likewise say so plainly instead of guessing or filling "
    "in from outside knowledge -- an honest 'not in your watch history' is always "
    "better than a fabricated answer."
)


# done.usage key -> the SDK's Usage attribute, summed over every model call
# in the turn. Input tokens are counted once per call (each iteration
# re-reads the stored context), so the sum is the billed total.
USAGE_FIELDS = {
    "input_tokens": "total_input_tokens",
    "output_tokens": "total_output_tokens",
    "thought_tokens": "total_thought_tokens",
    "cached_tokens": "total_cached_tokens",
    "tool_use_tokens": "total_tool_use_tokens",
    "total_tokens": "total_tokens",
}

# Loop failures that become an error event instead of raising -- see the
# module docstring. httpx is the SDK's transport.
RECOVERABLE_ERRORS = (genai_errors.APIError, httpx.HTTPError, TimeoutError, ConnectionError)

UNUSABLE_STATUSES = {"failed", "cancelled", "incomplete"}

DUPLICATE_CALL_ERROR = (
    "This exact call (same tool, same arguments) was already tried "
    "earlier in this conversation and will return the same result -- "
    "do not repeat it. Use a different tool, different arguments, or "
    "tell the user honestly that this couldn't be resolved."
)


def _elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000)


def _add_usage(totals, interaction):
    """Add one response's token usage into totals; a key stays None until
    the SDK reports it at least once."""
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return
    for key, attr in USAGE_FIELDS.items():
        value = getattr(usage, attr, None)
        if value is not None:
            totals[key] = (totals[key] or 0) + value


def _retryable(e):
    if isinstance(e, genai_errors.ServerError):
        return True
    if isinstance(e, genai_errors.APIError):
        return getattr(e, "code", None) == 429
    return isinstance(e, (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError))


def _error_event(where, iteration, error_type, message, retryable):
    return {"type": "error", "where": where, "iteration": iteration, "error_type": error_type,
            "message": message, "retryable": retryable}


def run_agent(query, client, ctx, max_iterations=AGENT_MAX_ITERATIONS):
    """Run one turn of the agent loop, yielding plain-JSON events (see the
    module docstring). Never closes ctx -- the caller owns it."""
    start = time.perf_counter()
    # Step 14: (tool name, canonical-json arguments) pairs already executed
    # this turn. First occurrence goes through to call_tool(); any exact
    # repeat is short-circuited with a synthetic error.
    seen_calls = set()
    call_count = 0
    model_calls = 0
    usage = dict.fromkeys(USAGE_FIELDS)
    responses = []  # per model response: iteration, interaction_id, status, function_calls

    # Stateful mode: first input is the query string; Google holds state
    # from then on, chained via previous_interaction_id.
    previous_interaction_id = None
    current_input = query

    def done(exit_reason, iterations, error=None):
        return {
            "type": "done",
            "exit_reason": exit_reason,
            "iterations": iterations,
            "interaction_id": previous_interaction_id,
            "resumable": exit_reason == "final_answer",
            "model_calls": model_calls,
            "usage": usage,
            "elapsed_ms": _elapsed_ms(start),
            "responses": responses,
            "error": error,
        }

    for iteration in range(1, max_iterations + 1):
        try:
            interaction = client.interactions.create(
                model=GEMINI_MODEL,
                system_instruction=SYSTEM_PROMPT,
                input=current_input,
                tools=TOOL_SCHEMAS,
                store=True,
                previous_interaction_id=previous_interaction_id,
            )
        except RECOVERABLE_ERRORS as e:
            error = _error_event("model_call", iteration, type(e).__name__, str(e), _retryable(e))
            yield error
            yield done("error", iteration, error)
            return

        model_calls += 1
        previous_interaction_id = interaction.id
        _add_usage(usage, interaction)
        status = getattr(interaction, "status", None)
        status = None if status is None else str(status)
        function_call_steps = [s for s in (interaction.steps or []) if getattr(s, "type", None) == "function_call"]
        responses.append({"iteration": iteration, "interaction_id": interaction.id, "status": status,
                          "function_calls": len(function_call_steps)})

        if not function_call_steps:
            answer = interaction.output_text
            if status in UNUSABLE_STATUSES or not answer:
                detail = f"status={status!r}, output_text={'empty' if not answer else 'present'}"
                if getattr(interaction, "errors", None):
                    detail += f", errors={[str(err) for err in interaction.errors]}"
                error = _error_event("model_response", iteration, "UnusableResponse",
                                     f"Model response had no function calls and no usable answer ({detail})",
                                     status in ("cancelled", "incomplete"))
                yield error
                yield done("error", iteration, error)
                return
            yield {"type": "answer", "text": answer, "iteration": iteration}
            yield done("final_answer", iteration)
            return

        result_items = []
        for index, fc in enumerate(function_call_steps):
            call_count += 1
            call_id = f"c{call_count}"
            name = fc.name
            arguments = dict(fc.arguments) if fc.arguments else {}
            yield {"type": "tool_call", "id": call_id, "iteration": iteration, "index": index,
                   "name": name, "arguments": arguments, "model_call_id": fc.id}

            call_start = time.perf_counter()
            blocked_duplicate = False
            try:
                signature = (name, json.dumps(arguments, sort_keys=True, default=str))
            except TypeError as e:
                # fc.arguments is already a parsed dict, but something in it
                # wasn't serializable for the duplicate-call signature.
                # Treat as a failed call rather than letting it kill the run.
                result = {"error": f"Couldn't canonicalize arguments for duplicate-call tracking: {e}"}
            else:
                if signature in seen_calls:
                    blocked_duplicate = True
                    result = {"error": DUPLICATE_CALL_ERROR}
                else:
                    seen_calls.add(signature)
                    result = call_tool(name, arguments, ctx)

            yield {"type": "tool_result", "id": call_id, "iteration": iteration, "name": name,
                   "result": result, "blocked_duplicate": blocked_duplicate,
                   "duration_ms": _elapsed_ms(call_start)}

            result_items.append({
                "type": "function_result",
                "name": name,
                "call_id": fc.id,
                "result": [{"type": "text", "text": json.dumps(result, default=str)}],
            })

        current_input = result_items

    yield {"type": "answer", "text": FALLBACK_MESSAGE, "iteration": max_iterations}
    yield done("max_iterations_reached", max_iterations)


class _TurnObserver:
    """Accumulates one turn's events into trace steps, answer, error and
    done -- shared by collect() and with_trace_log()."""

    def __init__(self):
        self.calls = {}
        self.steps = []          # trace-log superset of trace_steps
        self.answer = None
        self.error = None
        self.done = None
        self.last_iteration = None

    def observe(self, event):
        kind = event["type"]
        if "iteration" in event:
            self.last_iteration = event["iteration"]
        if kind == "tool_call":
            self.calls[event["id"]] = event
        elif kind == "tool_result":
            call = self.calls[event["id"]]
            self.steps.append({
                "tool": call["name"],
                "arguments": call["arguments"],
                "result": event["result"],
                "blocked_duplicate": event["blocked_duplicate"],
                "id": event["id"],
                "iteration": event["iteration"],
                "index": call["index"],
                "duration_ms": event["duration_ms"],
            })
        elif kind == "answer":
            self.answer = event["text"]
        elif kind == "error":
            self.error = event
        elif kind == "done":
            self.done = event

    def trace_steps(self):
        """Exactly the pre-generator trace_steps shape: four keys per step."""
        return [{"tool": s["tool"], "arguments": s["arguments"], "result": s["result"],
                 "blocked_duplicate": s["blocked_duplicate"]} for s in self.steps]


def collect(events):
    """Drain an event stream into (answer, trace_steps, done). trace_steps
    has exactly the pre-generator shape ({tool, arguments, result,
    blocked_duplicate} per call), so eval grading and stored results are
    unchanged. answer is None if the turn errored; done is None only if
    the stream ended without one."""
    observer = _TurnObserver()
    for event in events:
        observer.observe(event)
    return observer.answer, observer.trace_steps(), observer.done


def _write_trace_log(record, path):
    """Append one JSON record to path. Never raises on a logging failure --
    a broken trace log shouldn't take down an agent run."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        print(f"  [trace log write failed: {e}]")


def with_trace_log(query, events, log_path=TRACE_LOG_PATH):
    """Pass events through unchanged, then append the turn's JSONL trace
    record when the stream ends -- normally, after an error event, when
    the consumer closes the stream early (exit_reason "aborted": closing
    this generator raises GeneratorExit at its yield), or when an exception
    propagates out of run_agent ("crashed"). log_path defaults to
    logs/agent_traces.jsonl; tests pass a temporary path."""
    observer = _TurnObserver()
    start = time.perf_counter()
    reason = None
    try:
        for event in events:
            observer.observe(event)
            yield event
    except GeneratorExit:
        reason = "aborted"
        raise
    except BaseException:
        reason = "crashed"
        raise
    finally:
        # Closing this wrapper doesn't close the inner generator by itself.
        if hasattr(events, "close"):
            events.close()
        done = observer.done or {}
        if reason is None:
            reason = done.get("exit_reason", "aborted")
        _write_trace_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": GEMINI_MODEL,
            "query": query,
            "iterations": done.get("iterations", observer.last_iteration),
            "exit_reason": reason,
            "steps": observer.steps,
            "answer": observer.answer,
            "interaction_id": done.get("interaction_id"),
            "resumable": done.get("resumable", False),
            "usage": done.get("usage"),
            "model_calls": done.get("model_calls"),
            "elapsed_ms": done.get("elapsed_ms", _elapsed_ms(start)),
            "error": observer.error,
            "responses": done.get("responses"),
        }, Path(log_path))
