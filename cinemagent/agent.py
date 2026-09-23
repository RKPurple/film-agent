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
- Citations are title + year (no bracket-number scheme, since one turn may
  combine results from several tools).

Design notes:
- Exact-duplicate tool calls (same name + arguments) are
  short-circuited in the loop with a synthetic error -- a code backstop
  against a model that mis-parses a failure and repeats the same call.
  SYSTEM_PROMPT distinguishes an {"error": ...} result (call failed, don't
  repeat) from an empty list (valid "nothing matches").
- Every run_agent() call appends one untruncated JSON record to
  logs/agent_traces.jsonl (query, model, iterations, exit reason, per-step
  tool/args/result/blocked flag).
- Tuning: SYSTEM_PROMPT forbids passing a film title as
  search_my_history query text -- a literal title match dominates the
  hybrid retrieval's BM25 component and crowds out genuine thematic matches
  (fuzzy_found_family degenerated into a "Guardians of the Galaxy" query).
"""

import json
from datetime import datetime, timezone

from cinemagent.config import AGENT_MAX_ITERATIONS, GEMINI_MODEL, LOGS_DIR
from cinemagent.tools import TOOL_SCHEMAS, call_tool

FALLBACK_MESSAGE = (
    "I wasn't able to resolve this in a reasonable number of steps -- "
    "something about the question may need to be broken down differently."
)


def _log_cache_usage(interaction):
    """Best-effort one-line prompt-cache report. Checks a few plausible
    attribute spellings (the field name isn't documented) and silently
    no-ops if none are present -- must never break a real completion."""
    usage = getattr(interaction, "usage", None) or getattr(interaction, "usage_metadata", None)
    if usage is None:
        return
    cached = getattr(usage, "cached_tokens", None)
    if cached is None:
        cached = getattr(usage, "cached_content_token_count", None)
    if cached is None:
        return
    total = (
        getattr(usage, "total_tokens", None)
        or getattr(usage, "total_token_count", None)
        or getattr(usage, "prompt_tokens", None)
        or getattr(usage, "prompt_token_count", None)
        or 0
    )
    pct = (cached / total * 100) if total else 0
    print(f"  [prompt cache: {cached}/{total} tokens cached ({pct:.0f}%)]")

# Step 15: one JSON record per run_agent() call, appended (never overwritten)
# so history accumulates across runs -- see module docstring.
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
    "Always cite films by title and year. If a tool returns no results, or "
    "search_my_history returns no confident matches, say so plainly instead of "
    "guessing or filling in from outside knowledge -- an honest 'not in your watch "
    "history' is always better than a fabricated answer."
)


def _write_trace_log(query, steps, iterations, exit_reason, answer):
    """Append one JSON record for this run_agent() call. Never raises on a
    logging failure -- a broken trace log shouldn't take down an agent run."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": GEMINI_MODEL,
            "query": query,
            "iterations": iterations,
            "exit_reason": exit_reason,
            "steps": steps,
            "answer": answer,
        }
        with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        print(f"  [trace log write failed: {e}]")


def run_agent(query, client, ctx, max_iterations=AGENT_MAX_ITERATIONS, trace=True, log_trace=True,
              return_trace=False):
    # Step 14: (tool name, canonical-json arguments) pairs already executed
    # this run. First occurrence goes through to call_tool(); any exact
    # repeat is short-circuited with a synthetic error.
    seen_calls = set()
    # Step 15: one entry per tool call (live or duplicate-blocked), flushed
    # to TRACE_LOG_PATH when the run concludes.
    trace_steps = []

    # Stateful mode: first input is the query string; Google holds state
    # from then on, chained via previous_interaction_id.
    previous_interaction_id = None
    current_input = query

    for iteration in range(1, max_iterations + 1):
        interaction = client.interactions.create(
            model=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            input=current_input,
            tools=TOOL_SCHEMAS,
            store=True,
            previous_interaction_id=previous_interaction_id,
        )
        previous_interaction_id = interaction.id
        _log_cache_usage(interaction)

        function_call_steps = [s for s in interaction.steps if getattr(s, "type", None) == "function_call"]

        if not function_call_steps:
            answer = interaction.output_text
            if log_trace:
                _write_trace_log(query, trace_steps, iteration, "final_answer", answer)
            if return_trace:
                return answer, trace_steps
            return answer

        result_items = []
        for fc in function_call_steps:
            name = fc.name
            blocked_duplicate = False
            arguments = dict(fc.arguments) if fc.arguments else {}
            try:
                signature = (name, json.dumps(arguments, sort_keys=True, default=str))
            except TypeError as e:
                # fc.arguments is already a parsed dict, but something in it
                # wasn't serializable for the duplicate-call signature.
                # Treat as a failed call rather than letting it kill the run.
                result = {"error": f"Couldn't canonicalize arguments for duplicate-call tracking: {e}"}
                logged_arguments = arguments
            else:
                logged_arguments = arguments
                if signature in seen_calls:
                    blocked_duplicate = True
                    result = {
                        "error": (
                            "This exact call (same tool, same arguments) was already tried "
                            "earlier in this conversation and will return the same result -- "
                            "do not repeat it. Use a different tool, different arguments, or "
                            "tell the user honestly that this couldn't be resolved."
                        )
                    }
                else:
                    seen_calls.add(signature)
                    result = call_tool(name, arguments, ctx)

            trace_steps.append({
                "tool": name,
                "arguments": logged_arguments,
                "result": result,
                "blocked_duplicate": blocked_duplicate,
            })

            if trace:
                preview = json.dumps(result, default=str)[:200]
                print(f"  [iter {iteration}] {name}({arguments}) -> {preview}")

            result_items.append({
                "type": "function_result",
                "name": name,
                "call_id": fc.id,
                "result": [{"type": "text", "text": json.dumps(result, default=str)}],
            })

        current_input = result_items

    if log_trace:
        _write_trace_log(query, trace_steps, max_iterations, "max_iterations_reached", FALLBACK_MESSAGE)
    if return_trace:
        return FALLBACK_MESSAGE, trace_steps
    return FALLBACK_MESSAGE
