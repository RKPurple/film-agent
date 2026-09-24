import argparse
import json
import os
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

from google import genai

from cinemagent.agent import run_agent
from cinemagent.citations import film_label, resolve_citations
from cinemagent.config import ENRICHED_JSON, GEMINI_MODEL
from cinemagent.tools import build_tool_context, close_tool_context

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
RESULTS_DIR = EVAL_DIR / "results"

# Schema knowledge about cinemagent.tools' dispatch functions -- which key in
# each tool's JSON result holds the film list to grade against. See module
# docstring.
RESULT_KEYS = {
    "filter_by_actor": "films",
    "filter_by_director": "films",
    "filter_by_genre": "films",
    "filter_by_rating": "films",
    "cross_reference": "credits",
}


def _film_set_from_result(tool_name, result):
    """(film_id, title, year) tuples from a tool's raw JSON result, or None
    if the result doesn't have the expected shape (e.g. an {"error": ...}
    dict from a failed call)."""
    key = RESULT_KEYS.get(tool_name)
    if not key or not isinstance(result, dict) or key not in result:
        return None
    return {(r.get("film_id"), r.get("title"), r.get("year")) for r in result[key]}


def _ground_truth_set(ground_truth):
    return {(g.get("film_id"), g.get("title"), g.get("year")) for g in ground_truth}


@cache
def _watched_corpus():
    """Every watched film (tmdb_id, title, year), for citation grading."""
    with open(ENRICHED_JSON, encoding="utf-8") as f:
        return tuple({"tmdb_id": film["tmdb_id"], "title": film["title"], "year": film.get("year")}
                     for film in json.load(f))


def _live_calls(trace_steps, tool_name):
    """Non-duplicate-blocked calls to `tool_name` this run, in order. Only
    live calls actually hit the tool and produced a real result -- a
    blocked duplicate's result is the synthetic step-14 error, not data."""
    return [s for s in trace_steps if s["tool"] == tool_name and not s["blocked_duplicate"]]


def grade_question(question, trace_steps, answer=None):
    expected_tools = question.get("expected_tools") or []
    called_tools = {s["tool"] for s in trace_steps if not s["blocked_duplicate"]}
    routing_correct = all(t in called_tools for t in expected_tools) if expected_tools else None

    mode = question["grading"]

    if mode in ("auto_exact_set", "auto_empty"):
        tool = expected_tools[0]
        calls = _live_calls(trace_steps, tool)
        if not calls:
            return {"routing_correct": routing_correct, "answer_correct": False,
                    "detail": f"{tool} was never called"}
        got = _film_set_from_result(tool, calls[-1]["result"])
        if got is None:
            return {"routing_correct": routing_correct, "answer_correct": False,
                    "detail": f"{tool}'s result had no usable film list (likely an error result)"}
        expected = _ground_truth_set(question["ground_truth"])
        return {
            "routing_correct": routing_correct,
            "answer_correct": got == expected,
            "got": sorted(got, key=lambda t: (t[2] or 0, t[1] or "")),
            "expected": sorted(expected, key=lambda t: (t[2] or 0, t[1] or "")),
        }

    if mode == "auto_intersection":
        tool_a, tool_b = expected_tools[0], expected_tools[1]
        calls_a = _live_calls(trace_steps, tool_a)
        calls_b = _live_calls(trace_steps, tool_b)
        if not calls_a or not calls_b:
            return {"routing_correct": routing_correct, "answer_correct": False,
                    "detail": "one or both of the expected chained calls never happened"}
        set_a = _film_set_from_result(tool_a, calls_a[-1]["result"])
        set_b = _film_set_from_result(tool_b, calls_b[-1]["result"])
        if set_a is None or set_b is None:
            return {"routing_correct": routing_correct, "answer_correct": False,
                    "detail": "one or both chained calls returned an unusable result"}
        got = set_a & set_b
        expected = _ground_truth_set(question["ground_truth"])
        return {
            "routing_correct": routing_correct,
            "answer_correct": got == expected,
            "got": sorted(got, key=lambda t: (t[2] or 0, t[1] or "")),
            "expected": sorted(expected, key=lambda t: (t[2] or 0, t[1] or "")),
        }

    if mode == "auto_genre_rating_threshold":
        tool = expected_tools[0]
        calls = _live_calls(trace_steps, tool)
        if not calls:
            return {"routing_correct": routing_correct, "answer_correct": False,
                    "detail": f"{tool} was never called"}
        result = calls[-1]["result"]
        key = RESULT_KEYS.get(tool)
        if not key or not isinstance(result, dict) or key not in result:
            return {"routing_correct": routing_correct, "answer_correct": False,
                    "detail": f"{tool}'s result had no usable film list (likely an error result)"}
        min_rating = question["min_rating"]
        got = {
            (r.get("film_id"), r.get("title"), r.get("year"))
            for r in result[key]
            if (r.get("my_rating") or 0) >= min_rating
        }
        expected = _ground_truth_set(question["ground_truth"])
        return {
            "routing_correct": routing_correct,
            "answer_correct": got == expected,
            "got": sorted(got, key=lambda t: (t[2] or 0, t[1] or "")),
            "expected": sorted(expected, key=lambda t: (t[2] or 0, t[1] or "")),
        }

    if mode == "auto_duplicate_guard":
        triggered = any(s["blocked_duplicate"] for s in trace_steps)
        return {
            "routing_correct": routing_correct,
            "answer_correct": triggered,
            "detail": (
                "duplicate-call guard fired as expected" if triggered
                else "no blocked duplicate seen -- either the model never repeated the "
                     "call (also acceptable) or the guard didn't catch a repeat that happened"
            ),
        }

    if mode == "auto_no_watched_citations":
        # Graded on the final answer, not tool results: search_my_history
        # always returns its closest films, so the question is whether the
        # agent presented any watched film as a match. Films outside the
        # watched corpus (e.g. TMDB suggestions) don't count.
        #
        # Currently unused: it can't distinguish citing a film AS a match
        # from naming it to rule it out ("X is not a werewolf film"), which
        # is acceptable behavior -- so the adversarial "nothing fits"
        # questions are graded manually. An LLM-as-judge grader is the
        # planned replacement.
        cited = resolve_citations(answer or "", _watched_corpus())
        return {
            "routing_correct": routing_correct,
            "answer_correct": not cited,
            "cited_watched_films": [film_label(f) for f in cited],
            "detail": ("no watched film cited" if not cited
                       else f"cited {len(cited)} watched film(s): {', '.join(film_label(f) for f in cited)}"),
        }

    # mode == "manual"
    return {"routing_correct": routing_correct, "answer_correct": None,
            "detail": "no ground truth -- grade manual_score by hand"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=str, default=None,
                         help="Comma-separated question ids to run (default: all). Takes priority over --category.")
    parser.add_argument("--category", type=str, default=None, choices=["checkable", "fuzzy", "adversarial"],
                         help="Only run questions in this category (default: all)")
    parser.add_argument("--list", action="store_true",
                         help="Print every question id, category and grading mode, then exit without running anything")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    if args.list:
        for q in questions:
            print(f"{q['id']:35s} {q['category']:12s} {q['grading']}")
        return

    filtered = bool(args.ids or args.category)
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        by_id = {q["id"]: q for q in questions}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            raise SystemExit(f"ERROR: unknown question id(s): {', '.join(unknown)} (see --list for valid ids)")
        questions = [by_id[i] for i in wanted]
    elif args.category:
        questions = [q for q in questions if q["category"] == args.category]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set the GEMINI_API_KEY environment variable first (see module docstring).")
    tmdb_api_key = os.environ.get("TMDB_API_KEY")

    suffix = " (filtered)" if filtered else ""
    print(f"Loaded {len(questions)} eval question(s) from {QUESTIONS_PATH}{suffix}")
    print("Building tool context (loading models, opening DB connection)...")
    ctx = build_tool_context(tmdb_api_key=tmdb_api_key)
    client = genai.Client(api_key=api_key)

    entries = []
    try:
        for q in questions:
            print(f"\n=== [{q['id']}] ({q['category']}) {q['query']}")
            try:
                answer, trace_steps = run_agent(
                    q["query"], client, ctx, trace=False, log_trace=True, return_trace=True
                )
            except Exception as e:
                # A single question's failure (e.g. agent.py's own retry
                # budget for LLM API rate limiting exhausted, or anything
                # else unexpected) shouldn't cost the results already
                # collected
                # for every question run before it -- record this one as
                # failed and move on, rather than letting one bad question
                # take the whole run down with it.
                print(f"  -> ERROR: {type(e).__name__}: {e}")
                entries.append({
                    "id": q["id"],
                    "category": q["category"],
                    "query": q["query"],
                    "answer": None,
                    "tool_call_count": 0,
                    "trace_steps": [],
                    "grade": {
                        "routing_correct": None,
                        "answer_correct": None,
                        "detail": f"question raised {type(e).__name__}: {e}",
                    },
                    "manual_score": None,
                })
                continue

            grade = grade_question(q, trace_steps, answer)

            if grade["answer_correct"] is True:
                status = "PASS"
            elif grade["answer_correct"] is False:
                status = "FAIL"
            else:
                status = "MANUAL"
            print(f"  -> {status}  (routing_correct={grade['routing_correct']})")
            print(f"  answer: {answer}")

            entries.append({
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "answer": answer,
                "tool_call_count": len(trace_steps),
                "trace_steps": trace_steps,
                "grade": grade,
                "manual_score": None,  # fill in by hand for MANUAL entries after the run
            })
    finally:
        close_tool_context(ctx)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"run_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": GEMINI_MODEL, "timestamp": timestamp, "results": entries}, f, indent=2, default=str)

    auto_graded = [e for e in entries if e["grade"]["answer_correct"] is not None]
    n_pass = sum(1 for e in auto_graded if e["grade"]["answer_correct"])
    routing_checked = [e for e in entries if e["grade"]["routing_correct"] is not None]
    n_routing_correct = sum(1 for e in routing_checked if e["grade"]["routing_correct"])
    avg_calls = sum(e["tool_call_count"] for e in entries) / len(entries)
    n_manual = len(entries) - len(auto_graded)

    print("\n=== SUMMARY ===")
    print(f"Auto-graded: {n_pass}/{len(auto_graded)} answer-correct")
    print(f"Routing correct: {n_routing_correct}/{len(routing_checked)}")
    print(f"Avg tool calls per question: {avg_calls:.1f}")
    print(f"{n_manual} question(s) need manual_score filled in by hand (fuzzy/adversarial).")
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
