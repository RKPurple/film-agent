"""
Recompute eval ground truth from Postgres and diff it against eval_questions.json.

Every question with a stored ground_truth list also carries a
ground_truth_call: the exact tool call (or, for auto_intersection, the list
of calls) that produces it. This script runs each one through
cinemagent.tools.call_tool() -- the same path the agent's tools take -- and
builds the film set with run_eval.py's own grade_question(), so the result
is exactly what the grader would accept as a correct answer.

Questions are selected by having a ground_truth_call, NOT by grading mode:
a manually graded question can keep one to check its data premise (e.g.
adversarial_documentaries: "no Documentary genre exists" must stay true).
Those are recomputed as an exact set from their single ground_truth_call.

Re-run this after EVERY dataset refresh (new Letterboxd export -> pipeline
-> load_postgres.py). Stored ground truth is a fixed film list, so newly
watched or re-rated films would otherwise make correct agent answers fail.

Review the diff by hand before passing --write. The recomputed ground truth
comes from the same cinemagent.queries functions the tools call, so it can't
catch a bug in those functions: if filter_by_actor were wrong, the expected
answer would be wrong in exactly the same way and the eval would still pass.
Reading the added/removed films is the only check on that.

Usage:
    python3 eval/refresh_ground_truth.py           # dry run: print diffs, exit 1 if anything differs
    python3 eval/refresh_ground_truth.py --write   # save recomputed ground truth for changed questions
"""

import argparse
import json
import sys

from cinemagent import queries
from cinemagent.tools import ToolContext, call_tool

from run_eval import QUESTIONS_PATH, _ground_truth_set, grade_question


def dump_questions(questions):
    """Serialize in eval_questions.json's hand-kept layout: one key per line,
    scalars and short lists inline, and one compact object per line inside
    lists of objects (ground_truth, multi-call ground_truth_call)."""
    def value(v):
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return "[\n" + ",\n".join(f"      {json.dumps(x, ensure_ascii=False)}" for x in v) + "\n    ]"
        return json.dumps(v, ensure_ascii=False)

    blocks = [
        "  {\n" + ",\n".join(f"    {json.dumps(k)}: {value(v)}" for k, v in q.items()) + "\n  }"
        for q in questions
    ]
    return "[\n" + ",\n".join(blocks) + "\n]\n"


# Grading modes whose grade_question() branch builds a film set from the
# ground_truth_call results. Any other mode (e.g. manual) is recomputed as
# an exact set -- see the module docstring.
GROUND_TRUTH_MODES = {"auto_exact_set", "auto_empty", "auto_intersection", "auto_genre_rating_threshold"}


def recompute(question, ctx):
    """Run the question's ground_truth_call(s) and return the film set the
    grader builds from them, as {(film_id, title, year), ...}."""
    calls = question["ground_truth_call"]
    if isinstance(calls, dict):
        calls = [calls]
    if question["grading"] not in GROUND_TRUTH_MODES:
        if len(calls) != 1:
            raise RuntimeError(f"{question['grading']} question needs exactly one ground_truth_call to recompute")
        question = {**question, "grading": "auto_exact_set", "expected_tools": [calls[0]["tool"]]}

    trace_steps = []
    for call in calls:
        result = call_tool(call["tool"], call["args"], ctx)
        if "error" in result:
            raise RuntimeError(f"{call['tool']}({call['args']}) returned an error: {result['error']}")
        trace_steps.append({"tool": call["tool"], "arguments": call["args"],
                            "result": result, "blocked_duplicate": False})

    grade = grade_question(question, trace_steps)
    if "got" not in grade:
        raise RuntimeError(f"grader couldn't build a film set: {grade.get('detail')}")
    return set(grade["got"])


def describe(films):
    return [f"{title} ({year})" for _fid, title, year in sorted(films, key=lambda t: (t[2] or 0, t[1] or ""))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true",
                        help="Save recomputed ground truth for every changed question back into eval_questions.json")
    args = parser.parse_args()

    raw = QUESTIONS_PATH.read_text(encoding="utf-8")
    questions = json.loads(raw)
    targets = [q for q in questions if "ground_truth_call" in q]

    # Structured tools only touch ctx.conn -- no retrieval index, no models.
    ctx = ToolContext(queries.get_connection(), index=None)
    changed, errors = {}, []
    try:
        for q in targets:
            try:
                new = recompute(q, ctx)
            except RuntimeError as e:
                errors.append(q["id"])
                print(f"{q['id']:34s} ERROR: {e}")
                continue
            old = _ground_truth_set(q["ground_truth"])
            if new == old:
                print(f"{q['id']:34s} unchanged ({len(new)} film(s))")
                continue
            changed[q["id"]] = new
            print(f"{q['id']:34s} CHANGED ({len(old)} -> {len(new)} film(s))")
            for film in describe(new - old):
                print(f"    + {film}")
            for film in describe(old - new):
                print(f"    - {film}")
    finally:
        ctx.conn.close()

    print(f"\n{len(targets)} question(s) checked: {len(targets) - len(changed) - len(errors)} unchanged, "
          f"{len(changed)} changed, {len(errors)} error(s).")

    if errors:
        sys.exit(2)
    if not changed:
        return
    if not args.write:
        print("Review the diff above, then re-run with --write to save it.")
        sys.exit(1)

    if dump_questions(questions) != raw:
        sys.exit(f"ERROR: {QUESTIONS_PATH.name} isn't in the layout dump_questions() writes -- "
                 "refusing to rewrite it and lose its formatting.")
    for q in questions:
        if q["id"] in changed:
            q["ground_truth"] = [
                {"film_id": fid, "title": title, "year": year}
                for fid, title, year in sorted(changed[q["id"]], key=lambda t: (t[2] or 0, t[1] or ""))
            ]
    QUESTIONS_PATH.write_text(dump_questions(questions), encoding="utf-8")
    print(f"Wrote {len(changed)} updated ground_truth list(s) to {QUESTIONS_PATH}.")


if __name__ == "__main__":
    main()
