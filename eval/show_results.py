"""Pretty-print an eval run JSON (eval/results/run_*.json).

Usage:
    python eval/show_results.py                 # newest run in eval/results/
    python eval/show_results.py <path-or-name>  # a specific run file
    python eval/show_results.py --traces        # also dump the tool-call trace
"""

import sys
import glob
import json
import os
import textwrap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def find_run(arg):
    if arg and os.path.isfile(arg):
        return arg
    if arg:
        cand = os.path.join(RESULTS_DIR, arg)
        if os.path.isfile(cand):
            return cand
        matches = glob.glob(os.path.join(RESULTS_DIR, f"*{arg}*"))
        if matches:
            return max(matches, key=os.path.getmtime)
        sys.exit(f"no run file matching {arg!r}")
    runs = glob.glob(os.path.join(RESULTS_DIR, "run_*.json"))
    if not runs:
        sys.exit(f"no run_*.json files in {RESULTS_DIR}")
    return max(runs, key=os.path.getmtime)


FREE_ROUTING = "n/a (free routing)"
NO_ANSWER = "(no answer: error)"  # run_eval stores answer None when a question raised
EXCERPT_CHARS = 240  # per-turn answer excerpt for multi-turn questions


def excerpt(answer, limit=EXCERPT_CHARS):
    if answer is None:
        return NO_ANSWER
    flat = " ".join(answer.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def print_turns(r):
    """Multi-turn question, turn by turn: query, tools, exit_reason (any
    value, e.g. max_iterations_answered, is shown as-is), answer excerpt."""
    for i, turn in enumerate(r["turns"], 1):
        done = turn.get("done") or {}
        tools = " -> ".join(st["tool"] for st in turn.get("trace_steps", [])) or "(none)"
        print(f"  turn {i}: {turn['query']}")
        print(f"    tools: {tools}   exit: {done.get('exit_reason', '?')}")
        print(textwrap.indent(textwrap.fill(excerpt(turn.get("answer")), 88), "    A: "))


def fmt_bool(v):
    return {True: "PASS", False: "FAIL", None: " -- "}[v]


def fmt_routing(v):
    """routing_correct is None exactly when the question has expected_tools
    null (routing left to the agent) -- not a routing failure."""
    return FREE_ROUTING if v is None else fmt_bool(v)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_traces = "--traces" in sys.argv[1:]
    path = find_run(args[0] if args else None)

    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"\nRun:    {os.path.basename(path)}")
    print(f"Model:  {data.get('model', '?')}")
    print(f"Time:   {data.get('timestamp', '?')}")
    print(f"Cases:  {len(results)}\n")

    # per-question table
    hdr = f"{'id':<34} {'category':<12} {'route':>18} {'answer':>7} {'tools':>6}  {'manual':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        g = r.get("grade", {})
        print(
            f"{r['id']:<34} {r['category']:<12} "
            f"{fmt_routing(g.get('routing_correct')):>18} "
            f"{fmt_bool(g.get('answer_correct')):>7} "
            f"{r.get('tool_call_count', 0):>6}  "
            f"{str(r.get('manual_score')):>6}"
        )

    # aggregates
    def rate(key):
        graded = [r["grade"].get(key) for r in results if r["grade"].get(key) is not None]
        if not graded:
            return "n/a"
        return f"{sum(bool(x) for x in graded)}/{len(graded)}"

    print("\nTotals")
    print(f"  routing correct: {rate('routing_correct')}")
    print(f"  answer correct:  {rate('answer_correct')}")

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    print("\nBy category")
    for cat, rs in sorted(by_cat.items()):
        route = sum(1 for r in rs if r["grade"].get("routing_correct"))
        route_checked = sum(1 for r in rs if r["grade"].get("routing_correct") is not None)
        free = len(rs) - route_checked
        free_note = f" (+{free} {FREE_ROUTING})" if free else ""
        ans = sum(1 for r in rs if r["grade"].get("answer_correct"))
        ans_graded = sum(1 for r in rs if r["grade"].get("answer_correct") is not None)
        print(f"  {cat:<14} n={len(rs):<3} routing {route}/{route_checked}{free_note}   answer {ans}/{ans_graded}")

    multi = [r for r in results if "structural_check" in r]
    if multi:
        passed = sum(1 for r in multi if r["structural_check"]["passed"])
        print(f"\nMulti-turn structural checks: {passed}/{len(multi)} passed")
        for r in multi:
            if not r["structural_check"]["passed"]:
                print(f"  FAIL {r['id']}: {'; '.join(r['structural_check']['problems'])}")

    # failures / needs-review detail
    flagged = [
        r for r in results
        if r["grade"].get("routing_correct") is False
        or r["grade"].get("answer_correct") is False
        or r["grade"].get("answer_correct") is None
    ]
    if flagged:
        print("\nNeeds attention")
        print("=" * 60)
        for r in flagged:
            g = r["grade"]
            tag = "FAIL" if (g.get("routing_correct") is False or g.get("answer_correct") is False) else "MANUAL"
            print(f"\n[{tag}] {r['id']}  ({r['category']})")
            if "turns" in r:
                sc = r.get("structural_check") or {}
                print(f"  structural check: {'PASS' if sc.get('passed') else 'FAIL'}"
                      + (f" -- {'; '.join(sc['problems'])}" if sc.get("problems") else "")
                      + (f"   info: {r['info']}" if r.get("info") else ""))
                print_turns(r)
                continue
            print(f"  Q: {r['query']}")
            if g.get("detail"):
                print(f"  note: {g['detail']}")
            tools = " -> ".join(s["tool"] for s in r.get("trace_steps", [])) or "(none)"
            print(f"  tools: {tools}")
            answer = r.get("answer")
            answer = answer.replace("\n", " ") if answer is not None else NO_ANSWER
            print(textwrap.indent(textwrap.fill(answer, 90), "  A: "))

    if show_traces:
        print("\nFull traces")
        print("=" * 60)
        for r in results:
            print(f"\n{r['id']}")
            groups = ([(f"turn {t}", turn.get("trace_steps", [])) for t, turn in enumerate(r["turns"], 1)]
                      if "turns" in r else [(None, r.get("trace_steps", []))])
            for label, steps in groups:
                if label:
                    print(f"  {label}:")
                for i, s in enumerate(steps, 1):
                    dup = "  [blocked duplicate]" if s.get("blocked_duplicate") else ""
                    print(f"  {'  ' if label else ''}{i}. {s['tool']}({json.dumps(s.get('arguments', {}))}){dup}")
    print()


if __name__ == "__main__":
    main()
