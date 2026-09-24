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
            for i, s in enumerate(r.get("trace_steps", []), 1):
                dup = "  [blocked duplicate]" if s.get("blocked_duplicate") else ""
                print(f"  {i}. {s['tool']}({json.dumps(s.get('arguments', {}))}){dup}")
    print()


if __name__ == "__main__":
    main()
