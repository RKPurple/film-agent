"""
Run the data-foundation pipeline end to end, pausing
for the user to confirm before moving from one step to the next.

Every generated artifact lands in
data/intermediate/ (raw TMDB / tone responses are cached under data/cache/).

Sequence (each step's output feeds the next):
    1. merge_letterboxd.py         data/letterboxd_export/*.csv        -> data/intermediate/letterboxd_merged.csv
    2. enrich_tmdb.py              data/intermediate/letterboxd_merged.csv -> data/intermediate/films_enriched.json
    3. generate_tone_summaries.py  data/intermediate/films_enriched.json  (adds "tone_summary" in place)

After each step this script surfaces anything worth a human glance -- sanity
report warnings, low-confidence TMDB matches, failed / skipped tone calls --
and then asks whether to continue. Answer "n" at any prompt to stop cleanly;
everything done so far is already written to disk and each underlying script
is cache-aware, so re-running later is cheap.

Usage:
    python3 pipeline/build_dataset.py
    python3 pipeline/build_dataset.py --yes      # don't pause between steps (still prints flags)
    python3 pipeline/build_dataset.py --limit 5  # pass --limit through to enrich + tone (smoke test)
    python3 pipeline/build_dataset.py --force    # pass --force through to enrich + tone
    python3 pipeline/build_dataset.py --start 2  # skip straight to step 2 (merge already done)

Pass-through flags (--limit / --force) only apply to the scripts that accept
them (steps 2 and 3); merge_letterboxd.py takes neither and is always run as-is.
"""

import argparse
import csv
import subprocess
import sys

from cinemagent.config import ENRICHED_JSON, MERGED_CSV, PIPELINE_DIR, SANITY_REPORT, UNMATCHED_CSV


def rule(char="="):
    print(char * 72)


def run_script(script_name, extra_args):
    """Run a pipeline script as a subprocess, streaming its output live.
    Returns the process exit code."""
    cmd = [sys.executable, str(PIPELINE_DIR / script_name), *extra_args]
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def confirm(prompt, assume_yes):
    if assume_yes:
        print(f"{prompt} [auto-yes]")
        return True
    while True:
        ans = input(f"{prompt} [y/n] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  please answer y or n")


def abort(msg):
    print(f"\n{msg}")
    print("Stopping here. Nothing already written is lost; re-run to resume.")
    sys.exit(1)


# --------------------------------------------------------------------------
# Per-step "things to double-check" reporting
# --------------------------------------------------------------------------

def flag_merge_results():
    flags = []
    if not MERGED_CSV.exists():
        abort(f"Expected {MERGED_CSV.name} was not produced -- check the merge output above.")

    with open(MERGED_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n  {MERGED_CSV.name}: {len(rows)} merged film rows")

    if SANITY_REPORT.exists():
        report = SANITY_REPORT.read_text(encoding="utf-8")
        warn_lines = [ln for ln in report.splitlines() if ln.lstrip().startswith("[!]")]
        if warn_lines:
            flags.append("sanity_report.txt has warnings (collisions / duplicates):")
            flags.extend(f"    {ln.strip()}" for ln in warn_lines)
            flags.append(f"    -> open {SANITY_REPORT} for the full context")
        # coverage + rewatch lines are informational but handy to echo
        for ln in report.splitlines():
            if ln.startswith("Coverage:") or "rewatch" in ln.lower():
                print(f"  {ln.strip()}")
    else:
        flags.append("No sanity_report.txt was written -- merge may not have completed normally.")

    return flags


def flag_enrich_results():
    flags = []
    if not ENRICHED_JSON.exists():
        abort(f"Expected {ENRICHED_JSON.name} was not produced -- check the enrich output above.")

    import json
    films = json.loads(ENRICHED_JSON.read_text(encoding="utf-8"))
    print(f"\n  {ENRICHED_JSON.name}: {len(films)} enriched films")

    low_conf = [f for f in films if f.get("low_confidence_match")]
    if low_conf:
        flags.append(f"{len(low_conf)} film(s) matched TMDB with LOW CONFIDENCE (year mismatch) -- verify these are the right films:")
        for f in low_conf[:15]:
            flags.append(f"    {f.get('title')} ({f.get('year')})  ->  TMDB: {f.get('tmdb_title')} [{f.get('release_date')}]  (tmdb_id={f.get('tmdb_id')})")
        if len(low_conf) > 15:
            flags.append(f"    ... and {len(low_conf) - 15} more")

    if UNMATCHED_CSV.exists():
        with open(UNMATCHED_CSV, newline="", encoding="utf-8") as f:
            unmatched = list(csv.DictReader(f))
        # rows whose reason isn't just the low-confidence flag = genuine misses
        hard_misses = [r for r in unmatched if r.get("reason", "") != "low_confidence_year_mismatch"]
        if hard_misses:
            flags.append(f"{len(hard_misses)} film(s) TMDB could NOT resolve at all (no metadata for these downstream):")
            for r in hard_misses[:15]:
                flags.append(f"    {r.get('title')} ({r.get('year')}) -- {r.get('reason')}")
        flags.append(f"    -> full list in {UNMATCHED_CSV}")

    return flags


def flag_tone_results():
    flags = []
    import json
    if not ENRICHED_JSON.exists():
        abort(f"{ENRICHED_JSON.name} disappeared -- something went wrong in the tone step.")

    films = json.loads(ENRICHED_JSON.read_text(encoding="utf-8"))
    missing = [f for f in films if not f.get("tone_summary")]
    have = len(films) - len(missing)
    print(f"\n  tone_summary present on {have}/{len(films)} films")

    if missing:
        flags.append(f"{len(missing)} film(s) have NO tone_summary (failed calls, or run stopped early -- re-run this step to fill them in):")
        for f in missing[:15]:
            flags.append(f"    {f.get('title')} ({f.get('year')})")
        if len(missing) > 15:
            flags.append(f"    ... and {len(missing) - 15} more")

    return flags


def report(flags):
    if not flags:
        print("\n  Nothing flagged -- looks clean.")
        return
    print("\n  !! THINGS TO DOUBLE-CHECK:")
    for line in flags:
        print(f"  {line}")


# --------------------------------------------------------------------------

STEPS = [
    ("merge_letterboxd.py", flag_merge_results, False),
    ("enrich_tmdb.py", flag_enrich_results, True),
    ("generate_tone_summaries.py", flag_tone_results, True),
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--yes", action="store_true", help="Don't pause between steps (flags still printed)")
    parser.add_argument("--limit", type=int, default=None, help="Pass --limit N to enrich + tone steps")
    parser.add_argument("--force", action="store_true", help="Pass --force to enrich + tone steps")
    parser.add_argument("--start", type=int, default=1, choices=(1, 2, 3), help="Start from step N (1-3)")
    args = parser.parse_args()

    passthrough = []
    if args.limit is not None:
        passthrough += ["--limit", str(args.limit)]
    if args.force:
        passthrough += ["--force"]

    total = len(STEPS)
    for idx, (script_name, flagger, accepts_flags) in enumerate(STEPS, 1):
        if idx < args.start:
            continue

        rule()
        print(f"STEP {idx}/{total}: {script_name}")
        rule()

        extra = passthrough if accepts_flags else []
        code = run_script(script_name, extra)

        if code != 0:
            abort(f"STEP {idx} ({script_name}) exited with code {code}.")

        rule("-")
        print(f"STEP {idx} finished. Reviewing output...")
        report(flagger())
        rule("-")

        if idx == total:
            print("\nAll steps complete.")
            break

        next_script = STEPS[idx][0]
        if not confirm(f"\nProceed to STEP {idx + 1}/{total} ({next_script})?", args.yes):
            print("\nStopped by user. Re-run with --start "
                  f"{idx + 1} to pick up from the next step.")
            sys.exit(0)

    print("\nDone. Next: python3 pipeline/load_chroma.py "
          "(rebuild the Chroma collection so new fields get embedded).")


if __name__ == "__main__":
    main()
