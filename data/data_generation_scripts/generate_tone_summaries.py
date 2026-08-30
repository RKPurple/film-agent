"""
Synthesize a short "tone" line per film and fold it into
films_enriched.json, so search_my_history has something real to match
against for mood/vibe queries

- One Gemini call per film, synthesizing a single tone/mood sentence from
  the TMDB overview, the full keyword list, and the Letterboxd review
  ("My take") when present. The prompt explicitly asks the model to pull
  OUT only the tone-relevant signal from the keyword list and ignore
  plot/setting keywords ("new york city", "jazz"), deliberately not
  pre-filtering keywords with a hardcoded tone-keyword list.
- The model is explicitly told not to restate the plot -- a tone line
  reads like "tense and claustrophobic, with a creeping sense of dread",
  never "a drummer pursues perfection under a ruthless instructor".
- Cached per film in data/cache/tone/{tmdb_id}.json, same pattern as
  enrich_tmdb.py's data/cache/tmdb/

  
- Output: films_enriched.json is rewritten in place with a new
  "tone_summary" field added to every film record; every other field is
  left untouched. It stays the single source of truth downstream loaders
  read from -- nothing else needs to change to pick this up except
  load_chroma.py's build_chunk_text(), which gets one new "Tone:" line.

Usage:
    python3 data/data_generation_scripts/generate_tone_summaries.py            # full run, cache-aware
    python3 data/data_generation_scripts/generate_tone_summaries.py --limit 5   # smoke-test on first 5 films
    python3 data/data_generation_scripts/generate_tone_summaries.py --force     # ignore cache, regenerate everything

After this finishes:
    python3 data/load_chroma.py    # rebuild the Chroma collection so the
                                   # new "Tone:" line actually gets embedded
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

SCRIPT_DIR = Path(__file__).resolve().parent          # .../data/data_generation_scripts
DATA_DIR = SCRIPT_DIR.parent                           # .../data
PROJECT_ROOT = DATA_DIR.parent                         # .../EntertainmentAI
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_config import GEMINI_BASE_URL, GEMINI_MODEL, TONE_TEMPERATURE, load_env

ENRICHED_JSON = DATA_DIR / "intermediate" / "films_enriched.json"
TONE_CACHE_DIR = DATA_DIR / "cache" / "tone"

# Pause between LLM calls, same spirit as enrich_tmdb.py's
# REQUEST_SLEEP_SEC for TMDB. This was 13s to stay under the free tier's 5
# RPM cap; now that tier 1 is set up (billing linked), the AI Studio
# dashboard shows roughly 1,000 RPM / 2M TPM / 10,000 RPD, so per-call
# pacing barely matters anymore. 0.1s just matches enrich_tmdb.py's "don't
# hammer it" spirit -- not a real throttle at these limits.
REQUEST_SLEEP_SEC = 0.1

# On a RateLimitError, pause a full minute and retry the same film. At the
# free tier's 20 RPD this doubled as a way to detect the daily cap (see the
# RateLimitError handling below); at tier 1's ~10,000 RPD a 170-film run
# won't come close to that ceiling, so hitting this now more likely means
# something transient or a real problem, not "come back tomorrow." Retries
# bumped back up since they're cheap at this tier and there's less reason
# to give up fast.
RATE_LIMIT_SLEEP_SEC = 60
RATE_LIMIT_MAX_RETRIES = 5

TONE_PROMPT = """You are tagging a film with a short description of its emotional TONE -- not its plot.

Title: {title} ({year})
Overview: {overview}
Keywords (a mix of real tone/mood signal and plot/setting noise -- pull out only what actually describes tone or feeling, ignore the rest): {keywords}
{review_line}
Write ONE sentence (12-20 words) describing the film's emotional tone or mood -- the kind of feeling a viewer comes away with (for example: "tense and paranoid, with a creeping sense of dread" or "warm and bittersweet, carried by quiet grief"). Do NOT summarize the plot or describe what happens in the film. Respond with only that one sentence, nothing else -- no preamble, no quotation marks."""


def build_prompt(film):
    review = film.get("review_text")
    review_line = f'The viewer\'s own reaction to it: "{review}"\n' if review else ""
    keywords = ", ".join(film.get("keywords") or []) or "(none)"
    return TONE_PROMPT.format(
        title=film.get("title", ""),
        year=film.get("year", ""),
        overview=film.get("overview") or "(no overview available)",
        keywords=keywords,
        review_line=review_line,
    )


def generate_tone(client, film):
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": build_prompt(film)}],
        temperature=TONE_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def main():
    load_env()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N films (for testing)")
    parser.add_argument("--force", action="store_true", help="Ignore cache and regenerate every film's tone summary")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set the GEMINI_API_KEY environment variable first (see script docstring).")

    if not ENRICHED_JSON.exists():
        sys.exit(f"ERROR: {ENRICHED_JSON} not found -- run enrich_tmdb.py first.")

    with open(ENRICHED_JSON, encoding="utf-8") as f:
        all_films = json.load(f)

    # A plain slice shares the same dict objects with all_films (shallow
    # copy of the list, not of its contents) -- mutating film["tone_summary"]
    # below on an element of `films` also mutates the matching element of
    # `all_films`, so writing all_films back out always preserves every
    # record, whether or not --limit narrowed this run.
    films = all_films[: args.limit] if args.limit else all_films

    TONE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    n_cached = 0
    n_generated = 0
    n_failed = 0
    stopped_early = False
    for i, film in enumerate(films, 1):
        title, year = film.get("title", "?"), film.get("year", "")
        cache_path = TONE_CACHE_DIR / f"{film['tmdb_id']}.json"
        print(f"[{i}/{len(films)}] {title} ({year})...", end=" ", flush=True)

        if cache_path.exists() and not args.force:
            with open(cache_path, encoding="utf-8") as f:
                tone = json.load(f)["tone_summary"]
            film["tone_summary"] = tone
            n_cached += 1
            print("cached")
            continue

        try:
            for rl_attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    tone = generate_tone(client, film)
                    break
                except RateLimitError:
                    if rl_attempt == RATE_LIMIT_MAX_RETRIES:
                        raise
                    print(f"rate limited -- sleeping {RATE_LIMIT_SLEEP_SEC}s then retrying", flush=True)
                    time.sleep(RATE_LIMIT_SLEEP_SEC)
        except RateLimitError:
            # Still rate limited after RATE_LIMIT_MAX_RETRIES full 60s
            # waits. At the free tier's 20 RPD this reliably meant the
            # daily cap; at tier 1's ~10,000 RPD a 170-film run won't come
            # close to that ceiling, so this isn't a confident "come back
            # tomorrow" signal anymore -- it's more likely something
            # transient (an outage, a quota misconfiguration, etc.) worth
            # surfacing rather than silently grinding through every
            # remaining film against the same error. Stopping the whole
            # run here either way -- everything generated so far is
            # already cached per-film and still gets written out below.
            print(
                f"\nStill rate limited after {RATE_LIMIT_MAX_RETRIES} wait(s) of {RATE_LIMIT_SLEEP_SEC}s each -- "
                "stopping this run rather than repeating the same doomed retry on every remaining film. "
                f"{title} ({year}) and everything after it in this batch was not generated this time. "
                "This doesn't look like the free tier's daily cap anymore now that tier 1's limits are much "
                "higher -- worth checking what's actually going on before re-running."
            )
            stopped_early = True
            break
        except Exception as e:
            print(f"FAILED ({type(e).__name__}: {e})")
            n_failed += 1
            continue

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"tone_summary": tone}, f, indent=2)
        film["tone_summary"] = tone
        n_generated += 1
        print(tone)
        time.sleep(REQUEST_SLEEP_SEC)

    with open(ENRICHED_JSON, "w", encoding="utf-8") as f:
        json.dump(all_films, f, indent=2, ensure_ascii=False)

    print(f"\n{n_generated} generated, {n_cached} loaded from cache, {n_failed} failed.")
    print(f"{ENRICHED_JSON} updated.")
    if stopped_early:
        print(
            "Stopped early after repeated rate limit errors. On tier 1's much higher limits this probably "
            "isn't the daily cap -- worth checking what happened before just re-running. Everything generated "
            "so far is cached and won't be re-requested either way."
        )
    elif n_failed:
        print("Re-run the same command to retry only the failed films -- everything else is cached.")
    print("\nNext: python3 data/load_chroma.py  (rebuild the collection so the new Tone: line gets embedded)")


if __name__ == "__main__":
    main()
