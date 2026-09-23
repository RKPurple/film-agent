"""
Enrich each watched film with TMDB metadata
(cast, crew/director, genres, keywords, synopsis), caching raw responses
locally so re-runs are cheap and idempotent

Input:  data/intermediate/letterboxd_merged.csv   (from pipeline/merge_letterboxd.py)
Output: data/cache/tmdb/{tmdb_id}.json           -- one raw TMDB API response per film
        data/intermediate/films_enriched.json    -- Letterboxd fields + TMDB fields, one record/film
        data/intermediate/unmatched.csv          -- films TMDB search couldn't confidently resolve

Usage:
    python3 pipeline/enrich_tmdb.py            # full run
    python3 pipeline/enrich_tmdb.py --limit 5   # smoke-test on first 5 films
    python3 pipeline/enrich_tmdb.py --force     # ignore cache, re-fetch everything

Matching strategy: TMDB /search/movie by title, then prefer a candidate
whose release year matches exactly. If no candidate matches the year,
take the top search result but mark it low_confidence=True so it shows
up in unmatched.csv for a manual glance.
"""

import argparse
import csv
import json
import os
import sys
import time

from cinemagent.config import ENRICHED_JSON, INTERMEDIATE_DIR, MERGED_CSV, TMDB_CACHE_DIR, UNMATCHED_CSV
from cinemagent.tmdb_client import fetch_details, tmdb_get

REQUEST_SLEEP_SEC = 0.1  # be polite; TMDB's limits are generous but no need to hammer it


def search_movie(title, year, api_key):
    """Return (best_match_dict_or_None, low_confidence_bool)."""
    data = tmdb_get("/search/movie", api_key, {"query": title, "include_adult": "false"})
    results = data.get("results", [])
    if not results:
        return None, True

    year_str = str(year)
    for r in results:
        release_date = r.get("release_date") or ""
        if release_date[:4] == year_str:
            return r, False

    # No exact year match -- fall back to top result, flagged low confidence
    return results[0], True


def load_cache(tmdb_id):
    path = TMDB_CACHE_DIR / f"{tmdb_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(tmdb_id, data):
    TMDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (TMDB_CACHE_DIR / f"{tmdb_id}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def summarize(details):
    credits = details.get("credits", {})
    cast = [
        {"tmdb_person_id": c["id"], "name": c["name"], "character": c.get("character", ""),
         "popularity": c.get("popularity")}
        for c in credits.get("cast", [])[:10]
    ]
    directors = [
        {"tmdb_person_id": c["id"], "name": c["name"], "popularity": c.get("popularity")}
        for c in credits.get("crew", []) if c.get("job") == "Director"
    ]
    keywords = [k["name"] for k in details.get("keywords", {}).get("keywords", [])]

    return {
        "tmdb_id": details["id"],
        "tmdb_title": details.get("title"),
        "release_date": details.get("release_date"),
        "runtime_minutes": details.get("runtime"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "overview": details.get("overview"),
        "directors": directors,
        "cast": cast,
        "keywords": keywords,
        "original_language": details.get("original_language"),
        "vote_average": details.get("vote_average"),
        "popularity": details.get("popularity"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N films (for testing)")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-fetch every film")
    args = parser.parse_args()

    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        sys.exit("ERROR: set the TMDB_API_KEY environment variable first (see script docstring).")

    if not MERGED_CSV.exists():
        sys.exit(f"ERROR: {MERGED_CSV} not found -- run merge_letterboxd.py first.")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    with open(MERGED_CSV, newline="", encoding="utf-8") as f:
        films = list(csv.DictReader(f))

    if args.limit:
        films = films[: args.limit]

    enriched = []
    unmatched = []

    for i, film in enumerate(films, 1):
        title, year = film["title"], film["year"]
        print(f"[{i}/{len(films)}] {title} ({year})...", end=" ", flush=True)

        try:
            match, low_conf = search_movie(title, year, api_key)
        except RuntimeError as e:
            print(f"SEARCH FAILED: {e}")
            unmatched.append({**film, "reason": f"search_failed: {e}"})
            continue

        if match is None:
            print("NO MATCH")
            unmatched.append({**film, "reason": "no_tmdb_search_results"})
            continue

        tmdb_id = match["id"]
        cached = None if args.force else load_cache(tmdb_id)
        if cached is not None:
            details = cached
            print(f"cached (tmdb_id={tmdb_id})")
        else:
            try:
                details = fetch_details(tmdb_id, api_key)
                save_cache(tmdb_id, details)
                time.sleep(REQUEST_SLEEP_SEC)
                print(f"fetched (tmdb_id={tmdb_id})" + (" [low confidence]" if low_conf else ""))
            except RuntimeError as e:
                print(f"DETAILS FETCH FAILED: {e}")
                unmatched.append({**film, "reason": f"details_failed: {e}"})
                continue

        record = {**film, **summarize(details), "low_confidence_match": low_conf}
        enriched.append(record)
        if low_conf:
            unmatched.append({**film, "reason": "low_confidence_year_mismatch", "tmdb_id": tmdb_id,
                               "tmdb_title": details.get("title"), "tmdb_release_date": details.get("release_date")})

    ENRICHED_JSON.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(enriched)} enriched film(s) -> {ENRICHED_JSON}")

    if unmatched:
        fieldnames = sorted({k for row in unmatched for k in row.keys()})
        with open(UNMATCHED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unmatched)
        print(f"Wrote {len(unmatched)} row(s) needing review -> {UNMATCHED_CSV}")
    else:
        print("No unmatched/low-confidence rows.")


if __name__ == "__main__":
    main()