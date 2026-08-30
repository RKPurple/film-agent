"""
Merge the raw Letterboxd export CSVs into a single
per-film dataset, and sanity-check the result.

Input:  data/letterboxd_export/{watched,ratings,diary,reviews}.csv  (from the
        Letterboxd "Export data" zip, already unzipped into
        data/letterboxd_export/)
Output: data/intermediate/letterboxd_merged.csv  -- one row per unique watched film
        data/intermediate/sanity_report.txt      -- collisions / mismatches worth eyeballing

Join key: (Name, Year) -- NOT "Letterboxd URI".

IMPORTANT quirk discovered while building this:
the "Letterboxd URI" in diary.csv and reviews.csv is a
per-LOG-ENTRY short link, not a stable per-film identifier, it does
NOT match the URI for the same film in watched.csv/ratings.csv. E.g.
"Project Hail Mary" is boxd.it/qXXXX in watched.csv but boxd.it/e8WRCd in
diary.csv. ratings.csv's URI *does* happen to match watched.csv's, but
diary.csv/reviews.csv's does not. Join everything on (Name, Year)
instead, which Letterboxd's own dedup already guarantees is unique within
watched.csv and is consistent across all five files.

Design note: watched.csv is the master list (every film ever marked
watched). ratings.csv, diary.csv, and reviews.csv are narrower slices
"""

import csv
from collections import defaultdict
from pathlib import Path

# This script lives in data/data_generation_scripts/. The raw export sits in
# data/letterboxd_export/; generated artifacts go to data/intermediate/.
DATA_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATA_DIR / "letterboxd_export"
OUT_DIR = DATA_DIR / "intermediate"


def read_csv(name):
    path = RAW_DIR / name
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    watched = read_csv("watched.csv")
    ratings = read_csv("ratings.csv")
    diary = read_csv("diary.csv")
    reviews = read_csv("reviews.csv")

    def ny(row):
        return (row["Name"], row["Year"])

    ratings_by_ny = {ny(r): r for r in ratings}

    # Group diary/review entries by (Name, Year) -- a film can have several
    # rows (rewatches).
    diary_by_ny = defaultdict(list)
    for d in diary:
        diary_by_ny[ny(d)].append(d)
    reviews_by_ny = defaultdict(list)
    for r in reviews:
        reviews_by_ny[ny(r)].append(r)

    sanity_lines = [
        "NOTE: diary.csv/reviews.csv 'Letterboxd URI' is a per-log-entry short "
        "link and does NOT match watched.csv/ratings.csv's URI for the same "
        "film -- all joins here use (Name, Year) instead. See script docstring.",
        "",
    ]

    # --- collision check: duplicate URIs within watched.csv itself ---
    uri_counts = defaultdict(int)
    for w in watched:
        uri_counts[w["Letterboxd URI"]] += 1
    dup_uris = {uri: n for uri, n in uri_counts.items() if n > 1}
    if dup_uris:
        sanity_lines.append(f"[!] {len(dup_uris)} duplicate URI(s) in watched.csv:")
        for uri, n in dup_uris.items():
            sanity_lines.append(f"    {uri} appears {n}x")
    else:
        sanity_lines.append(f"[ok] No duplicate URIs in watched.csv ({len(watched)} rows).")

    # --- collision check: same (Name, Year) but different URI ---
    # (legitimately different films that happen to share a title+year, OR
    #  a sign the export has something weird going on -- worth a human glance)
    name_year_to_uris = defaultdict(set)
    for w in watched:
        name_year_to_uris[(w["Name"], w["Year"])].add(w["Letterboxd URI"])
    ny_collisions = {k: v for k, v in name_year_to_uris.items() if len(v) > 1}
    if ny_collisions:
        sanity_lines.append(f"\n[!] {len(ny_collisions)} (Name, Year) pair(s) map to >1 URI:")
        for (name, year), uris in ny_collisions.items():
            sanity_lines.append(f"    {name} ({year}): {sorted(uris)}")
    else:
        sanity_lines.append("[ok] No (Name, Year) collisions across distinct URIs.")

    # --- coverage stats ---
    n_watched = len(watched)
    n_rated = sum(1 for w in watched if ny(w) in ratings_by_ny)
    n_diaried = sum(1 for w in watched if ny(w) in diary_by_ny)
    n_reviewed = sum(1 for w in watched if ny(w) in reviews_by_ny)
    n_rewatched = sum(1 for rows in diary_by_ny.values() if len(rows) > 1)
    sanity_lines.append(
        f"\nCoverage: {n_watched} watched | {n_rated} rated | "
        f"{n_diaried} have a diary entry | {n_reviewed} have a review text | "
        f"{n_rewatched} film(s) show >1 diary entry (rewatch)"
    )

    # --- watched films with no rating at all (fine, but worth knowing) ---
    unrated = [w for w in watched if ny(w) not in ratings_by_ny]
    if unrated:
        sanity_lines.append(f"\n[i] {len(unrated)} watched film(s) have no rating logged:")
        for w in unrated[:20]:
            sanity_lines.append(f"    {w['Name']} ({w['Year']})")
        if len(unrated) > 20:
            sanity_lines.append(f"    ... and {len(unrated) - 20} more")

    # --- build merged rows ---
    merged = []
    for w in watched:
        key = ny(w)
        rating_row = ratings_by_ny.get(key)
        diary_rows = sorted(diary_by_ny.get(key, []), key=lambda d: d["Watched Date"])
        review_rows = sorted(reviews_by_ny.get(key, []), key=lambda r: r["Watched Date"])
        latest_diary = diary_rows[-1] if diary_rows else None
        latest_review = review_rows[-1] if review_rows else None

        merged.append({
            "letterboxd_uri": w["Letterboxd URI"],
            "title": w["Name"],
            "year": w["Year"],
            "date_logged": w["Date"],
            "rating": rating_row["Rating"] if rating_row else "",
            "watched_date": (latest_diary["Watched Date"] if latest_diary
                              else latest_review["Watched Date"] if latest_review else ""),
            "rewatch_count": len(diary_rows),
            "review_text": latest_review["Review"] if latest_review else "",
            "tags": (latest_diary["Tags"] if latest_diary and latest_diary.get("Tags")
                     else (latest_review["Tags"] if latest_review else "")),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "letterboxd_merged.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        writer.writeheader()
        writer.writerows(merged)

    report_path = OUT_DIR / "sanity_report.txt"
    report_path.write_text("\n".join(sanity_lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(merged)} merged rows -> {out_csv}")
    print(f"Wrote sanity report -> {report_path}")
    print()
    print("\n".join(sanity_lines))


if __name__ == "__main__":
    main()