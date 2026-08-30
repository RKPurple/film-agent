"""
Load data/intermediate/films_enriched.json into the Postgres schema
defined in ../schema.sql (films / people / film_cast / film_crew / genres /
film_genres / keywords / film_keywords).

Design: this script is idempotent BY BEING DESTRUCTIVE, on purpose. Every
run re-applies schema.sql (drop + recreate every table) and reloads
everything from films_enriched.json from scratch, inside a single
transaction -- either the whole load lands, or none of it does. That's the
right call here because nothing in Postgres is ever hand-edited;
films_enriched.json is the single source of truth, so "wipe and reload" is
simpler and safer than trying to diff/upsert against a moving JSON file.

Usage:
    python3 data/load_postgres.py                  # loads into `entertainmentai`
    python3 data/load_postgres.py --dbname other_db
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

SCRIPT_DIR = Path(__file__).resolve().parent      # .../EntertainmentAI/data
PROJECT_ROOT = SCRIPT_DIR.parent                   # .../EntertainmentAI
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_config import DB_NAME

ENRICHED_JSON = SCRIPT_DIR / "intermediate" / "films_enriched.json"
SCHEMA_SQL = PROJECT_ROOT / "schema.sql"


def to_float(value):
    """Letterboxd CSV fields arrive as strings (or '') -- coerce blanks to None."""
    if value in (None, ""):
        return None
    return float(value)


def to_int(value):
    if value in (None, ""):
        return None
    return int(value)


def apply_schema(cur):
    cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def build_rows(films):
    """Walk films_enriched.json once, deduping people/genres/keywords along
    the way, and return everything as plain lists of tuples ready for
    execute_values -- one list per destination table. Parent tables
    (films/people/genres/keywords) get built before the join tables that
    reference them, so the caller can insert in an order that satisfies the
    foreign keys.
    """
    film_rows = []
    person_rows = {}   # person_id -> row tuple; dedups actors/directors who show up in multiple films
    genre_names = set()
    keyword_names = set()
    cast_rows = []
    crew_rows = []
    film_genre_rows = []
    film_keyword_rows = []

    for film in films:
        film_id = film["tmdb_id"]

        film_rows.append((
            film_id,
            film.get("letterboxd_uri"),
            film["title"],
            to_int(film.get("year")),
            film.get("tmdb_title"),
            film.get("release_date"),
            to_int(film.get("runtime_minutes")),
            film.get("overview"),
            film.get("original_language"),
            to_float(film.get("vote_average")),
            to_float(film.get("popularity")),
            to_float(film.get("rating")),       # -> films.my_rating
            film.get("date_logged"),
            film.get("watched_date"),
            to_int(film.get("rewatch_count")),
            film.get("review_text"),
            film.get("tags"),
            bool(film.get("low_confidence_match")),
        ))

        # enumerate() index doubles as billing order: TMDB's credits.cast is
        # already returned pre-sorted by billing, and enrich_tmdb.py's
        # summarize() preserves that order when it keeps the top 10.
        for order, member in enumerate(film.get("cast", [])):
            pid = member["tmdb_person_id"]
            person_rows[pid] = (pid, member["name"], to_float(member.get("popularity")))
            cast_rows.append((film_id, pid, member.get("character", ""), order))

        for director in film.get("directors", []):
            pid = director["tmdb_person_id"]
            person_rows[pid] = (pid, director["name"], to_float(director.get("popularity")))
            crew_rows.append((film_id, pid, "Director"))

        for genre in film.get("genres", []):
            genre_names.add(genre)
            film_genre_rows.append((film_id, genre))

        for keyword in film.get("keywords", []):
            keyword_names.add(keyword)
            film_keyword_rows.append((film_id, keyword))

    return {
        "films": film_rows,
        "people": list(person_rows.values()),
        "genres": [(name,) for name in sorted(genre_names)],
        "keywords": [(name,) for name in sorted(keyword_names)],
        "film_cast": cast_rows,
        "film_crew": crew_rows,
        "film_genres": film_genre_rows,
        "film_keywords": film_keyword_rows,
    }


def insert(cur, table, columns, rows, page_size=500):
    if not rows:
        return
    query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
    psycopg2.extras.execute_values(cur, query, rows, page_size=page_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dbname", default=DB_NAME)
    args = parser.parse_args()

    if not ENRICHED_JSON.exists():
        raise SystemExit(f"ERROR: {ENRICHED_JSON} not found -- run enrich_tmdb.py first.")

    with open(ENRICHED_JSON, encoding="utf-8") as f:
        films = json.load(f)
    print(f"Loaded {len(films)} film records from {ENRICHED_JSON}")

    rows = build_rows(films)

    conn = psycopg2.connect(dbname=args.dbname)
    try:
        with conn:  # commits the whole transaction on success, rolls back all of it on any error
            with conn.cursor() as cur:
                print("Applying schema.sql (drop + recreate all tables)...")
                apply_schema(cur)

                insert(cur, "films", [
                    "film_id", "letterboxd_uri", "title", "year", "tmdb_title",
                    "release_date", "runtime_minutes", "overview", "original_language",
                    "vote_average", "popularity", "my_rating", "date_logged",
                    "watched_date", "rewatch_count", "review_text", "tags",
                    "low_confidence_match",
                ], rows["films"])
                print(f"  films: {len(rows['films'])}")

                insert(cur, "people", ["person_id", "name", "popularity"], rows["people"])
                print(f"  people: {len(rows['people'])}")

                insert(cur, "genres", ["name"], rows["genres"])
                print(f"  genres: {len(rows['genres'])}")

                insert(cur, "keywords", ["name"], rows["keywords"])
                print(f"  keywords: {len(rows['keywords'])}")

                # Join tables last -- they reference the parent rows just inserted above.
                insert(cur, "film_cast", ["film_id", "person_id", "character_name", "cast_order"], rows["film_cast"])
                print(f"  film_cast: {len(rows['film_cast'])}")

                insert(cur, "film_crew", ["film_id", "person_id", "role"], rows["film_crew"])
                print(f"  film_crew: {len(rows['film_crew'])}")

                insert(cur, "film_genres", ["film_id", "genre_name"], rows["film_genres"])
                print(f"  film_genres: {len(rows['film_genres'])}")

                insert(cur, "film_keywords", ["film_id", "keyword_name"], rows["film_keywords"])
                print(f"  film_keywords: {len(rows['film_keywords'])}")
        print("Done -- transaction committed.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
