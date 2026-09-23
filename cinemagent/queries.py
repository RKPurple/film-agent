"""
Deterministic query functions over the Postgres schema --
plain parameterized SQL, no LLM involved. These are the "exact lookup" half
of hybrid retrieval


- Actor/director name matching uses ILIKE '%...%' (case-insensitive,
  substring) since a caller, human or LLM, won't always type a name
  exactly as TMDB has it capitalized/spelled. Genre matching uses ILIKE
  with no wildcards (case-insensitive, but exact) since genres are a small
  controlled vocabulary where substring matching would just cause
  surprising over-matches.
- Rows come back as plain dicts (via RealDictCursor), not tuples, easier
  to hand straight to an LLM as tool-call output later, and easier to read
  while testing now.

Usage (manual smoke test):
    python -m cinemagent.queries
"""

import psycopg2
import psycopg2.extras

from cinemagent.config import DB_NAME


def get_connection(dbname=DB_NAME):
    conn = psycopg2.connect(dbname=dbname, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def filter_by_actor(conn, name):
    """All watched films where `name` (substring, case-insensitive) appears in the cast."""
    sql = """
        SELECT f.film_id, f.title, f.year, f.my_rating, fc.character_name, fc.cast_order
        FROM film_cast fc
        JOIN people p ON p.person_id = fc.person_id
        JOIN films f ON f.film_id = fc.film_id
        WHERE p.name ILIKE %s
        ORDER BY f.year, fc.cast_order;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (f"%{name}%",))
        return cur.fetchall()


def filter_by_director(conn, name):
    """All watched films where `name` (substring, case-insensitive) directed."""
    sql = """
        SELECT f.film_id, f.title, f.year, f.my_rating
        FROM film_crew fcw
        JOIN people p ON p.person_id = fcw.person_id
        JOIN films f ON f.film_id = fcw.film_id
        WHERE fcw.role = 'Director' AND p.name ILIKE %s
        ORDER BY f.year;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (f"%{name}%",))
        return cur.fetchall()


def filter_by_genre(conn, genre):
    """All watched films tagged with an exact (case-insensitive) genre name."""
    sql = """
        SELECT f.film_id, f.title, f.year, f.my_rating
        FROM film_genres fg
        JOIN films f ON f.film_id = fg.film_id
        WHERE fg.genre_name ILIKE %s
        ORDER BY f.my_rating DESC NULLS LAST;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (genre,))
        return cur.fetchall()


def filter_by_rating(conn, min_rating=None, max_rating=None):
    """Watched films with my_rating in [min_rating, max_rating] (either bound optional)."""
    clauses = []
    params = []
    if min_rating is not None:
        clauses.append("my_rating >= %s")
        params.append(min_rating)
    if max_rating is not None:
        clauses.append("my_rating <= %s")
        params.append(max_rating)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT film_id, title, year, my_rating
        FROM films
        {where}
        ORDER BY my_rating DESC NULLS LAST;
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def cross_reference(conn, person_name):
    """Every watched film involving `person_name` in ANY role (acting or
    directing), with which role attached to each row -- this is the
    "find overlaps across watched films" tool from CLAUDE.md's checklist.
    """
    sql = """
        SELECT f.film_id, f.title, f.year, 'cast' AS role, fc.character_name AS detail
        FROM film_cast fc
        JOIN people p ON p.person_id = fc.person_id
        JOIN films f ON f.film_id = fc.film_id
        WHERE p.name ILIKE %s

        UNION ALL

        SELECT f.film_id, f.title, f.year, fcw.role AS role, NULL AS detail
        FROM film_crew fcw
        JOIN people p ON p.person_id = fcw.person_id
        JOIN films f ON f.film_id = fcw.film_id
        WHERE p.name ILIKE %s

        ORDER BY year;
    """
    pattern = f"%{person_name}%"
    with conn.cursor() as cur:
        cur.execute(sql, (pattern, pattern))
        return cur.fetchall()


def get_all_genres(conn):
    """Every genre in the controlled vocabulary, with how many watched films
    use it. Backs the agent's ability to see which genre names actually exist
    before calling filter_by_genre (which does exact, not fuzzy, matching)."""
    sql = """
        SELECT g.name, COUNT(fg.film_id) AS film_count
        FROM genres g
        LEFT JOIN film_genres fg ON fg.genre_name = g.name
        GROUP BY g.name
        ORDER BY film_count DESC, g.name;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def _print_rows(label, rows):
    print(f"\n-- {label} ({len(rows)} row(s)) --")
    for row in rows:
        print(" ", dict(row))


if __name__ == "__main__":
    # Manual smoke test against real rows from Rohan's own watched history --
    # not an automated test suite, just a quick "does this look right" check
    # you can eyeball after running the loader.
    conn = get_connection()
    try:
        _print_rows("filter_by_actor('Ryan Gosling')", filter_by_actor(conn, "Ryan Gosling"))
        _print_rows("filter_by_director('Celine Song')", filter_by_director(conn, "Celine Song"))
        _print_rows("filter_by_genre('Comedy')", filter_by_genre(conn, "Comedy")[:5])
        _print_rows("filter_by_rating(min_rating=4.5)", filter_by_rating(conn, min_rating=4.5)[:5])
        _print_rows("cross_reference('Jordan')", cross_reference(conn, "Jordan"))
        _print_rows("get_all_genres()", get_all_genres(conn))
    finally:
        conn.close()
