-- EntertainmentAI: Postgres schema for the structured metadata store (Phase 2).
--
-- This file is meant to be re-run from scratch every time the loader runs --
-- all data here is fully derived from data/films_enriched.json and
-- data/tmdb_cache/, so there's no need for migrations at this stage: drop
-- everything, recreate it, reload it. CASCADE on each drop means the order
-- below doesn't strictly matter, but child tables are listed first for
-- readability.
--
-- Apply with:  psql entertainmentai -f pipeline/schema.sql
-- (pipeline/load_postgres.py applies it automatically on every load.)

DROP TABLE IF EXISTS film_cast CASCADE;
DROP TABLE IF EXISTS film_crew CASCADE;
DROP TABLE IF EXISTS film_genres CASCADE;
DROP TABLE IF EXISTS film_keywords CASCADE;
DROP TABLE IF EXISTS films CASCADE;
DROP TABLE IF EXISTS people CASCADE;
DROP TABLE IF EXISTS genres CASCADE;
DROP TABLE IF EXISTS keywords CASCADE;

CREATE TABLE films (
    film_id              INTEGER PRIMARY KEY,   -- TMDB movie id
    letterboxd_uri       TEXT,
    title                TEXT NOT NULL,
    year                 INTEGER,               -- Letterboxd's listed year
    tmdb_title           TEXT,
    release_date         TEXT,                  -- TMDB's actual release date
    runtime_minutes      INTEGER,
    overview             TEXT,
    original_language    TEXT,
    vote_average         REAL,                  -- TMDB's public rating
    popularity           REAL,                  -- TMDB's film popularity score
    my_rating            REAL,                  -- your Letterboxd rating
    date_logged          TEXT,
    watched_date         TEXT,
    rewatch_count        INTEGER,
    review_text          TEXT,
    tags                 TEXT,
    low_confidence_match BOOLEAN
);

CREATE TABLE people (
    person_id   INTEGER PRIMARY KEY,   -- TMDB person id
    name        TEXT NOT NULL,
    popularity  REAL                   -- TMDB's person popularity score (distinct from film popularity)
);

CREATE TABLE film_cast (
    film_id         INTEGER REFERENCES films(film_id),
    person_id       INTEGER REFERENCES people(person_id),
    character_name  TEXT,
    cast_order      INTEGER,           -- billing order
    PRIMARY KEY (film_id, person_id, character_name)
);

CREATE TABLE film_crew (
    film_id     INTEGER REFERENCES films(film_id),
    person_id   INTEGER REFERENCES people(person_id),
    role        TEXT,                  -- "Director" today; room for Writer/Composer later
    PRIMARY KEY (film_id, person_id, role)
);

CREATE TABLE genres (
    name    TEXT PRIMARY KEY
);

CREATE TABLE film_genres (
    film_id     INTEGER REFERENCES films(film_id),
    genre_name  TEXT REFERENCES genres(name),
    PRIMARY KEY (film_id, genre_name)
);

CREATE TABLE keywords (
    name    TEXT PRIMARY KEY
);

CREATE TABLE film_keywords (
    film_id       INTEGER REFERENCES films(film_id),
    keyword_name  TEXT REFERENCES keywords(name),
    PRIMARY KEY (film_id, keyword_name)
);
