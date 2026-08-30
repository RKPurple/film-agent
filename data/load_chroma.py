"""
chunk + embed films_enriched.json into a persistent local
Chroma vector store, one chunk per film.

- One blended chunk per film: title/year, TMDB synopsis, up to the first 10
  keywords as a "Themes:" line, an LLM-synthesized "Tone:" line, Letterboxd review as "My take:"
  when one exists. Genres are deliberately NOT embedded -- they're a small,
  rigid taxonomy better served by exact SQL matching (see queries.py's
  filter_by_genre) than fuzzy vector similarity.
- Embedding model: BAAI/bge-base-en-v1.5, chosen deliberately over Chroma's
  default MiniLM since compute is a non-issue at this scale.
  BGE is retrieval-tuned and ASYMMETRIC: documents get NO instruction prefix, 
  but queries need one prepended at query time.
  This script only ever embeds documents, so it never adds a prefix -- the
  query-side script must apply QUERY_PREFIX consistently or retrieval
  quality silently degrades without erroring.
- Like data/load_postgres.py, this is idempotent by being destructive: every
  run drops and recreates the Chroma collection and reloads everything from
  films_enriched.json, since that JSON is the single source of truth and
  nothing in Chroma is ever hand-edited.

Usage:
    python3 data/load_chroma.py

First run downloads the bge-base-en-v1.5 model weights (~440MB) from
Hugging Face -- needs network, and needs the venv active with
requirements.txt installed
"""

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env_config import (
    CHROMA_COLLECTION_NAME,
    CHUNK_KEYWORD_CAP,
    CHUNK_REVIEW_WEIGHT,
    CHUNK_TONE_WEIGHT,
    EMBEDDING_MODEL_NAME,
)

ENRICHED_JSON = SCRIPT_DIR / "intermediate" / "films_enriched.json"
CHROMA_DIR = SCRIPT_DIR / "chroma_db"

# Historical names kept for existing importers; values now come from env_config.
COLLECTION_NAME = CHROMA_COLLECTION_NAME
KEYWORD_CAP = CHUNK_KEYWORD_CAP

# BGE's asymmetric-retrieval instruction prefix. Only ever applied to QUERIES
# at query time (in the not-yet-written search script) -- never to documents,
# which is why this constant is defined here but never used in this file.
# Kept here so both scripts import the same literal instead of risking two
# copies drifting apart.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_chunk_text(film):
    lines = [f"{film['title']} ({film.get('year', '')})", film.get("overview") or ""]

    keywords = (film.get("keywords") or [])[:KEYWORD_CAP]
    if keywords:
        lines.append(f"Themes: {', '.join(keywords)}")

    # Phase 5 tuning: a short LLM-synthesized tone/mood line (see
    # generate_tone_summaries.py), added to give vibe/mood queries actual
    # text to match against -- TMDB overviews are plot summaries and most
    # films have no personal review, so neither had anything to say about
    # tone before this. Optional: a films_enriched.json that hasn't been
    # through generate_tone_summaries.py yet simply won't have the field.
    #
    # Phase 5 tuning, weighting pass: fuzzy_surprised (eval run
    # 20260829T221851Z) failed to retrieve either film Rohan actually had
    # in mind, and both films' tone_summary leaned toward tension/dread
    # rather than surprise -- the real "surprise" signal only lived in one
    # short review_text line, competing unweighted against the rest of the
    # chunk. CHUNK_TONE_WEIGHT/CHUNK_REVIEW_WEIGHT (env_config.py) repeat
    # these lines so their terms carry proportionally more influence on the
    # pooled embedding -- review weighted heaviest (Rohan's own reaction),
    # tone next (the LLM's read), everything else above gets no extra
    # weight at all. Rohan's own note: current review text is often terse
    # ("Way deeper than I thought") since it was written for Letterboxd,
    # not for retrieval -- he plans to write more descriptive reviews going
    # forward, which this weighting is set up to take advantage of without
    # further code changes.
    tone = film.get("tone_summary")
    if tone:
        lines.extend([f"Tone: {tone}"] * CHUNK_TONE_WEIGHT)

    review = film.get("review_text")
    if review:
        lines.extend([f"My take: {review}"] * CHUNK_REVIEW_WEIGHT)

    return "\n".join(lines)


def build_metadata(film):
    # Chroma metadata values must be scalars (str/int/float/bool) -- and
    # won't accept None at all, so optional fields get omitted rather than
    # set to null.
    metadata = {
        "film_id": film["tmdb_id"],
        "title": film["title"],
    }
    year = film.get("year")
    if year not in (None, ""):
        metadata["year"] = int(year)

    rating = film.get("rating")
    if rating not in (None, ""):
        metadata["my_rating"] = float(rating)

    return metadata


def main():
    if not ENRICHED_JSON.exists():
        raise SystemExit(f"ERROR: {ENRICHED_JSON} not found -- run enrich_tmdb.py first.")

    with open(ENRICHED_JSON, encoding="utf-8") as f:
        films = json.load(f)
    print(f"Loaded {len(films)} film records from {ENRICHED_JSON}")

    ids = [str(film["tmdb_id"]) for film in films]
    documents = [build_chunk_text(film) for film in films]
    metadatas = [build_metadata(film) for film in films]

    print(f"Loading embedding model {EMBEDDING_MODEL_NAME} (first run downloads it, ~440MB)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Embedding {len(documents)} film chunks (documents get no query prefix)...")
    embeddings = model.encode(documents, show_progress_bar=True, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet on a first run -- fine

    # BGE embeddings are meant to be compared with cosine similarity.
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(f"Loaded {collection.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
