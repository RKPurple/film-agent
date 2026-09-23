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
- Like pipeline/load_postgres.py, this is idempotent by being destructive: every
  run drops and recreates the Chroma collection and reloads everything from
  films_enriched.json, since that JSON is the single source of truth and
  nothing in Chroma is ever hand-edited.

Usage:
    python3 pipeline/load_chroma.py

First run downloads the bge-base-en-v1.5 model weights (~440MB) from
Hugging Face -- needs network, and needs the venv active with
requirements.txt installed
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer

from cinemagent.chunking import build_chunk_text
from cinemagent.config import CHROMA_COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL_NAME, ENRICHED_JSON


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
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet on a first run -- fine

    # BGE embeddings are meant to be compared with cosine similarity.
    collection = client.create_collection(CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(f"Loaded {collection.count()} chunks into '{CHROMA_COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
