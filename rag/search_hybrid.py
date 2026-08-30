"""
Hybrid retrieval: vector + BM25 fused via reciprocal rank fusion, then
cross-encoder reranking.

- BM25 indexes the SAME blended chunk text Chroma embeds (via
  build_chunk_text() in load_chroma.py), so comparisons test the
  retrieval mechanism, not different text.
- Tokens are lowercased, split on alphanumeric runs, stopword-filtered,
  and stemmed (snowballstemmer) identically at index and query time --
  without stemming + stopword removal, naive BM25 underperforms
  Postgres's built-in full-text search.
- BM25 index isn't persisted (unlike Chroma) -- rebuilt in memory each
  run; indexing ~170 short docs is instant.
- RRF: each method contributes 1/(k+rank) per doc by RANK POSITION (not
  raw incomparable scores), summed across methods, k=60 (standard).
- Reranker is a cross-encoder (BAAI/bge-reranker-base) that scores
  (query, document) jointly rather than comparing independent embeddings,
  catching e.g. "ABOUT anxiety" vs. "makes you FEEL anxious". It isn't
  precomputable, so it only runs over the narrow fused candidate pool.
- bm25_search() returns only strictly-positive scores: a 0.0 score is
  zero lexical overlap, not a weak match, so it shouldn't pad the fused
  pool out to N_RESULTS with zero-signal docs in films_enriched.json's
  arbitrary order. Abstract queries BM25 can't help with (e.g. "found
  family") now lean honestly on vector search alone.

Usage:
    python3 rag/search_hybrid.py
    (type a query, press enter, repeat; blank line or "quit" to exit)
"""

import json
import re
import sys
from pathlib import Path

import chromadb
import snowballstemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
for _p in (PROJECT_ROOT, DATA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from load_chroma import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    ENRICHED_JSON,
    QUERY_PREFIX,
    build_chunk_text,
)
from env_config import (
    RERANK_TOP_N,
    RERANKER_MODEL_NAME,
    RETRIEVAL_N_RESULTS,
    RRF_K,
)

# Historical name kept for existing importers / call-site defaults below.
N_RESULTS = RETRIEVAL_N_RESULTS

TOKEN_RE = re.compile(r"[a-z0-9]+")
STEMMER = snowballstemmer.stemmer("english")

# Standard English stopwords, filtered out BEFORE stemming (so this list
# stays in plain dictionary form, not stemmed form -- easier to read/edit).
# Without this, rank_bm25 has no protection against generic function words
# ("that", "made", "me") scoring incidental matches across a big chunk of
# the corpus -- Postgres's built-in full-text search gets this for free
# from its English dictionary, which is part of why naive BM25 loses to it
# without both stemming AND stopword removal, not just one.
#
# "movie"/"movies"/"film"/"films" are added on top of the standard list:
# they're not stopwords in general English, but every single document in
# THIS corpus is a movie, so they carry ~zero discriminating power here --
# the same reasoning as "the" carrying none in general English.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "were", "will", "with", "i", "you", "your",
    "yours", "we", "they", "this", "these", "those", "but", "or",
    "not", "no", "so", "if", "then", "than", "too", "very", "can",
    "could", "would", "should", "do", "does", "did", "just", "about",
    "into", "over", "after", "before", "up", "down", "out", "off",
    "again", "further", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "only", "own", "same", "me", "my",
    "myself", "him", "his", "her", "hers", "herself", "himself",
    "itself", "them", "their", "theirs", "themselves", "what",
    "which", "who", "whom", "am", "been", "being", "had", "having",
    "made", "make", "makes",
    "movie", "movies", "film", "films",
}


def tokenize(text):
    words = TOKEN_RE.findall(text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return STEMMER.stemWords(words)


def build_bm25_index(texts):
    tokenized_docs = [tokenize(text) for text in texts]
    return BM25Okapi(tokenized_docs)


def bm25_search(bm25, film_ids, query, n=N_RESULTS):
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(film_ids, scores), key=lambda pair: pair[1], reverse=True)
    # Only keep documents BM25 found real lexical signal for -- see the
    # module docstring's design note. A 0.0-score document isn't a weak
    # match, it's a tie with no signal at all, and shouldn't ride into the
    # fused pool just to pad the list out to n.
    return [film_id for film_id, score in ranked[:n] if score > 0]


def vector_search(model, collection, query, n=N_RESULTS):
    # BGE's asymmetric half: queries get the instruction prefix, documents
    # (already embedded in load_chroma.py) never did.
    query_embedding = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n)
    return results["ids"][0]  # Chroma always returns ids, regardless of `include`


def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def rerank(cross_encoder, query, candidate_ids, film_texts, top_n=RERANK_TOP_N):
    pairs = [(query, film_texts[fid]) for fid in candidate_ids]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidate_ids, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_n]


def main():
    if not ENRICHED_JSON.exists():
        raise SystemExit(f"ERROR: {ENRICHED_JSON} not found -- run enrich_tmdb.py first.")

    with open(ENRICHED_JSON, encoding="utf-8") as f:
        films = json.load(f)
    film_ids = [str(film["tmdb_id"]) for film in films]
    film_lookup = dict(zip(film_ids, films))

    texts = [build_chunk_text(film) for film in films]
    film_texts = dict(zip(film_ids, texts))

    def describe(film_id):
        film = film_lookup.get(film_id)
        return f"{film['title']} ({film.get('year', '?')})" if film else film_id

    print(f"Building BM25 index over {len(films)} films...")
    bm25 = build_bm25_index(texts)

    print(f"Loading {EMBEDDING_MODEL_NAME} and '{COLLECTION_NAME}' @ {CHROMA_DIR}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    print(f"Loading reranker {RERANKER_MODEL_NAME}...")
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

    print("Ready. Type a query, blank line or 'quit' to exit.\n")

    while True:
        query = input("query> ").strip()
        if not query or query.lower() in ("quit", "exit"):
            break

        vector_ids = vector_search(model, collection, query)
        bm25_ids = bm25_search(bm25, film_ids, query)
        fused = reciprocal_rank_fusion([vector_ids, bm25_ids])
        reranked = rerank(cross_encoder, query, [fid for fid, _score in fused], film_texts)

        # Intermediary stages (vector-only / bm25-only / fused) are computed
        # above but no longer printed -- only the final reranked results are
        # shown. Uncomment to inspect the pipeline stage by stage again.
        # print("\n-- vector-only --")
        # for rank, fid in enumerate(vector_ids, 1):
        #     print(f"  {rank}. {describe(fid)}")

        # print("-- bm25-only --")
        # for rank, fid in enumerate(bm25_ids, 1):
        #     print(f"  {rank}. {describe(fid)}")

        # print("-- fused (RRF) --")
        # for rank, (fid, score) in enumerate(fused, 1):
        #     print(f"  {rank}. {describe(fid)}  rrf_score={score:.4f}")

        print(f"\n-- reranked (top {RERANK_TOP_N}, cross-encoder) --")
        for rank, (fid, score) in enumerate(reranked, 1):
            print(f"  {rank}. {describe(fid)}  rerank_score={score:.4f}")
        print()


if __name__ == "__main__":
    main()
