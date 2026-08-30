"""
Single-tool RAG generation: retrieve -> confidence-gate -> generate w/ citations.

- Retrieval/reranking is reused from search_hybrid.py (vector + bm25 -> RRF ->
  rerank), not duplicated. This file only adds confidence calibration, the
  answer/no-answer decision, and prompt/citation formatting.
- The reranker (BAAI/bge-reranker-base) emits an unbounded logit; sigmoid is
  applied here to turn it into a 0-1 probability
- "Not in corpus" fallback: if nothing in the reranked top-N clears
  RERANK_CONFIDENCE_THRESHOLD, the LLM is never called. 0.5 is a placeholder
  ("more likely relevant than not")
- Generation model: GEMINI_MODEL, via Google's native GenAI SDK (google-genai).
  Previously Groq (openai/gpt-oss-120b), then Gemini's OpenAI-compatible endpoint
- Citations: context blocks are numbered [1], [2], ... after confidence
  filtering (so numbers stay contiguous). The model must cite claims with the
  matching bracket number, so citation-checking is just mapping numbers back
  to film_lookup.

Usage:
    python3 rag/generate.py
    (type a question, press enter, repeat; blank line or "quit" to exit)
"""

import json
import math
import os
import sys
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
for _p in (PROJECT_ROOT, DATA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from env_config import (
    GEMINI_MODEL,
    GENERATION_TEMPERATURE,
    RERANK_CONFIDENCE_THRESHOLD,
    load_env,
)
from load_chroma import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME, ENRICHED_JSON, build_chunk_text
from search_hybrid import (
    RERANK_TOP_N,
    RERANKER_MODEL_NAME,
    bm25_search,
    build_bm25_index,
    reciprocal_rank_fusion,
    rerank,
    vector_search,
)

NOT_IN_CORPUS_MESSAGE = (
    "I don't have anything in your watch history that confidently matches that -- "
    "rather than guess, I'm flagging that retrieval didn't find a strong match."
)

SYSTEM_PROMPT = (
    "You answer questions about a specific person's personal movie-watching history. "
    "You will be given numbered context blocks, each describing one film from that "
    "history (synopsis, themes, and sometimes the person's own short review). "
    "Answer ONLY using information in those numbered blocks -- never use outside "
    "knowledge about a film beyond what's given, and never mention a film that isn't "
    "in the numbered context. When you state something a specific block supports, "
    "cite it inline with its bracket number, e.g. [2]. If the provided context doesn't "
    "actually answer the question, say so plainly instead of stretching to answer anyway."
)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def retrieve(query, bm25, film_ids, model, collection, cross_encoder, film_texts):
    """Run the full retrieval pipeline (vector + BM25 -> RRF -> rerank) and
    return [(film_id, confidence_probability), ...], sorted by confidence
    descending. confidence_probability is the reranker's raw logit passed
    through sigmoid -- see module docstring."""
    vector_ids = vector_search(model, collection, query)
    bm25_ids = bm25_search(bm25, film_ids, query)
    fused = reciprocal_rank_fusion([vector_ids, bm25_ids])
    reranked = rerank(cross_encoder, query, [fid for fid, _score in fused], film_texts)
    return [(fid, sigmoid(raw_score)) for fid, raw_score in reranked]


def build_context_blocks(reranked, film_lookup, film_texts, threshold=RERANK_CONFIDENCE_THRESHOLD):
    """Number every reranked candidate, then keep only the ones confident
    enough to hand to the LLM. Numbering happens before filtering so a
    citation number always matches a real position, but low-confidence
    films are dropped before the model ever sees them."""
    blocks = []
    for i, (fid, confidence) in enumerate(reranked, start=1):
        if confidence < threshold:
            continue
        film = film_lookup[fid]
        title_year = f"{film['title']} ({film.get('year', '?')})"
        blocks.append({
            "index": i,
            "film_id": fid,
            "title_year": title_year,
            "confidence": confidence,
            "text": film_texts[fid],
        })
    return blocks


def build_user_message(query, blocks):
    context = "\n\n".join(f"[{b['index']}] {b['title_year']}\n{b['text']}" for b in blocks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(client, query, blocks):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_user_message(query, blocks),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=GENERATION_TEMPERATURE,
        ),
    )
    return response.text


def main():
    load_env()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set the GEMINI_API_KEY environment variable first (see script docstring).")

    if not ENRICHED_JSON.exists():
        raise SystemExit(f"ERROR: {ENRICHED_JSON} not found -- run enrich_tmdb.py first.")

    with open(ENRICHED_JSON, encoding="utf-8") as f:
        films = json.load(f)
    film_ids = [str(film["tmdb_id"]) for film in films]
    film_lookup = dict(zip(film_ids, films))

    texts = [build_chunk_text(film) for film in films]
    film_texts = dict(zip(film_ids, texts))

    print(f"Building BM25 index over {len(films)} films...")
    bm25 = build_bm25_index(texts)

    print(f"Loading {EMBEDDING_MODEL_NAME} and '{COLLECTION_NAME}' @ {CHROMA_DIR}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client_chroma.get_collection(COLLECTION_NAME)

    print(f"Loading reranker {RERANKER_MODEL_NAME}...")
    cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

    gemini_client = genai.Client(api_key=api_key)

    print(f"Ready (generation model: {GEMINI_MODEL}). Type a question, blank line or 'quit' to exit.\n")

    while True:
        query = input("ask> ").strip()
        if not query or query.lower() in ("quit", "exit"):
            break

        reranked = retrieve(query, bm25, film_ids, model, collection, cross_encoder, film_texts)
        blocks = build_context_blocks(reranked, film_lookup, film_texts)

        print(f"\n-- retrieval confidence (top {RERANK_TOP_N}, post-sigmoid) --")
        for fid, confidence in reranked:
            flag = "" if confidence >= RERANK_CONFIDENCE_THRESHOLD else "  [below threshold]"
            print(f"  {film_lookup[fid]['title']} ({film_lookup[fid].get('year', '?')})  "
                  f"confidence={confidence:.3f}{flag}")

        if not blocks:
            print(f"\n{NOT_IN_CORPUS_MESSAGE}\n")
            continue

        print(f"\nGenerating answer from {len(blocks)} confident source(s)...")
        answer = generate_answer(gemini_client, query, blocks)

        print(f"\n{answer}\n")
        print("Sources:")
        for b in blocks:
            print(f"  [{b['index']}] {b['title_year']}  (confidence={b['confidence']:.3f})")
        print()


if __name__ == "__main__":
    main()
