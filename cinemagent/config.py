"""
cinemagent.config -- central config + .env loader for the whole project.

Three jobs:

1. Load .env -- at import time, from the project-root .env regardless of
   the current working directory. load_dotenv()'s default search is relative
   to cwd, which breaks the moment a script is invoked from somewhere other
   than the project root. Pointing it at PROJECT_ROOT / ".env" explicitly
   makes it behave the same regardless of cwd, matching every other
   PROJECT_ROOT-relative path in this codebase. Any script that reads
   os.environ just needs to import something from this module first.

2. Tunable settings -- model names, retrieval knobs, the Postgres DB name,
   etc. Every script imports these from here directly, so there is exactly
   ONE place to change any of them, and they can all be overridden from .env
   without touching code.

3. Paths -- every data/output location, derived from PROJECT_ROOT (the repo
   root, one level above this package).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# This file lives at <repo>/cinemagent/config.py, so the repo root is two up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def _get_str(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _get_int(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"env var {name}={raw!r} is not a valid integer")


def _get_float(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"env var {name}={raw!r} is not a valid float")


# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"                # offline data pipeline scripts
SCHEMA_SQL = PIPELINE_DIR / "schema.sql"                # Postgres schema, applied by load_postgres.py
LETTERBOXD_EXPORT_DIR = DATA_DIR / "letterboxd_export"  # raw Letterboxd CSVs
INTERMEDIATE_DIR = DATA_DIR / "intermediate"            # everything the pipeline generates
MERGED_CSV = INTERMEDIATE_DIR / "letterboxd_merged.csv"
SANITY_REPORT = INTERMEDIATE_DIR / "sanity_report.txt"
ENRICHED_JSON = INTERMEDIATE_DIR / "films_enriched.json"
UNMATCHED_CSV = INTERMEDIATE_DIR / "unmatched.csv"
CHROMA_DIR = DATA_DIR / "chroma_db"
CACHE_DIR = DATA_DIR / "cache"
TMDB_CACHE_DIR = CACHE_DIR / "tmdb"
TONE_CACHE_DIR = CACHE_DIR / "tone"
LOGS_DIR = PROJECT_ROOT / "logs"
TRACE_LOG_PATH = LOGS_DIR / "agent_traces.jsonl"


# --- LLM: Gemini ---
# Everything (the agent and the tone-summary script) calls Gemini through the
# native GenAI SDK (google-genai).
GEMINI_MODEL = _get_str("GEMINI_MODEL", "gemini-3.6-flash")
# Tone summaries get a little sampling room rather than fully deterministic output.
TONE_TEMPERATURE = _get_float("TONE_TEMPERATURE", 0.3)

# --- Agent loop ---
AGENT_MAX_ITERATIONS = _get_int("AGENT_MAX_ITERATIONS", 6)

# --- Embeddings + vector store ---
EMBEDDING_MODEL_NAME = _get_str("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
CHROMA_COLLECTION_NAME = _get_str("CHROMA_COLLECTION_NAME", "films")
# Max TMDB keywords folded into each film's embedded chunk (chunking knob).
CHUNK_KEYWORD_CAP = _get_int("CHUNK_KEYWORD_CAP", 10)
# Phase 5 tuning: how many times build_chunk_text() (chunking.py) repeats
# the tone/review line in a film's chunk -- repetition is a simple way to
# boost a field's proportional weight in the pooled embedding without
# per-field weighted embeddings, which a single blended chunk string can't
# do natively. Ordering is deliberate: the review (Rohan's own reaction) is
# weighted heaviest, the LLM-generated tone line next, everything else
# (title, overview, keywords) gets no extra repetition at all.
CHUNK_TONE_WEIGHT = _get_int("CHUNK_TONE_WEIGHT", 2)
CHUNK_REVIEW_WEIGHT = _get_int("CHUNK_REVIEW_WEIGHT", 3)

# --- Hybrid retrieval + reranking ---
RETRIEVAL_N_RESULTS = _get_int("RETRIEVAL_N_RESULTS", 10)  # per-retriever candidate width
RRF_K = _get_int("RRF_K", 60)                              # reciprocal-rank-fusion constant
RERANKER_MODEL_NAME = _get_str("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
RERANK_TOP_N = _get_int("RERANK_TOP_N", 5)                 # candidates kept after rerank

# --- Postgres ---
DB_NAME = _get_str("DB_NAME", "entertainmentai")
