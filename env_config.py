"""
Central config + .env loader for the whole project.

Two jobs:

1. load_env() -- points python-dotenv at the project-root .env regardless of
   the current working directory. load_dotenv()'s default search is relative
   to cwd, which breaks the moment a script is invoked from somewhere other
   than the project root (e.g. `cd agent && python3 agent.py`). Pointing it
   at PROJECT_ROOT / ".env" explicitly makes it behave the same regardless
   of cwd, matching every other PROJECT_ROOT-relative path in this codebase.

2. Tunable settings -- model names, retrieval knobs, the Postgres DB name,
   etc. Every script imports these from here (directly, or re-exported under
   their historical names by load_chroma.py / search_hybrid.py / generate.py
   / agent.py), so there is exactly ONE place to change any of them, and
   they can all be overridden from .env without touching code.

.env is loaded at import time, so the constants below resolve even if a
script forgets to call load_env() first. load_env() is kept for existing
call sites and is safe to call again.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def load_env():
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


# --- LLM: Gemini ---
# generate.py uses the native GenAI SDK (google-genai); agent.py and the tone-
# summary script still use the OpenAI-compatible endpoint below.
GEMINI_MODEL = _get_str("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_BASE_URL = _get_str(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
# Generation is kept near-deterministic; tone summaries get a little more room.
GENERATION_TEMPERATURE = _get_float("GENERATION_TEMPERATURE", 0.2)
TONE_TEMPERATURE = _get_float("TONE_TEMPERATURE", 0.3)

# --- Agent loop ---
AGENT_MAX_ITERATIONS = _get_int("AGENT_MAX_ITERATIONS", 6)

# --- Embeddings + vector store ---
EMBEDDING_MODEL_NAME = _get_str("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
CHROMA_COLLECTION_NAME = _get_str("CHROMA_COLLECTION_NAME", "films")
# Max TMDB keywords folded into each film's embedded chunk (chunking knob).
CHUNK_KEYWORD_CAP = _get_int("CHUNK_KEYWORD_CAP", 10)
# Phase 5 tuning: how many times build_chunk_text() (load_chroma.py) repeats
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
RERANK_CONFIDENCE_THRESHOLD = _get_float("RERANK_CONFIDENCE_THRESHOLD", 0.5)

# --- Postgres ---
DB_NAME = _get_str("DB_NAME", "entertainmentai")
