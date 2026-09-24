# EntertainmentAI

A retrieval-augmented, agentic question-answering system built over my own movie-watching history. It combines a personal [Letterboxd](https://letterboxd.com) export with live [TMDB](https://www.themoviedb.org) metadata, indexes it with hybrid (vector + keyword) retrieval and cross-encoder reranking, and answers natural-language questions through an LLM agent that can call structured tools, semantic search, and live TMDB lookups as needed.

Built as a hands-on exercise in real RAG and agentic-tooling patterns — not toy versions of them.

## What it can answer

- **Structured lookups**: "What movies with Idris Elba have I watched?", "What war movies have I seen?", "What's the lowest-rated thing I've watched?"
- **Cross-referencing**: "What's my history with Jon Favreau?" (actor and director credits together)
- **Vibe / theme search**: "What have I watched that felt uneasy or unsettling?", "What's something in my history that surprised me with how much I ended up loving it?"
- **Recommendations**: "Recommend something like Parasite that I haven't seen" — resolves the film on TMDB, pulls candidates, filters out what's already watched, and ranks the rest
- **Adversarial / ambiguous questions**: "What Marvel movies did I really enjoy?" (no explicit rating cutoff given — the agent has to pick a reasonable bar and apply it)

## How it works

```
Letterboxd CSV export ───┐
                         ├─▶ merge + TMDB enrichment ──▶ Postgres (structured metadata)
TMDB API ────────────────┘                          └──▶ Chroma (vector embeddings)

User query ──▶ Agent (Gemini, function-calling)
                 │
                 ├──▶ structured tools (filter_by_actor, filter_by_genre, filter_by_rating, get_all_genres, ...) ──▶ Postgres
                 ├──▶ search_my_history (hybrid retrieval: vector + BM25 → RRF fusion → cross-encoder rerank) ──▶ Chroma
                 └──▶ search_tmdb / tmdb_recommendations ──▶ live TMDB API
                 │
                 ▼
         synthesized answer, citing specific films
```

## Key techniques

- **Hybrid retrieval**: vector similarity (BGE embeddings via Chroma) fused with BM25 keyword search via reciprocal rank fusion, so both semantic and literal-keyword matches contribute.
- **Cross-encoder reranking**: the fused candidate pool gets rescored by a `BAAI/bge-reranker-base` cross-encoder, which jointly attends over the query and each candidate rather than comparing precomputed vectors — catches distinctions a bi-encoder can't (e.g. "text is *about* anxiety" vs. "text would *make you feel* anxious").
- **Agent-judged relevance**: `search_my_history` returns the reranked top films with a 0–1 confidence (one sigmoid over the cross-encoder logit) and each film's evidence text. There's no confidence cutoff: calibration against hand-labeled queries (`eval/calibrate_threshold.py`) showed reranker scores order results within a query but no global threshold separates relevant from irrelevant ones. Instead the agent's system prompt treats results as the closest films, not confirmed matches, and requires it to judge fit from each film's overview, tone summary and review, citing only films that fit and saying plainly when nothing in your watch history does.
- **Weighted chunk embedding**: each film's embedded text repeats the personal review most heavily, then the tone summary, then everything else once — a repetition-based way to bias a single blended embedding toward the most personally meaningful signal.
- **Agentic tool routing**: a 9-tool agent (4 structured filters, cross-reference, a genre-vocabulary lookup, semantic search, TMDB search, TMDB recommendations) that chains multiple tools per turn when a question needs it, short-circuits exact-duplicate tool calls, and logs a full reasoning trace per run.
- **Eval-driven tuning**: a 24-question eval set split between auto-gradable "checkable" questions (exact expected results), manually-rated "fuzzy" questions (recommendations, vibe search), and adversarial trap questions (subjects absent from the watch history, graded manually) — every retrieval/prompt change gets checked against all of them before/after.

## Tech stack

| Layer | Tool |
|---|---|
| LLM | Google Gemini (`gemini-3.6-flash`), via the native `google-genai` SDK |
| Vector store | [Chroma](https://www.trychroma.com/) (local, persistent) |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) |
| Reranker | `BAAI/bge-reranker-base` (cross-encoder) |
| Keyword search | BM25 (`rank-bm25`), Snowball-stemmed |
| Structured store | Postgres |
| Metadata source | [TMDB API](https://www.themoviedb.org/documentation/api) |

## Project structure

```
cinemagent/   The installable package -- everything the live system runs on.
  config.py       Loads .env; every tunable setting and data path in one place.
  queries.py      Structured Postgres queries (actor, director, genre, rating, cross-reference).
  chunking.py     The one definition of a film's embedded/BM25 chunk text.
  retrieval.py    Hybrid retrieval: vector + BM25 -> RRF fusion -> cross-encoder rerank.
  citations.py    Resolves the "Title (Year)" films an answer cites (used by eval grading).
  tmdb_client.py  Shared TMDB HTTP client (pipeline enrichment + live agent tools).
  tools.py        Agent tool schemas and dispatch.
  agent.py        The tool-calling agent loop, duplicate-call guard, trace logging.
pipeline/     Offline data pipeline, run as scripts.
  build_dataset.py            Runs merge -> enrich -> tone in order, confirming between steps.
  merge_letterboxd.py         Merge the raw Letterboxd CSVs into one dataset.
  enrich_tmdb.py              Enrich each film with TMDB metadata.
  generate_tone_summaries.py  LLM-generated tone/mood line per film.
  load_postgres.py            Load the structured store (applies schema.sql).
  load_chroma.py              Build the vector store.
  schema.sql                  Postgres schema for the structured metadata store.
cli/          Interactive entry points.
  chat.py         Chat with the agent.
  search.py       Retrieval debugger (no LLM); --stages shows every pipeline stage.
eval/         The eval harness: question set, runner, and a result viewer.
data/         Data only, no code.
  letterboxd_export/  The raw unzipped Letterboxd CSV export (input).
  intermediate/       Generated datasets: letterboxd_merged.csv,
                      films_enriched.json, unmatched.csv, sanity_report.txt.
  cache/              Raw cached API responses (cache/tmdb/, cache/tone/)
                      so re-runs never re-hit the APIs.
  chroma_db/          The persistent Chroma vector store.
```

## Getting started

**Prerequisites**: Python 3.10+, a local Postgres instance, a [TMDB API key](https://www.themoviedb.org/settings/api) (free), a [Gemini API key](https://aistudio.google.com/apikey), and your own Letterboxd data export (Settings → Import & Export → Export Data).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .    # installs the cinemagent package so every script can import it
```

Create a `.env` in the project root:

```
TMDB_API_KEY=...
GEMINI_API_KEY=...
DB_NAME=entertainmentai
```

Unzip your Letterboxd export into `data/letterboxd_export/`, then build the dataset:

```bash
# One command: runs merge -> enrich -> tone summaries in order, pausing to
# confirm between steps and flagging anything worth a manual check
# (sanity-report warnings, low-confidence TMDB matches, failed tone calls).
# Outputs land in data/intermediate/; API responses cache in data/cache/.
python3 pipeline/build_dataset.py
```

<details>
<summary>...or run the three generation steps individually</summary>

```bash
python3 pipeline/merge_letterboxd.py         # merge the raw CSVs into one dataset
python3 pipeline/enrich_tmdb.py              # enrich each film with TMDB metadata
python3 pipeline/generate_tone_summaries.py  # (optional) LLM-generated tone/mood line per film
```
</details>

Then load the two stores:

```bash
psql entertainmentai -f pipeline/schema.sql   # create the Postgres schema
python3 pipeline/load_postgres.py             # load structured metadata
python3 pipeline/load_chroma.py               # build the vector store
```

Then chat with the agent:

```bash
python3 cli/chat.py           # full agent, interactive
```

inspect retrieval directly (no LLM calls):

```bash
python3 cli/search.py            # final reranked results per query
python3 cli/search.py --stages   # also show vector-only, BM25-only and fused (RRF) lists
```

or run the eval set:

```bash
python3 eval/run_eval.py
python3 eval/show_results.py  # pretty-print the most recent run
```

## Eval results

The eval set has 24 questions split into three categories: **checkable** (10 structured questions with a known-correct expected answer, auto-graded against Postgres directly), **fuzzy** (6 recommendation and vibe/theme questions, rated manually since there's no single correct answer), and **adversarial** (8 deliberately ambiguous or trap questions, e.g. asking about "documentaries" or werewolf movies when none exist in the watch history, to check the system reports that honestly rather than fabricating one).

Grading modes: `auto_exact_set`, `auto_empty`, `auto_intersection` and `auto_genre_rating_threshold` compare a tool's returned film set to stored ground truth (refreshed from Postgres by `eval/refresh_ground_truth.py`); `auto_duplicate_guard` checks the duplicate-call guard fired; `manual` is scored by hand. 11 questions are auto-graded (the 10 checkable ones plus the duplicate-guard check) and 13 are manual: the 6 fuzzy questions and 7 adversarial ones. Adversarial "nothing fits" questions are graded manually because a correct answer may name watched films specifically to rule them out, which an automatic citation check can't tell apart from presenting them as matches.

In the latest run, all 11 auto-graded questions pass, with correct tool routing on every actor/director/genre/rating/cross-reference question. The fuzzy category is judged qualitatively per run — see `eval/results/` for the full history of runs and the reasoning behind each retrieval/prompt change.
