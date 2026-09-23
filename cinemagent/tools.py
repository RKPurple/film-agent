"""
TOOL_SCHEMAS (the function declarations for the google-genai Interactions API's `tools=`
parameter, flat {"type", "name", "description", "parameters"} dicts) and
call_tool() (dispatch).

Design notes:
- Six tools are thin wraps of already-tested queries.py functions
  (filter_by_actor/filter_by_director/filter_by_genre/filter_by_rating/
  cross_reference/get_all_genres), adding only argument unpacking and
  JSON-safe output. get_all_genres is a whole-table read so the agent can
  check the exact genre vocabulary before an exact-match filter_by_genre.
- search_my_history calls cinemagent.retrieval's retrieve() +
  confident_matches() only -- no LLM call of its own, since stacking two
  LLM calls would make citations survive paraphrasing twice. It returns raw
  confident matches and leaves all synthesis (including "not in corpus")
  to the agent.
- search_tmdb has no prior script. It reuses cinemagent.tmdb_client's
  tmdb_get()/fetch_details() but not enrich_tmdb.py's search_movie() (which requires an exact
  release-year match an ad hoc lookup rarely has), using its own looser
  _tmdb_search_by_title(). Results are tagged already_watched for
  "recommend something I haven't seen" scenario.
- Connection lifecycle: one Postgres connection per agent run. ToolContext
  holds it, build_tool_context() opens it, close_tool_context() closes it.
- Every dispatch function catches its own exceptions and returns
  {"error": "..."}; the loop's retry/reformulate/give-up policy is built in
  agent.py.

Usage (manual smoke test):
    python -m cinemagent.tools
"""

import json

from cinemagent import config  # noqa: F401 -- loads .env before the __main__ smoke test reads os.environ
from cinemagent import queries
from cinemagent.retrieval import confident_matches, load_retrieval_index, retrieve
from cinemagent.tmdb_client import fetch_details, tmdb_get


class ToolContext:
    """Everything a dispatch function needs, built once per agent run and
    threaded through every call_tool() invocation."""

    def __init__(self, conn, index, tmdb_api_key=None):
        self.conn = conn
        self.index = index                       # cinemagent.retrieval.RetrievalIndex
        self.tmdb_api_key = tmdb_api_key


def build_tool_context(tmdb_api_key=None):
    """Load/open everything once: the retrieval index (BM25, embedding model,
    Chroma collection, reranker) and the Postgres connection."""
    index = load_retrieval_index()
    conn = queries.get_connection()
    return ToolContext(conn, index, tmdb_api_key)


def close_tool_context(ctx):
    ctx.conn.close()


# ---------------------------------------------------------------------------
# Tool schemas -- handed to the google-genai Interactions API's `tools=` parameter as-is.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "filter_by_actor",
        "description": (
            "Find watched films where a given person appears in the cast. "
            "Name matching is case-insensitive substring, so partial names work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Actor name, full or partial."},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "filter_by_director",
        "description": (
            "Find watched films directed by a given person. Name matching is "
            "case-insensitive substring, so partial names work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Director name, full or partial."},
            },
            "required": ["name"],
        },
    },
    {
        "type": "function",
        "name": "filter_by_genre",
        "description": (
            "Find watched films tagged with an exact genre (e.g. 'Comedy', "
            "'Horror'). Case-insensitive but not a substring match -- use the "
            "real genre name, not a loose theme or vibe."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Exact genre name."},
            },
            "required": ["genre"],
        },
    },
    {
        "type": "function",
        "name": "filter_by_rating",
        "description": (
            "Find watched films within a personal-rating range (my_rating, "
            "0-5 scale). Either bound may be omitted for an open-ended range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "min_rating": {"type": "number", "description": "Lower bound, inclusive."},
                "max_rating": {"type": "number", "description": "Upper bound, inclusive."},
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "cross_reference",
        "description": (
            "Find every watched film a given person was involved in, in ANY "
            "role (acting or directing), tagged with which role applies to "
            "each film. Use this over filter_by_actor/filter_by_director when "
            "the question is about someone's overall overlap with the watch "
            "history, not one specific role."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "person_name": {"type": "string", "description": "Person name, full or partial."},
            },
            "required": ["person_name"],
        },
    },
    {
        "type": "function",
        "name": "get_all_genres",
        "description": (
            "List every genre name in the watch history's controlled "
            "vocabulary, each with a count of how many watched films use it. "
            "Takes no arguments. Call this before filter_by_genre when unsure "
            "of the exact genre name -- filter_by_genre matches exactly, not "
            "on loose themes or vibes."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "search_my_history",
        "description": (
            "Semantic search over the watched-film corpus for vibe/theme/mood "
            "queries that don't map to a rigid field -- e.g. 'movies that felt "
            "hopeful', 'something with a twisty plot'. Returns confident matches "
            "only (low-confidence retrieval results are already filtered out); "
            "an empty result means nothing in the watch history confidently "
            "matches, which should be reported honestly, not guessed around."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language description of what to find."},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "search_tmdb",
        "description": (
            "Look up a film on TMDB by title, INCLUDING films not in the local "
            "watch history -- use this for recommendation-style questions about "
            "films that may not have been watched yet. Each result is tagged "
            "already_watched so it's clear whether it's already in the local "
            "corpus (in which case search_my_history/filter_by_* have richer "
            "data on it) or genuinely new. Returns a tmdb_id -- pass that id to "
            "tmdb_recommendations to find films similar to this one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Film title to search for."},
                "year": {
                    "type": "integer",
                    "description": "Release year, if known -- helps disambiguate remakes/similarly-titled films.",
                },
            },
            "required": ["title"],
        },
    },
    {
        "type": "function",
        "name": "tmdb_recommendations",
        "description": (
            "Get TMDB's own 'similar films' recommendations for a given film, by "
            "its TMDB id -- use search_tmdb first to resolve a title into a "
            "tmdb_id, then pass that id here. Use this for 'recommend something "
            "like X' style questions. Results are tagged already_watched but are "
            "NOT pre-filtered -- exclude already-watched candidates and rank the "
            "rest yourself based on fit (genre/vibe similarity from the "
            "overviews, vote_average) before recommending them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tmdb_id": {
                    "type": "integer",
                    "description": "TMDB id of the film to find recommendations for (from a prior search_tmdb call).",
                },
            },
            "required": ["tmdb_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch -- one function per tool, each: (arguments dict, ctx) -> JSON-safe value.
# ---------------------------------------------------------------------------

def _rows_to_dicts(rows):
    return [dict(row) for row in rows]


def _dispatch_filter_by_actor(args, ctx):
    return {"films": _rows_to_dicts(queries.filter_by_actor(ctx.conn, args["name"]))}


def _dispatch_filter_by_director(args, ctx):
    return {"films": _rows_to_dicts(queries.filter_by_director(ctx.conn, args["name"]))}


def _dispatch_filter_by_genre(args, ctx):
    return {"films": _rows_to_dicts(queries.filter_by_genre(ctx.conn, args["genre"]))}


def _dispatch_filter_by_rating(args, ctx):
    rows = queries.filter_by_rating(
        ctx.conn, min_rating=args.get("min_rating"), max_rating=args.get("max_rating")
    )
    return {"films": _rows_to_dicts(rows)}


def _dispatch_cross_reference(args, ctx):
    return {"credits": _rows_to_dicts(queries.cross_reference(ctx.conn, args["person_name"]))}


def _dispatch_get_all_genres(args, ctx):
    return {"genres": _rows_to_dicts(queries.get_all_genres(ctx.conn))}


def _dispatch_search_my_history(args, ctx):
    reranked = retrieve(ctx.index, args["query"])
    matches = []
    for fid, confidence in confident_matches(reranked):
        film = ctx.index.film_lookup[fid]
        matches.append({
            "title": film["title"],
            "year": film.get("year"),
            "confidence": round(confidence, 3),
            "text": ctx.index.film_texts[fid],
        })
    return {"matches": matches}


def _tmdb_search_by_title(title, year, api_key):
    """Looser than enrich_tmdb.py's search_movie(): year is an optional
    disambiguator, not a required match. Falls back to the top result when
    no year is given or nothing matches it."""
    data = tmdb_get("/search/movie", api_key, {"query": title, "include_adult": "false"})
    results = data.get("results", [])
    if not results:
        return None
    if year is not None:
        year_str = str(year)
        for r in results:
            if (r.get("release_date") or "")[:4] == year_str:
                return r
    return results[0]


def _dispatch_search_tmdb(args, ctx):
    if not ctx.tmdb_api_key:
        return {"error": "TMDB_API_KEY not configured for this run -- search_tmdb is unavailable."}

    match = _tmdb_search_by_title(args["title"], args.get("year"), ctx.tmdb_api_key)
    if match is None:
        return {"error": f"No TMDB results for '{args['title']}'."}

    details = fetch_details(match["id"], ctx.tmdb_api_key)
    credits = details.get("credits", {})
    return {
        "tmdb_id": details["id"],
        "title": details.get("title"),
        "year": (details.get("release_date") or "")[:4] or None,
        "overview": details.get("overview"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "directors": [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"],
        "cast": [c["name"] for c in credits.get("cast", [])[:5]],
        "already_watched": str(details["id"]) in ctx.index.film_lookup,
    }


# Step 13: pairs with search_tmdb for a two-step chain (tmdb_id isn't known
# until search_tmdb returns one). Does NOT filter already_watched
# candidates -- that judgment plus ranking is left to the agent.
#
# Phase 5 tuning: TMDB's /recommendations is collaborative-filtering-based
# and, for a film in a franchise/collection, comes back dominated by that
# film's own sequels (fuzzy_rec_dark_knight got 8/10 Batman films).
# Standalone films (Sinners, Parasite) score well since there's nowhere
# franchise-y to cluster. The helpers below supplement /recommendations
# with a genre-based /discover/movie call, but ONLY when belongs_to_collection
# is set -- standalone films keep the exact prior code path, so they can't
# regress.
def _collection_member_ids(collection_id, api_key):
    """All tmdb_ids in a TMDB collection (franchise/series grouping), used
    to exclude franchise-mates from the discover-based supplement."""
    data = tmdb_get(f"/collection/{collection_id}", api_key)
    return {part["id"] for part in data.get("parts", [])}


def _discover_by_genre(genre_ids, exclude_ids, api_key, limit=10):
    """Genre-based supplement for recommendations, sorted by vote_average.
    with_genres uses '|' (OR) not ',' (AND) -- requiring every genre of a
    multi-genre film would usually be too narrow."""
    if not genre_ids:
        return []
    data = tmdb_get(
        "/discover/movie",
        api_key,
        {
            "with_genres": "|".join(str(g) for g in genre_ids),
            "sort_by": "vote_average.desc",
            "vote_count.gte": "200",  # filter out obscure/low-signal titles
        },
    )
    results = [r for r in data.get("results", []) if r["id"] not in exclude_ids]
    return results[:limit]


def _dispatch_tmdb_recommendations(args, ctx):
    if not ctx.tmdb_api_key:
        return {"error": "TMDB_API_KEY not configured for this run -- tmdb_recommendations is unavailable."}

    tmdb_id = args["tmdb_id"]
    source = tmdb_get(f"/movie/{tmdb_id}", ctx.tmdb_api_key)
    collection = source.get("belongs_to_collection")

    rec_data = tmdb_get(f"/movie/{tmdb_id}/recommendations", ctx.tmdb_api_key)
    rec_results = rec_data.get("results", [])

    if collection is None:
        # Standalone film -- unchanged behavior from before this tuning pass.
        merged = rec_results[:10]
    else:
        # Franchise film -- exclude every film TMDB itself lists as part of
        # this collection, then supplement with genre-based discover results
        # so franchise-mates don't crowd out everything else.
        franchise_ids = _collection_member_ids(collection["id"], ctx.tmdb_api_key)
        franchise_ids.add(tmdb_id)
        genre_ids = [g["id"] for g in source.get("genres", [])]

        non_franchise_recs = [r for r in rec_results if r["id"] not in franchise_ids]
        seen_ids = franchise_ids | {r["id"] for r in non_franchise_recs}
        discover_results = _discover_by_genre(genre_ids, seen_ids, ctx.tmdb_api_key)

        merged = (non_franchise_recs + discover_results)[:10]

    recommendations = [
        {
            "tmdb_id": r["id"],
            "title": r.get("title"),
            "year": (r.get("release_date") or "")[:4] or None,
            "overview": r.get("overview"),
            "vote_average": r.get("vote_average"),
            "already_watched": str(r["id"]) in ctx.index.film_lookup,
        }
        for r in merged
    ]
    return {"recommendations": recommendations}


DISPATCH = {
    "filter_by_actor": _dispatch_filter_by_actor,
    "filter_by_director": _dispatch_filter_by_director,
    "filter_by_genre": _dispatch_filter_by_genre,
    "filter_by_rating": _dispatch_filter_by_rating,
    "cross_reference": _dispatch_cross_reference,
    "get_all_genres": _dispatch_get_all_genres,
    "search_my_history": _dispatch_search_my_history,
    "search_tmdb": _dispatch_search_tmdb,
    "tmdb_recommendations": _dispatch_tmdb_recommendations,
}


def call_tool(name, arguments, ctx):
    """Look up and run a tool by name, always returning a JSON-safe dict,
    never raising. Unknown tool names and dispatch exceptions both come
    back as {"error": "..."} so the loop can always feed something back."""
    handler = DISPATCH.get(name)
    if handler is None:
        return {"error": f"Unknown tool '{name}'."}
    try:
        return handler(arguments, ctx)
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import os

    print("Building tool context (loading models, opening DB connection)...")
    ctx = build_tool_context(tmdb_api_key=os.environ.get("TMDB_API_KEY"))
    try:
        smoke_tests = [
            ("filter_by_actor", {"name": "Ryan Gosling"}),
            ("filter_by_director", {"name": "Celine Song"}),
            ("filter_by_genre", {"genre": "Comedy"}),
            ("filter_by_rating", {"min_rating": 4.5}),
            ("cross_reference", {"person_name": "Jordan"}),
            ("get_all_genres", {}),
            ("search_my_history", {"query": "movies that made me uneasy"}),
            ("search_tmdb", {"title": "Sinners", "year": 2025}),
            ("nonexistent_tool", {}),
        ]
        for name, args in smoke_tests:
            result = call_tool(name, args, ctx)
            print(f"\n-- {name}({args}) --")
            print(json.dumps(result, indent=2, default=str)[:1000])

        # tmdb_recommendations needs a real tmdb_id, which normally comes
        # from a prior search_tmdb call -- chained here the same way the
        # agent loop will chain them.
        sinners = call_tool("search_tmdb", {"title": "Sinners", "year": 2025}, ctx)
        if "tmdb_id" in sinners:
            result = call_tool("tmdb_recommendations", {"tmdb_id": sinners["tmdb_id"]}, ctx)
            print(f"\n-- tmdb_recommendations({{'tmdb_id': {sinners['tmdb_id']}}}) [chained from search_tmdb('Sinners')] --")
            print(json.dumps(result, indent=2, default=str)[:1000])
    finally:
        close_tool_context(ctx)
