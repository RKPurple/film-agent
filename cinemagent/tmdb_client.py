"""
cinemagent.tmdb_client -- the shared TMDB v3 HTTP client.

Used by both the offline enrichment pipeline
(pipeline/enrich_tmdb.py) and the agent's live TMDB
tools (cinemagent.tools). Pipeline-specific pieces -- strict year matching,
the on-disk response cache, request pacing -- stay in enrich_tmdb.py.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request

TMDB_BASE = "https://api.themoviedb.org/3"
MAX_RETRIES = 3


def tmdb_get(path, api_key, params=None):
    """GET a TMDB endpoint with retry/backoff on transient errors and 429s."""
    params = dict(params or {})
    params["api_key"] = api_key
    url = f"{TMDB_BASE}{path}?{urllib.parse.urlencode(params)}"

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = int(e.headers.get("Retry-After", "2"))
                time.sleep(retry_after)
                last_err = e
                continue
            if e.code == 401:
                raise RuntimeError(
                    "TMDB returned 401 Unauthorized -- check TMDB_API_KEY is a valid v3 API key."
                ) from e
            last_err = e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"TMDB request failed after {MAX_RETRIES} attempts: {url}") from last_err


def fetch_details(tmdb_id, api_key):
    return tmdb_get(f"/movie/{tmdb_id}", api_key, {"append_to_response": "credits,keywords"})
