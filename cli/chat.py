"""
Interactive chat with the cinemagent agent over the watch history.

Needs GEMINI_API_KEY; TMDB_API_KEY is optional (only search_tmdb /
tmdb_recommendations use it). Every question is appended to
logs/agent_traces.jsonl.

Usage:
    python3 cli/chat.py
    (type a question, press enter, repeat; blank line or "quit" to exit)
"""

import os

from google import genai

from cinemagent.agent import run_agent
from cinemagent.config import GEMINI_MODEL
from cinemagent.tools import build_tool_context, close_tool_context


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set the GEMINI_API_KEY environment variable first (see module docstring).")

    tmdb_api_key = os.environ.get("TMDB_API_KEY")  # optional -- only needed for search_tmdb

    print("Building tool context (loading models, opening DB connection)...")
    ctx = build_tool_context(tmdb_api_key=tmdb_api_key)
    client = genai.Client(api_key=api_key)

    print(f"Ready (model: {GEMINI_MODEL}). Type a question, blank line or 'quit' to exit.\n")
    try:
        while True:
            query = input("agent> ").strip()
            if not query or query.lower() in ("quit", "exit"):
                break
            answer = run_agent(query, client, ctx)
            print(f"\n{answer}\n")
    finally:
        close_tool_context(ctx)


if __name__ == "__main__":
    main()
