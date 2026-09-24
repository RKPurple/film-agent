"""
Interactive chat with the cinemagent agent over the watch history.

Needs GEMINI_API_KEY; TMDB_API_KEY is optional (only search_tmdb /
tmdb_recommendations use it). Questions form one conversation: each turn
continues from the last completed one (cinemagent.conversation). Every
question is appended to logs/agent_traces.jsonl.

Usage:
    python3 cli/chat.py
    (type a question, press enter, repeat; "/new" starts a fresh
    conversation; blank line or "quit" to exit)
"""

import json
import os
from contextlib import closing

from google import genai

from cinemagent.conversation import Conversation
from cinemagent.config import GEMINI_MODEL
from cinemagent.tools import build_tool_context, close_tool_context


def render(events):
    """Print a turn's events: one trace line per tool call, the answer, any
    loop error, and a prompt-cache line from done. Returns the done event
    (None if the stream ended without one)."""
    calls = {}
    done = None
    for event in events:
        kind = event["type"]
        if kind == "tool_call":
            calls[event["id"]] = event
        elif kind == "tool_result":
            call = calls[event["id"]]
            preview = json.dumps(event["result"], default=str)[:200]
            print(f"  [iter {event['iteration']}] {call['name']}({call['arguments']}) -> {preview}")
        elif kind == "answer":
            print(f"\n{event['text']}\n")
        elif kind == "error":
            print(f"\n  [error in {event['where']}: {event['error_type']}: {event['message']}]\n")
        elif kind == "done":
            done = event
            usage = event["usage"] or {}
            inputs, cached = usage.get("input_tokens"), usage.get("cached_tokens") or 0
            if inputs:
                print(f"  [prompt cache: {cached}/{inputs} tokens cached ({cached / inputs * 100:.0f}%)]\n")
            else:
                print("  [prompt cache: n/a (no usage reported)]\n")
    return done


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set the GEMINI_API_KEY environment variable first (see module docstring).")

    tmdb_api_key = os.environ.get("TMDB_API_KEY")  # optional -- only needed for search_tmdb

    print("Building tool context (loading models, opening DB connection)...")
    ctx = build_tool_context(tmdb_api_key=tmdb_api_key)
    client = genai.Client(api_key=api_key)

    conversation = Conversation(client, ctx)
    print(f"Ready (model: {GEMINI_MODEL}). Type a question, '/new' for a fresh conversation, "
          f"blank line or 'quit' to exit.\n")
    try:
        while True:
            query = input(f"agent[{conversation.turn + 1}]> ").strip()
            if not query or query.lower() in ("quit", "exit"):
                break
            if query == "/new":
                conversation.reset()
                print("  [started a new conversation]\n")
                continue
            # closing(): if rendering stops early (e.g. Ctrl-C), the trace
            # record is still written right away, as "aborted".
            with closing(conversation.ask(query)) as events:
                done = render(events)
            if not (done and done["resumable"]):
                if conversation.last_interaction_id:
                    print("  [this turn didn't complete -- the next question continues from the last completed turn]\n")
                else:
                    print("  [this turn didn't complete -- the next question starts without earlier context]\n")
    finally:
        close_tool_context(ctx)


if __name__ == "__main__":
    main()
