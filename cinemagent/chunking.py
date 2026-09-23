"""
cinemagent.chunking -- the one definition of a film's blended chunk text.

Shared by the offline Chroma loader (pipeline/load_chroma.py, which embeds it)
and the online BM25 index (cinemagent/retrieval.py, which tokenizes it).
Both retrievers must see exactly the same text for every film, so it is
built in one place and must never diverge between them.
"""

from cinemagent.config import CHUNK_KEYWORD_CAP, CHUNK_REVIEW_WEIGHT, CHUNK_TONE_WEIGHT

# BGE's asymmetric-retrieval instruction prefix. Only ever applied to QUERIES
# at query time (in cinemagent/retrieval.py) -- never to documents, which is why
# build_chunk_text() below never uses it. Kept here so both sides import the
# same literal instead of risking two copies drifting apart.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_chunk_text(film):
    lines = [f"{film['title']} ({film.get('year', '')})", film.get("overview") or ""]

    keywords = (film.get("keywords") or [])[:CHUNK_KEYWORD_CAP]
    if keywords:
        lines.append(f"Themes: {', '.join(keywords)}")
    tone = film.get("tone_summary")
    if tone:
        lines.extend([f"Tone: {tone}"] * CHUNK_TONE_WEIGHT)

    review = film.get("review_text")
    if review:
        lines.extend([f"My take: {review}"] * CHUNK_REVIEW_WEIGHT)

    return "\n".join(lines)
