"""
cinemagent.citations -- which known films an answer cites.

Pure functions: no model loads, no DB. The agent cites films as
"Title (Year)" (see SYSTEM_PROMPT), so resolve_citations() looks for each
candidate film's own "Title (Year)" string in the text -- corpus-driven,
never by regex-parsing titles out of the text, since titles contain
parentheses, colons, digits and other punctuation ("(500) Days of Summer",
"Joker: Folie à Deux", "Ted 2").

- Boundary: a match must not start mid-word, so "Us (2019)" can't match
  inside "Bus (2019)".
- Overlaps: when two candidates match overlapping spans, the longest wins,
  so "Us (2019)" isn't also cited inside a longer "... Us (2019)" title that
  is itself a candidate.
- Exact match first; only a film with no exact match gets the normalized
  fallback (case, curly vs straight quotes, & vs "and", runs of
  whitespace, markdown emphasis markers like "*Title* (2019)", and
  markdown backslash escapes like "Thunderbolts\\*"). The
  fallback deliberately stops there -- a false citation is worse than a
  missed one.

unresolved_year_phrases() returns the "(YYYY)"-anchored phrases that no
candidate claimed, for diagnostics (typically films outside the candidate
set, e.g. unwatched TMDB recommendations).
"""

import re

YEAR_RE = re.compile(r"\((\d{4})\)")

_QUOTE_MAP = {
    "‘": "'", "’": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
}
_MARKDOWN_EMPHASIS = set("*_`")
_MARKDOWN_ESCAPABLE = set("\\`*_{}[]()#+-.!|")


def film_label(film):
    return f"{film['title']} ({film.get('year', '')})"


def _normalize(text):
    """(normalized text, index map) where map[i] is the index in `text` of
    normalized character i -- so normalized matches map back to original
    spans for ordering and overlap checks."""
    out, index_map = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch in _MARKDOWN_EMPHASIS:
            continue
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in _MARKDOWN_ESCAPABLE:
            continue  # markdown escape, e.g. "Thunderbolts\\*"
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            index_map.append(i)
            prev_space = True
            continue
        prev_space = False
        if ch == "&":
            piece = "and"
        else:
            piece = _QUOTE_MAP.get(ch, ch).lower()
        for c in piece:
            out.append(c)
            index_map.append(i)
    return "".join(out), index_map


def _normalize_label(label):
    return _normalize(label)[0].strip()


def _starts_at_boundary(text, start):
    return start == 0 or not text[start - 1].isalnum()


def _find_all(haystack, needle):
    start = haystack.find(needle)
    while start != -1:
        if _starts_at_boundary(haystack, start):
            yield start
        start = haystack.find(needle, start + 1)


def _candidate_spans(text, films):
    """Every (start, end, film) occurrence in `text`, exact matches for a
    film first, normalized ones only for films with no exact match. Spans
    are in original-text coordinates, end exclusive."""
    spans = []
    normalized = None
    for film in films:
        label = film_label(film)
        exact = [(s, s + len(label), film) for s in _find_all(text, label)]
        if exact:
            spans.extend(exact)
            continue
        if normalized is None:
            normalized = _normalize(text)
        norm_text, index_map = normalized
        norm_label = _normalize_label(label)
        if not norm_label:
            continue
        for s in _find_all(norm_text, norm_label):
            spans.append((index_map[s], index_map[s + len(norm_label) - 1] + 1, film))
    return spans


def _resolve_spans(text, films):
    """Non-overlapping (start, end, film) spans, longest match winning."""
    accepted = []
    for start, end, film in sorted(_candidate_spans(text, films), key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(end <= a_start or start >= a_end for a_start, a_end, _ in accepted):
            accepted.append((start, end, film))
    return sorted(accepted, key=lambda s: s[0])


def resolve_citations(text, films):
    """Films from `films` (dicts with tmdb_id, title, year) cited in `text`
    as "Title (Year)", in order of first appearance, de-duplicated by
    tmdb_id."""
    if not text:
        return []
    cited, seen = [], set()
    for _start, _end, film in _resolve_spans(text, list(films)):
        key = str(film["tmdb_id"])
        if key not in seen:
            seen.add(key)
            cited.append(film)
    return cited


def unresolved_year_phrases(text, films, max_words=8):
    """"(YYYY)"-anchored phrases in `text` that no film in `films` claimed:
    each is the "(YYYY)" plus up to max_words preceding words on the same
    line, for eyeballing what the matcher missed."""
    if not text:
        return []
    claimed_ends = {end for _start, end, _film in _resolve_spans(text, list(films))}
    phrases = []
    for m in YEAR_RE.finditer(text):
        if m.end() in claimed_ends:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        words = text[line_start:m.start()].split()
        phrases.append(" ".join(words[-max_words:] + [m.group(0)]))
    return phrases
