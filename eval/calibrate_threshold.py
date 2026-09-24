"""
Calibrate reranker confidence thresholds against hand-labeled queries.

Reads eval/calibration_queries.json (must have status "labeled"): positive
queries list `relevant` films (and optionally `borderline` ones, excluded
from both sides of every metric); negative queries should have no good
match in the watch history. The `candidates` field is ignored.

For every query it scores ALL films with the reranker under two document
text variants:
  - current: index.film_texts, exactly what the reranker sees today
  - dedup:   the same content with each line once (no Tone / My take
             repetition), built here to mirror cinemagent/chunking.py
and reports per-query ranks/AUC, a threshold sweep, a top-score vs.
margin-over-median comparison, and -- for the current variant -- how many
relevant films actually reach the live pipeline's fused pool and reranked
top N (a recall ceiling no threshold can fix).

Scores are TRUE cross-encoder logits. sentence-transformers' CrossEncoder
.predict() applies a sigmoid by default for single-label models like
bge-reranker-base, so this script passes activation_fct=Identity to get
logits, exactly as cinemagent.retrieval.rerank() does. The live pipeline
has no confidence gate (this script's results are why), so the threshold
sweep is analysis only.

Read-only with respect to cinemagent/, config, data and the eval
questions; writes only
eval/calibration_results/run_<UTC timestamp>_<model slug>.{json,md}.

--reranker swaps in a different CrossEncoder for ALL scoring, including the
live-pipeline view (set on the loaded index before retrieve_stages()). Each
run also records model load time, median scoring time (one query vs. all
films, and one query's live fused pool), peak process RSS after the model
loads, and the device.

Usage:
    python3 eval/calibrate_threshold.py
    python3 eval/calibrate_threshold.py --reranker BAAI/bge-reranker-large
"""

import argparse
import gc
import json
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import CrossEncoder

from cinemagent.config import (
    CHUNK_KEYWORD_CAP,
    CHUNK_REVIEW_WEIGHT,
    CHUNK_TONE_WEIGHT,
    EMBEDDING_MODEL_NAME,
    RERANK_TOP_N,
    RERANKER_MODEL_NAME,
    RETRIEVAL_N_RESULTS,
    RRF_K,
)
from cinemagent.retrieval import load_retrieval_index, rerank, retrieve_stages, sigmoid

EVAL_DIR = Path(__file__).resolve().parent
LABELS_PATH = EVAL_DIR / "calibration_queries.json"
RESULTS_DIR = EVAL_DIR / "calibration_results"

TITLE_YEAR_RE = re.compile(r"^(.*) \((\d{4})\)$")
VARIANTS = ("current", "dedup")
SWEEP_TABLE_ROWS = 25  # evenly spaced thresholds shown in the report table


# ---------------------------------------------------------------- loading

def load_labels():
    with open(LABELS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "labeled":
        sys.exit(f"ERROR: {LABELS_PATH} status is {data.get('status')!r}, not 'labeled' -- label it first.")
    return data["queries"]


def resolve_titles(queries, films):
    """Replace each query's relevant/borderline "Title (Year)" lists with
    tmdb_id (str) sets. Exits listing every problem if anything fails."""
    by_title_year = {}
    for film in films:
        by_title_year.setdefault((film["title"], str(film.get("year", ""))), []).append(str(film["tmdb_id"]))

    errors = []
    for q in queries:
        if q["type"] not in ("positive", "negative"):
            errors.append(f"{q['id']}: unknown type {q['type']!r}")
        for field in ("relevant", "borderline"):
            ids = set()
            for label in q.get(field, []):
                m = TITLE_YEAR_RE.match(label)
                matches = by_title_year.get((m.group(1), m.group(2)), []) if m else []
                if len(matches) != 1:
                    why = "malformed" if not m else ("not found" if not matches else f"ambiguous {matches}")
                    errors.append(f"{q['id']}.{field}: {label!r} {why}")
                else:
                    ids.add(matches[0])
            q[f"{field}_ids"] = ids
        if q["type"] == "positive" and not q["relevant_ids"] and not any(e.startswith(q["id"]) for e in errors):
            errors.append(f"{q['id']}: positive query with no relevant films")
        if q["type"] == "negative" and (q.get("relevant") or q.get("borderline")):
            errors.append(f"{q['id']}: negative query has relevant/borderline labels")
        overlap = q["relevant_ids"] & q["borderline_ids"]
        if overlap:
            errors.append(f"{q['id']}: films in both relevant and borderline: {sorted(overlap)}")

    if errors:
        sys.exit("ERROR: calibration labels failed to resolve:\n  " + "\n  ".join(errors))


# --------------------------------------------------------- text variants

def dedup_entries(film):
    """The distinct entries of a film's chunk, in chunking.py's order:
    [(text, repeat_weight_in_current_chunk), ...]."""
    entries = [(f"{film['title']} ({film.get('year', '')})", 1), (film.get("overview") or "", 1)]
    keywords = (film.get("keywords") or [])[:CHUNK_KEYWORD_CAP]
    if keywords:
        entries.append((f"Themes: {', '.join(keywords)}", 1))
    tone = film.get("tone_summary")
    if tone:
        entries.append((f"Tone: {tone}", CHUNK_TONE_WEIGHT))
    review = film.get("review_text")
    if review:
        entries.append((f"My take: {review}", CHUNK_REVIEW_WEIGHT))
    return entries


def build_dedup_text(film):
    """cinemagent.chunking.build_chunk_text() with every entry once."""
    return "\n".join(text for text, _weight in dedup_entries(film))


def build_variant_texts(index):
    dedup = {fid: build_dedup_text(index.film_lookup[fid]) for fid in index.film_ids}
    plain = [fid for fid in index.film_ids
             if not index.film_lookup[fid].get("tone_summary") and not index.film_lookup[fid].get("review_text")]
    for fid in plain:
        assert dedup[fid] == index.film_texts[fid], f"dedup != current for no-tone/no-review film {fid}"
    # Stronger check that also covers films WITH tone/review (the requested
    # check above can be vacuous): re-expanding the dedup entries by the
    # configured weights must reproduce the current chunk text exactly, so
    # the two variants differ ONLY in repetition.
    for fid in index.film_ids:
        entries = dedup_entries(index.film_lookup[fid])
        expanded = "\n".join(text for text, weight in entries for _ in range(weight))
        assert expanded == index.film_texts[fid], f"dedup isn't current minus repetition for {fid}"
    return {"current": index.film_texts, "dedup": dedup}, len(plain)


# ---------------------------------------------------------------- scoring

def score_all(cross_encoder, query, film_ids, texts):
    pairs = [(query, texts[fid]) for fid in film_ids]
    logits = cross_encoder.predict(pairs, activation_fct=torch.nn.Identity(), show_progress_bar=False)
    return np.asarray(logits, dtype=float)


def check_logit_activation(cross_encoder, query, text):
    """Confirm the Identity-activated score is the pre-sigmoid logit of the
    default predict() output, so 'logit' in this report means what it says."""
    logit = float(cross_encoder.predict([(query, text)], activation_fct=torch.nn.Identity(), show_progress_bar=False)[0])
    default = float(cross_encoder.predict([(query, text)], show_progress_bar=False)[0])
    assert abs(sigmoid(logit) - default) < 1e-4, (logit, default)
    return default


def auc(pos, neg):
    """P(random positive outranks random negative), ties count 0.5."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def conf(logit):
    return sigmoid(logit)


# ---------------------------------------------------------------- metrics

def per_query_metrics(q, film_ids, logits, titles):
    order = np.argsort(-logits, kind="stable")
    rank_of = {film_ids[i]: r for r, i in enumerate(order, start=1)}
    top = float(logits[order[0]])
    median = float(np.median(logits))
    out = {
        "top_score": top,
        "median_score": median,
        "margin": top - median,
        "top3": [{"film": titles[film_ids[i]], "tmdb_id": film_ids[i], "logit": float(logits[i]),
                  "sigmoid": conf(float(logits[i]))} for i in order[:3]],
    }
    if q["type"] == "positive":
        rel_mask, irr_mask = label_masks(q, film_ids)
        rel_idx = np.where(rel_mask)[0]
        irr_idx = np.where(irr_mask)[0]
        lowest_rel = min(rel_idx, key=lambda i: logits[i])
        highest_irr = max(irr_idx, key=lambda i: logits[i])
        out.update({
            "relevant_ranks": sorted(
                [{"film": titles[film_ids[i]], "tmdb_id": film_ids[i], "rank": rank_of[film_ids[i]],
                  "logit": float(logits[i]), "sigmoid": conf(float(logits[i]))} for i in rel_idx],
                key=lambda r: r["rank"]),
            "borderline_ranks": sorted(
                [{"film": titles[fid], "tmdb_id": fid, "rank": rank_of[fid],
                  "logit": float(logits[film_ids.index(fid)])} for fid in q["borderline_ids"]],
                key=lambda r: r["rank"]),
            "auc": auc(logits[rel_mask], logits[irr_mask]),
            "max_relevant_score": float(logits[rel_idx].max()),
            "lowest_relevant": {"film": titles[film_ids[lowest_rel]], "logit": float(logits[lowest_rel]),
                                "sigmoid": conf(float(logits[lowest_rel]))},
            "highest_irrelevant": {"film": titles[film_ids[highest_irr]], "logit": float(logits[highest_irr]),
                                   "sigmoid": conf(float(logits[highest_irr]))},
        })
    return out


def label_masks(q, film_ids):
    """(relevant, irrelevant) boolean masks over film_ids; borderline is in neither."""
    rel = np.array([fid in q["relevant_ids"] for fid in film_ids])
    irr = np.array([fid not in q["relevant_ids"] and fid not in q["borderline_ids"] for fid in film_ids])
    return rel, irr


def sweep_point(t, pos_qs, neg_qs, scores, masks):
    neg_gated = int(sum(scores[q["id"]].max() < t for q in neg_qs))
    pos_alive = 0
    tp = fp = fn = 0
    for q in pos_qs:
        rel, irr = masks[q["id"]]
        kept = scores[q["id"]] >= t
        pos_alive += bool((kept & rel).any())
        tp += int((kept & rel).sum())
        fp += int((kept & irr).sum())
        fn += int((~kept & rel).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    neg_frac = neg_gated / len(neg_qs)
    pos_frac = pos_alive / len(pos_qs)
    return {
        "threshold_logit": float(t), "threshold_sigmoid": conf(float(t)),
        "negatives_gated": neg_gated, "negatives_gated_frac": neg_frac,
        "positives_alive": pos_alive, "positives_alive_frac": pos_frac,
        "balanced": (neg_frac + pos_frac) / 2,
        "film_precision": precision, "film_recall": recall, "film_f1": f1,
        "kept_tp": tp, "kept_fp": fp,
    }


def threshold_sweep(pos_qs, neg_qs, scores, film_ids):
    """Exact sweep: every distinct observed score is a candidate threshold
    (a gate `score >= t` only changes behaviour at observed scores), plus
    one point above the max so 'gate everything' is represented."""
    observed = np.unique(np.concatenate([scores[q["id"]] for q in pos_qs + neg_qs]))
    candidates = np.append(observed, observed[-1] + 1e-6)
    masks = {q["id"]: label_masks(q, film_ids) for q in pos_qs}
    points = [sweep_point(t, pos_qs, neg_qs, scores, masks) for t in candidates]

    all_alive = [p for p in points if p["positives_alive_frac"] == 1.0]
    best_all_alive = max(all_alive, key=lambda p: (p["negatives_gated"], p["film_f1"], p["threshold_logit"]))
    best_balanced = max(points, key=lambda p: (p["balanced"], p["film_f1"], p["threshold_logit"]))
    best_f1 = max(points, key=lambda p: (p["film_f1"], p["threshold_logit"]))

    # Full separation: some t gates every negative AND keeps every positive
    # alive <=> max over negatives of top score < min over positives of
    # their best relevant film's score.
    max_neg_top = max(float(scores[q["id"]].max()) for q in neg_qs)
    min_pos_best_rel = min(
        float(max(scores[q["id"]][i] for i, fid in enumerate(film_ids) if fid in q["relevant_ids"]))
        for q in pos_qs)
    lo, hi = observed[0], observed[-1]
    grid = [sweep_point(t, pos_qs, neg_qs, scores, masks) for t in np.linspace(lo, hi, SWEEP_TABLE_ROWS)]
    return {
        "score_range_logit": [float(lo), float(hi)],
        "n_candidate_thresholds": len(points),
        "best_all_positives_alive": best_all_alive,
        "best_balanced": best_balanced,
        "best_film_f1": best_f1,
        "full_separation": {
            "exists": max_neg_top < min_pos_best_rel,
            "max_negative_top_logit": max_neg_top,
            "min_positive_best_relevant_logit": min_pos_best_rel,
            "gap_logit": min_pos_best_rel - max_neg_top,
        },
        "grid": grid,
    }


def query_level_separation(pos_qs, neg_qs, per_query, key):
    pos = [per_query[q["id"]][key] for q in pos_qs]
    neg = [per_query[q["id"]][key] for q in neg_qs]
    return {
        "auc_positive_vs_negative": auc(pos, neg),
        "min_positive": min(pos), "max_negative": max(neg),
        "separates": min(pos) > max(neg),
        "gap": min(pos) - max(neg),
    }


def pipeline_view(index, pos_qs):
    rows = []
    for q in pos_qs:
        stages = retrieve_stages(index, q["query"])
        pool = {fid for fid, _ in stages.fused}
        top_n = {fid for fid, _ in stages.reranked}
        rel = q["relevant_ids"]
        rows.append({
            "id": q["id"], "query": q["query"], "n_relevant": len(rel), "pool_size": len(pool),
            "relevant_in_pool": len(rel & pool), "relevant_in_top_n": len(rel & top_n),
            "frac_in_pool": len(rel & pool) / len(rel), "frac_in_top_n": len(rel & top_n) / len(rel),
            "top_n_ceiling": min(RERANK_TOP_N, len(rel)) / len(rel),
            "missing_from_pool": sorted(index.film_lookup[f]["title"] for f in rel - pool),
            "any_relevant_in_top_n": bool(rel & top_n),
        })
    tot_rel = sum(r["n_relevant"] for r in rows)
    return {
        "per_query": rows,
        "micro_frac_in_pool": sum(r["relevant_in_pool"] for r in rows) / tot_rel,
        "micro_frac_in_top_n": sum(r["relevant_in_top_n"] for r in rows) / tot_rel,
        "mean_frac_in_pool": float(np.mean([r["frac_in_pool"] for r in rows])),
        "mean_frac_in_top_n": float(np.mean([r["frac_in_top_n"] for r in rows])),
        "queries_with_zero_relevant_in_pool": [r["id"] for r in rows if r["relevant_in_pool"] == 0],
        "queries_with_zero_relevant_in_top_n": [r["id"] for r in rows if not r["any_relevant_in_top_n"]],
    }


# ------------------------------------------------------ timing / memory

def peak_rss_mb():
    """Peak resident set size of this process so far, in MB. psutil isn't a
    dependency, so this uses resource.getrusage, whose ru_maxrss is a PEAK
    (bytes on macOS, KB on Linux), not the current RSS."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss / (1024 * 1024) if sys.platform == "darwin" else maxrss / 1024


def load_reranker(name):
    """Load a CrossEncoder, timing it and measuring memory. Called BEFORE
    the retrieval index loads, so the RSS figures cover only the Python
    process, its imports and this model."""
    rss_before = peak_rss_mb()
    start = time.perf_counter()
    try:
        cross_encoder = CrossEncoder(name)
    except Exception as e:  # report and stop; the caller can try other models
        sys.exit(f"ERROR: couldn't load {name!r} as a sentence-transformers CrossEncoder: {type(e).__name__}: {e}")
    load_s = time.perf_counter() - start
    rss_after = peak_rss_mb()
    device = str(next(cross_encoder.model.parameters()).device)
    mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024) if device.startswith("mps") else None
    return cross_encoder, {
        "model_load_seconds": load_s,
        "device": device,
        "max_length": cross_encoder.max_length,
        "n_parameters": sum(p.numel() for p in cross_encoder.model.parameters()),
        "memory_method": "resource.getrusage(RUSAGE_SELF).ru_maxrss (PEAK RSS; psutil not installed)",
        "peak_rss_mb_before_model_load": rss_before,
        "peak_rss_mb_after_model_load": rss_after,
        "peak_rss_mb_increase_from_model_load": rss_after - rss_before,
        "mps_allocated_mb_after_model_load": mps_mb,
    }


def time_live_rerank(index, queries):
    """Median wall time of cinemagent.retrieval.rerank() -- exactly the live
    call, default activation -- over each query's real fused pool."""
    times, sizes = [], []
    for q in queries:
        pool = [fid for fid, _ in retrieve_stages(index, q["query"]).fused]
        start = time.perf_counter()
        rerank(index.cross_encoder, q["query"], pool, index.film_texts)
        times.append(time.perf_counter() - start)
        sizes.append(len(pool))
    return {
        "live_rerank_median_seconds": float(np.median(times)),
        "live_rerank_pool_size_median": float(np.median(sizes)),
        "live_rerank_pool_size_range": [min(sizes), max(sizes)],
    }


def model_slug(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name.split("/")[-1]).strip("-")


# ----------------------------------------------------------------- output

def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_pt(p):
    return (f"t={p['threshold_logit']:+.3f} logit (sigmoid {p['threshold_sigmoid']:.3f}): "
            f"negatives gated {p['negatives_gated']}/{p['_n_neg']}, positives alive {p['positives_alive']}/{p['_n_pos']}, "
            f"film P={fmt(p['film_precision'])} R={fmt(p['film_recall'])} F1={fmt(p['film_f1'])}")


def summary_lines(result):
    meta = result["_meta"]
    perf = meta["performance"]
    mps = perf["mps_allocated_mb_after_model_load"]
    out = [f"Calibration run {meta['run_id']}  reranker {meta['reranker_model']}  ({meta['film_count']} films, "
           f"{meta['positive_query_count']} positive + {meta['negative_query_count']} negative queries)",
           f"  device {perf['device']}, {perf['n_parameters'] / 1e6:.0f}M params, max_length {perf['max_length']}; "
           f"load {perf['model_load_seconds']:.1f}s; score all {meta['film_count']} films "
           f"{perf['score_all_films_median_seconds']:.3f}s/query; live rerank "
           f"(median pool {perf['live_rerank_pool_size_median']:.0f}) {perf['live_rerank_median_seconds']:.3f}s/query",
           f"  peak RSS after model load {perf['peak_rss_mb_after_model_load']:.0f} MB "
           f"(+{perf['peak_rss_mb_increase_from_model_load']:.0f} MB during load)"
           + (f"; MPS allocated {mps:.0f} MB" if mps is not None else ""), ""]
    for v in VARIANTS:
        r = result["variants"][v]
        sw = r["sweep"]
        fs = sw["full_separation"]
        out += [f"== {v} ==",
                f"  mean AUC over positive queries: {r['mean_auc']:.3f}  (min {r['min_auc']:.3f})",
                f"  observed logit range: {sw['score_range_logit'][0]:+.3f} .. {sw['score_range_logit'][1]:+.3f}",
                f"  (i)  100% positives alive, most negatives gated: {fmt_pt(sw['best_all_positives_alive'])}",
                f"  (ii) best balanced ((neg gated + pos alive)/2 = {sw['best_balanced']['balanced']:.3f}): "
                f"{fmt_pt(sw['best_balanced'])}",
                f"  best film-level F1: {fmt_pt(sw['best_film_f1'])}",
                f"  full separation: {'YES' if fs['exists'] else 'NO'} -- max negative top {fs['max_negative_top_logit']:+.3f} vs "
                f"min positive best-relevant {fs['min_positive_best_relevant_logit']:+.3f} (gap {fs['gap_logit']:+.3f})",
                f"  query-level, top score: AUC {r['sep_top']['auc_positive_vs_negative']:.3f}, "
                f"separates={r['sep_top']['separates']} (gap {r['sep_top']['gap']:+.3f})",
                f"  query-level, margin:    AUC {r['sep_margin']['auc_positive_vs_negative']:.3f}, "
                f"separates={r['sep_margin']['separates']} (gap {r['sep_margin']['gap']:+.3f})",
                ""]
    pv = result["pipeline_view"]
    out += [f"== live pipeline (current variant, pool = RRF of top {RETRIEVAL_N_RESULTS} vector + BM25, rerank top {RERANK_TOP_N}) ==",
            f"  relevant films in fused pool: {pv['micro_frac_in_pool']:.1%} overall, mean per query {pv['mean_frac_in_pool']:.1%}",
            f"  relevant films in reranked top {RERANK_TOP_N}: {pv['micro_frac_in_top_n']:.1%} overall, "
            f"mean per query {pv['mean_frac_in_top_n']:.1%}",
            f"  queries with NO relevant film in pool: {pv['queries_with_zero_relevant_in_pool'] or 'none'}",
            f"  queries with NO relevant film in top {RERANK_TOP_N}: {pv['queries_with_zero_relevant_in_top_n'] or 'none'}",
            "",
            "Note: the live pipeline has no confidence gate; retrieve() confidence = sigmoid(logit)."]
    return out


def markdown_report(result, queries):
    meta = result["_meta"]
    by_id = {q["id"]: q for q in queries}
    md = [f"# Reranker threshold calibration — {meta['run_id']} — {meta['reranker_model']}", "",
          f"- Reranker: `{meta['reranker_model']}`, embedding: `{meta['embedding_model']}`, "
          f"sentence-transformers {meta['sentence_transformers_version']}",
          f"- {meta['film_count']} films; {meta['positive_query_count']} positive, {meta['negative_query_count']} negative queries",
          f"- Settings: RETRIEVAL_N_RESULTS={RETRIEVAL_N_RESULTS}, RRF_K={RRF_K}, RERANK_TOP_N={RERANK_TOP_N}, "
          f"CHUNK_TONE_WEIGHT={CHUNK_TONE_WEIGHT}, "
          f"CHUNK_REVIEW_WEIGHT={CHUNK_REVIEW_WEIGHT}",
          f"- All scores are true logits; *sigmoid* = sigmoid(logit), the confidence `retrieve()` returns. "
          f"The live pipeline has no confidence gate, so thresholds here are analysis only.",
          f"- dedup==current assertion held for {meta['films_without_tone_or_review']} films with no tone and no review; "
          f"dedup re-expanded by the tone/review weights == current held for all films.",
          "", "## Timing and memory", "",
          *[f"- {k}: {v}" for k, v in meta["performance"].items()],
          "", "## Terminal summary", "", "```", *summary_lines(result), "```", ""]

    for v in VARIANTS:
        r = result["variants"][v]
        md += [f"## Variant: {v}", "", "### Positive queries", "",
               "| query | AUC | relevant ranks (of %d) | lowest relevant | highest irrelevant |" % meta["film_count"],
               "|---|---|---|---|---|"]
        for qid, m in r["per_query"].items():
            if by_id[qid]["type"] != "positive":
                continue
            ranks = ", ".join(f"{x['film']} #{x['rank']}" for x in m["relevant_ranks"])
            lo, hi = m["lowest_relevant"], m["highest_irrelevant"]
            md.append(f"| `{qid}` | {m['auc']:.3f} | {ranks} | {lo['film']} {lo['logit']:+.2f} ({lo['sigmoid']:.3f}) "
                      f"| {hi['film']} {hi['logit']:+.2f} ({hi['sigmoid']:.3f}) |")
        md += ["", "### Negative queries — top 3", "", "| query | #1 | #2 | #3 |", "|---|---|---|---|"]
        for qid, m in r["per_query"].items():
            if by_id[qid]["type"] != "negative":
                continue
            cells = [f"{t['film']} {t['logit']:+.2f} ({t['sigmoid']:.3f})" for t in m["top3"]]
            md.append(f"| `{qid}` | " + " | ".join(cells) + " |")

        sw = r["sweep"]
        md += ["", "### Threshold sweep", "",
               f"Exact sweep over {sw['n_candidate_thresholds']} candidate thresholds; best points:", "",
               f"- **(i) all positives alive:** {fmt_pt(sw['best_all_positives_alive'])}",
               f"- **(ii) best balanced:** {fmt_pt(sw['best_balanced'])}",
               f"- **best film F1:** {fmt_pt(sw['best_film_f1'])}",
               f"- **Full separation: {'YES' if sw['full_separation']['exists'] else 'NO'}** — highest negative top score "
               f"{sw['full_separation']['max_negative_top_logit']:+.3f}, lowest positive best-relevant score "
               f"{sw['full_separation']['min_positive_best_relevant_logit']:+.3f} "
               f"(gap {sw['full_separation']['gap_logit']:+.3f})", "",
               "| t (logit) | sigmoid | neg gated | pos alive | film P | film R | F1 |",
               "|---|---|---|---|---|---|---|"]
        for p in sw["grid"]:
            md.append(f"| {p['threshold_logit']:+.3f} | {p['threshold_sigmoid']:.3f} "
                      f"| {p['negatives_gated']}/{meta['negative_query_count']} | {p['positives_alive']}/{meta['positive_query_count']} "
                      f"| {fmt(p['film_precision'])} | {fmt(p['film_recall'])} | {fmt(p['film_f1'])} |")

        md += ["", "### Top score vs. margin (top − median over all films)", "",
               f"- Top score: AUC(pos vs neg queries) {r['sep_top']['auc_positive_vs_negative']:.3f}, "
               f"separates={r['sep_top']['separates']}, gap {r['sep_top']['gap']:+.3f}",
               f"- Margin: AUC(pos vs neg queries) {r['sep_margin']['auc_positive_vs_negative']:.3f}, "
               f"separates={r['sep_margin']['separates']}, gap {r['sep_margin']['gap']:+.3f}", "",
               "| query | type | top | median | margin |", "|---|---|---|---|---|"]
        rows = sorted(r["per_query"].items(), key=lambda kv: -kv[1]["top_score"])
        for qid, m in rows:
            md.append(f"| `{qid}` | {by_id[qid]['type']} | {m['top_score']:+.3f} | {m['median_score']:+.3f} | {m['margin']:+.3f} |")
        md.append("")

    pv = result["pipeline_view"]
    md += ["## Live pipeline view (current variant)", "",
           f"Fused pool = RRF of top {RETRIEVAL_N_RESULTS} vector + top {RETRIEVAL_N_RESULTS} BM25; gate sees reranked top {RERANK_TOP_N}. "
           f"Films missing from the pool are a retrieval-recall problem no threshold can fix.", "",
           f"- In fused pool: {pv['micro_frac_in_pool']:.1%} of relevant films overall (mean per query {pv['mean_frac_in_pool']:.1%})",
           f"- In reranked top {RERANK_TOP_N}: {pv['micro_frac_in_top_n']:.1%} overall (mean per query {pv['mean_frac_in_top_n']:.1%})",
           f"- Queries with no relevant film in pool: {pv['queries_with_zero_relevant_in_pool'] or 'none'}",
           f"- Queries with no relevant film in top {RERANK_TOP_N}: {pv['queries_with_zero_relevant_in_top_n'] or 'none'}", "",
           f"| query | relevant | pool size | in pool | in top {RERANK_TOP_N} | top-{RERANK_TOP_N} ceiling | missing from pool |",
           "|---|---|---|---|---|---|---|"]
    for row in pv["per_query"]:
        md.append(f"| `{row['id']}` | {row['n_relevant']} | {row['pool_size']} | {row['relevant_in_pool']} ({row['frac_in_pool']:.0%}) "
                  f"| {row['relevant_in_top_n']} ({row['frac_in_top_n']:.0%}) | {row['top_n_ceiling']:.0%} "
                  f"| {', '.join(row['missing_from_pool']) or '—'} |")
    return "\n".join(md) + "\n"


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reranker", default=RERANKER_MODEL_NAME,
                        help=f"CrossEncoder model to score with (default: {RERANKER_MODEL_NAME})")
    args = parser.parse_args()

    queries = load_labels()
    print(f"Loading reranker {args.reranker}...")
    cross_encoder, perf = load_reranker(args.reranker)
    index = load_retrieval_index(verbose=True)
    # Swap in the timed model everywhere, including retrieve_stages(), and
    # drop the default one load_retrieval_index() loaded.
    index.cross_encoder = cross_encoder
    gc.collect()
    resolve_titles(queries, index.films)
    film_ids = index.film_ids
    titles = {fid: f"{f['title']} ({f.get('year', '')})" for fid, f in index.film_lookup.items()}
    variant_texts, n_plain = build_variant_texts(index)
    check_logit_activation(index.cross_encoder, queries[0]["query"], index.film_texts[film_ids[0]])

    pos_qs = [q for q in queries if q["type"] == "positive"]
    neg_qs = [q for q in queries if q["type"] == "negative"]

    variants = {}
    for v in VARIANTS:
        print(f"Scoring {len(queries)} queries x {len(film_ids)} films ({v})...")
        scores, seconds = {}, []
        for q in queries:
            start = time.perf_counter()
            scores[q["id"]] = score_all(index.cross_encoder, q["query"], film_ids, variant_texts[v])
            seconds.append(time.perf_counter() - start)
        if v == "current":
            perf["score_all_films_median_seconds"] = float(np.median(seconds))
        per_query = {q["id"]: per_query_metrics(q, film_ids, scores[q["id"]], titles) for q in queries}
        sweep = threshold_sweep(pos_qs, neg_qs, scores, film_ids)
        for key in ("best_all_positives_alive", "best_balanced", "best_film_f1"):
            sweep[key]["_n_neg"], sweep[key]["_n_pos"] = len(neg_qs), len(pos_qs)
        aucs = [per_query[q["id"]]["auc"] for q in pos_qs]
        variants[v] = {
            "mean_auc": float(np.mean(aucs)), "min_auc": float(np.min(aucs)),
            "per_query": per_query,
            "sweep": sweep,
            "sep_top": query_level_separation(pos_qs, neg_qs, per_query, "top_score"),
            "sep_margin": query_level_separation(pos_qs, neg_qs, per_query, "margin"),
            "raw_logits": {qid: dict(zip(film_ids, map(float, s))) for qid, s in scores.items()},
        }

    print("Running live pipeline (retrieve_stages) for positive queries...")
    pv = pipeline_view(index, pos_qs)
    print("Timing live rerank over each query's fused pool...")
    perf.update(time_live_rerank(index, queries))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "_meta": {
            "run_id": run_id,
            "reranker_model": args.reranker,
            "reranker_is_config_default": args.reranker == RERANKER_MODEL_NAME,
            "performance": perf,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "sentence_transformers_version": sentence_transformers.__version__,
            "settings": {"RETRIEVAL_N_RESULTS": RETRIEVAL_N_RESULTS, "RRF_K": RRF_K, "RERANK_TOP_N": RERANK_TOP_N,
                         "CHUNK_KEYWORD_CAP": CHUNK_KEYWORD_CAP, "CHUNK_TONE_WEIGHT": CHUNK_TONE_WEIGHT,
                         "CHUNK_REVIEW_WEIGHT": CHUNK_REVIEW_WEIGHT},
            "score_units": "raw cross-encoder logits (predict with activation_fct=Identity); "
                           "sigmoid(logit) is the confidence live retrieve() returns.",
            "live_pipeline_gate": None,  # no confidence gate: search_my_history returns the reranked top N
            "film_count": len(film_ids),
            "query_count": len(queries),
            "positive_query_count": len(pos_qs),
            "negative_query_count": len(neg_qs),
            "films_without_tone_or_review": n_plain,
            "labels_file": str(LABELS_PATH.relative_to(EVAL_DIR.parent)),
            "film_titles": titles,
        },
        "queries": [{"id": q["id"], "type": q["type"], "query": q["query"],
                     "relevant_ids": sorted(q["relevant_ids"]), "borderline_ids": sorted(q["borderline_ids"])}
                    for q in queries],
        "variants": variants,
        "pipeline_view": pv,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"run_{run_id}_{model_slug(args.reranker)}"
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    md_path.write_text(markdown_report(result, queries), encoding="utf-8")

    print()
    print("\n".join(summary_lines(result)))
    print(f"\nWrote {json_path.relative_to(EVAL_DIR.parent)} and {md_path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
