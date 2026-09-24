# Reranker comparison — 20260923T221241Z

Scale-free metrics only (logit scales differ between models, so raw thresholds aren't compared). Document text = `current` variant unless noted. 174 films; 15 positive / 12 negative queries.

Runs: `run_20260923T221235Z_bge-reranker-base.json`, `run_20260923T214552Z_bge-reranker-large.json`, `run_20260923T220619Z_bge-reranker-v2-m3.json`

**Baseline reproduction:** `run_20260923T221235Z_bge-reranker-base.json` vs `run_20260923T032909Z.json` — all scores, metrics and pipeline results IDENTICAL; _meta differs only in ['performance', 'reranker_is_config_default', 'run_id']. All 2 base runs identical to baseline: True.

| metric | bge-reranker-base | bge-reranker-large | bge-reranker-v2-m3 |
|---|---|---|---|
| **Ranking** |  |  |  |
| mean AUC, positives (min) | 0.849 (0.607) | 0.846 (0.521) | 0.781 (0.485) |
| mean AUC — dedup text (min) | 0.792 (0.433) | 0.785 (0.528) | 0.754 (0.444) |
| &nbsp;&nbsp;AUC `pos_uneasy` | 0.885 | 0.912 | 0.944 |
| &nbsp;&nbsp;AUC `pos_found_family` | 0.781 | 0.840 | 0.606 |
| &nbsp;&nbsp;AUC `pos_pleasant_surprise` | 0.882 | 0.904 | 0.707 |
| &nbsp;&nbsp;AUC `pos_plot_twist` | 0.607 | 0.822 | 0.485 |
| &nbsp;&nbsp;AUC `pos_unhinged_comedy` | 0.816 | 0.521 | 0.595 |
| &nbsp;&nbsp;AUC `pos_visually_stunning` | 0.732 | 0.741 | 0.759 |
| &nbsp;&nbsp;AUC `pos_war` | 0.955 | 0.951 | 0.867 |
| &nbsp;&nbsp;AUC `pos_hopeful` | 0.750 | 0.806 | 0.820 |
| &nbsp;&nbsp;AUC `pos_bittersweet_romance` | 0.999 | 1.000 | 0.983 |
| &nbsp;&nbsp;AUC `pos_grief` | 0.985 | 0.967 | 0.888 |
| &nbsp;&nbsp;AUC `pos_coming_of_age` | 0.950 | 0.947 | 0.874 |
| &nbsp;&nbsp;AUC `pos_class_satire` | 0.979 | 0.984 | 0.777 |
| &nbsp;&nbsp;AUC `pos_feel_good` | 0.822 | 0.715 | 0.739 |
| &nbsp;&nbsp;AUC `pos_mind_bending` | 0.811 | 0.845 | 0.729 |
| &nbsp;&nbsp;AUC `pos_obsessive_ambition` | 0.781 | 0.730 | 0.941 |
| **Gating** |  |  |  |
| full separation (gap, logits) | NO (-6.202) | NO (-7.093) | NO (-4.670) |
| full-separation gap ÷ model's logit range | -0.533 | -0.658 | -0.371 |
| best balanced: neg gated / pos kept | 10/12 / 7/15 | 7/12 / 8/15 | 10/12 / 9/15 |
| (i) all 15 positives kept: neg gated | 1/12 | 0/12 | 0/12 |
| query-level AUC: top score | 0.794 | 0.606 | 0.806 |
| query-level AUC: margin | 0.772 | 0.794 | 0.894 |
| **Live pipeline** |  |  |  |
| % relevant in fused pool | 75.7% | 75.7% | 75.7% |
| % relevant in reranked top 8 | 52.9% | 50.0% | 45.7% |
| queries with none in top 8 | `pos_unhinged_comedy` | `pos_unhinged_comedy` | `pos_unhinged_comedy` |
| **Cost** |  |  |  |
| parameters | 278M | 560M | 568M |
| device | mps:0 | mps:0 | mps:0 |
| model load (s) | 2.1 | 4.2 | 2.9 |
| score 1 query × 174 films, median (s) | 5.975 | 20.235 | 19.979 |
| live rerank 1 fused pool, median (s) | 0.952 | 3.219 | 3.197 |
| fused pool size, median [range] | 28 [20, 36] | 28 [20, 36] | 28 [20, 36] |
| peak RSS after model load (MB) | 893 | 896 | 854 |
| peak RSS increase during load (MB) | 443 | 446 | 403 |
| MPS allocated after load (MB) | 1061 | 2136 | 2166 |

- Fused-pool membership identical across models (per-query pool sizes and pool %): **True**.
- Memory: resource.getrusage(RUSAGE_SELF).ru_maxrss (PEAK RSS; psutil not installed); measured after imports + reranker load, before the retrieval index loads. On MPS the weights live in Metal buffers, reported separately as *MPS allocated*.
- Timings are wall-clock with the other runs finished (sequential, same machine). Load time uses a warm Hugging Face cache (models were downloaded beforehand).
