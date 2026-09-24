# Reranker threshold calibration — 20260923T214552Z — BAAI/bge-reranker-large

- Reranker: `BAAI/bge-reranker-large`, embedding: `BAAI/bge-base-en-v1.5`, sentence-transformers 3.4.1
- 174 films; 15 positive, 12 negative queries
- Settings: RETRIEVAL_N_RESULTS=20, RRF_K=60, RERANK_TOP_N=8, RERANK_CONFIDENCE_THRESHOLD=0.5, CHUNK_TONE_WEIGHT=2, CHUNK_REVIEW_WEIGHT=3
- All scores are true logits; *sigmoid* = sigmoid(logit); *live-gate* = sigmoid(sigmoid(logit)), what `retrieve()` currently compares to the threshold (CrossEncoder.predict already applies sigmoid).
- dedup==current assertion held for 0 films with no tone and no review; dedup re-expanded by the tone/review weights == current held for all films.

## Timing and memory

- model_load_seconds: 4.237280209000346
- device: mps:0
- max_length: None
- n_parameters: 559891457
- memory_method: resource.getrusage(RUSAGE_SELF).ru_maxrss (PEAK RSS; psutil not installed)
- peak_rss_mb_before_model_load: 449.84375
- peak_rss_mb_after_model_load: 895.765625
- peak_rss_mb_increase_from_model_load: 445.921875
- mps_allocated_mb_after_model_load: 2135.824951171875
- score_all_films_median_seconds: 20.235354291000476
- live_rerank_median_seconds: 3.2186463750003895
- live_rerank_pool_size_median: 28.0
- live_rerank_pool_size_range: [20, 36]

## Terminal summary

```
Calibration run 20260923T214552Z  reranker BAAI/bge-reranker-large  (174 films, 15 positive + 12 negative queries)
  device mps:0, 560M params, max_length None; load 4.2s; score all 174 films 20.235s/query; live rerank (median pool 28) 3.219s/query
  peak RSS after model load 896 MB (+446 MB during load); MPS allocated 2136 MB

== current ==
  mean AUC over positive queries: 0.846  (min 0.521)
  observed logit range: -9.481 .. +1.306
  (i)  100% positives alive, most negatives gated: t=-7.409 logit (sigmoid 0.001; live-gate 0.5002): negatives gated 0/12, positives alive 15/15, film P=0.040 R=0.900 F1=0.076
  (ii) best balanced ((neg gated + pos alive)/2 = 0.558): t=-3.169 logit (sigmoid 0.040; live-gate 0.5101): negatives gated 7/12, positives alive 8/15, film P=0.439 R=0.257 F1=0.324
  best film-level F1: t=-3.550 logit (sigmoid 0.028; live-gate 0.5070): negatives gated 4/12, positives alive 11/15, film P=0.431 R=0.314 F1=0.364
  full separation: NO -- max negative top -0.316 vs min positive best-relevant -7.409 (gap -7.093)
  query-level, top score: AUC 0.606, separates=False (gap -5.966)
  query-level, margin:    AUC 0.794, separates=False (gap -1.579)

== dedup ==
  mean AUC over positive queries: 0.785  (min 0.528)
  observed logit range: -9.482 .. +1.708
  (i)  100% positives alive, most negatives gated: t=-6.404 logit (sigmoid 0.002; live-gate 0.5004): negatives gated 0/12, positives alive 15/15, film P=0.048 R=0.786 F1=0.090
  (ii) best balanced ((neg gated + pos alive)/2 = 0.500): t=-6.404 logit (sigmoid 0.002; live-gate 0.5004): negatives gated 0/12, positives alive 15/15, film P=0.048 R=0.786 F1=0.090
  best film-level F1: t=-3.604 logit (sigmoid 0.026; live-gate 0.5066): negatives gated 2/12, positives alive 11/15, film P=0.241 R=0.286 F1=0.261
  full separation: NO -- max negative top +1.708 vs min positive best-relevant -6.404 (gap -8.112)
  query-level, top score: AUC 0.372, separates=False (gap -7.341)
  query-level, margin:    AUC 0.422, separates=False (gap -3.206)

== live pipeline (current variant, pool = RRF of top 20 vector + BM25, rerank top 8) ==
  relevant films in fused pool: 75.7% overall, mean per query 79.6%
  relevant films in reranked top 8: 50.0% overall, mean per query 51.6%
  queries with NO relevant film in pool: none
  queries with NO relevant film in top 8: ['pos_unhinged_comedy']

Note: live retrieve() confidence = sigmoid(sigmoid(logit)) in [0.5, 0.731] (CrossEncoder.predict already applies sigmoid), so threshold 0.5 can never gate anything today.
```

## Variant: current

### Positive queries

| query | AUC | relevant ranks (of 174) | lowest relevant | highest irrelevant |
|---|---|---|---|---|
| `pos_uneasy` | 0.912 | Obsession (2025) #1, Companion (2025) #2, Longlegs (2024) #5, Weapons (2025) #9, Joker (2019) #30, Hereditary (2018) #73 | Hereditary (2018) -8.05 (0.000) | Nope (2022) -2.70 (0.063) |
| `pos_found_family` | 0.840 | Guardians of the Galaxy Vol. 2 (2017) #1, Coco (2017) #15, KPop Demon Hunters (2025) #33, Guardians of the Galaxy (2014) #35, Deadpool 2 (2018) #69 | Deadpool 2 (2018) -7.66 (0.000) | The Fantastic 4: First Steps (2025) -4.29 (0.014) |
| `pos_pleasant_surprise` | 0.904 | Flight (2012) #7, Roofman (2025) #8, Crazy, Stupid, Love. (2011) #13, Scott Pilgrim vs. the World (2010) #20, Bugonia (2025) #27, The Drama (2026) #48 | The Drama (2026) -7.55 (0.001) | Superman (2025) -3.05 (0.045) |
| `pos_plot_twist` | 0.822 | Avengers: Infinity War (2018) #1, Bugonia (2025) #2, The Hateful Eight (2015) #27, Parasite (2019) #100 | Parasite (2019) -6.45 (0.002) | Chef (2014) -4.69 (0.009) |
| `pos_unhinged_comedy` | 0.521 | 21 Jump Street (2012) #34, Ted 2 (2015) #84, Deadpool (2016) #99, Ted (2012) #103, Pineapple Express (2008) #104 | Pineapple Express (2008) -6.27 (0.002) | Bugonia (2025) -0.90 (0.290) |
| `pos_visually_stunning` | 0.741 | Doctor Strange (2016) #1, Past Lives (2023) #5, Shang-Chi and the Legend of the Ten Rings (2021) #28, Uncut Gems (2019) #30, Project Hail Mary (2026) #50, The Odyssey (2026) #78, Interstellar (2014) #143 | Interstellar (2014) -9.04 (0.000) | Black Panther (2018) -4.32 (0.013) |
| `pos_war` | 0.951 | War for the Planet of the Apes (2017) #4, The Greatest Beer Run Ever (2022) #8, The Odyssey (2026) #19 | The Odyssey (2026) -8.35 (0.000) | Roofman (2025) -6.28 (0.002) |
| `pos_hopeful` | 0.806 | Chef (2014) #5, Coco (2017) #6, Soul (2020) #16, Everything Everywhere All at Once (2022) #115 | Everything Everywhere All at Once (2022) -6.31 (0.002) | Big Hero 6 (2014) -1.85 (0.136) |
| `pos_bittersweet_romance` | 1.000 | (500) Days of Summer (2009) #1, La La Land (2016) #2, Crazy, Stupid, Love. (2011) #3, Materialists (2025) #4, Past Lives (2023) #5 | Past Lives (2023) -3.08 (0.044) | Guardians of the Galaxy Vol. 3 (2023) -3.36 (0.034) |
| `pos_grief` | 0.967 | Midsommar (2019) #2, Life of Pi (2012) #4, Hereditary (2018) #5, Flight (2012) #21 | Flight (2012) -5.13 (0.006) | Arrival (2016) -2.54 (0.073) |
| `pos_coming_of_age` | 0.947 | CODA (2021) #1, A Silent Voice: The Movie (2016) #3, Cha Cha Real Smooth (2022) #13, No Hard Feelings (2023) #29 | No Hard Feelings (2023) -5.59 (0.004) | GOAT (2026) -1.39 (0.200) |
| `pos_class_satire` | 0.984 | Parasite (2019) #1, American Psycho (2000) #2, Bugonia (2025) #5, Anora (2024) #13 | Anora (2024) -5.20 (0.005) | Chef (2014) -4.44 (0.012) |
| `pos_feel_good` | 0.715 | I Love You, Man (2009) #2, How High (2001) #4, Despicable Me 4 (2024) #6, Mike and Dave Need Wedding Dates (2016) #71, Soul Plane (2004) #89, Wedding Crashers (2005) #139 | Wedding Crashers (2005) -7.05 (0.001) | No Hard Feelings (2023) -2.77 (0.059) |
| `pos_mind_bending` | 0.845 | Arrival (2016) #4, Project Hail Mary (2026) #15, Nope (2022) #24, Interstellar (2014) #75 | Interstellar (2014) -5.04 (0.006) | Doctor Strange in the Multiverse of Madness (2022) -2.03 (0.116) |
| `pos_obsessive_ambition` | 0.730 | Whiplash (2014) #14, King Richard (2021) #59, tick, tick... BOOM! (2021) #72 | tick, tick... BOOM! (2021) -7.24 (0.001) | Ocean's Eight (2018) -4.22 (0.014) |

### Negative queries — top 3

| query | #1 | #2 | #3 |
|---|---|---|---|
| `neg_werewolves` | Coco (2017) -3.66 (0.025) | Ocean's Eight (2018) -4.10 (0.016) | Soul (2020) -4.14 (0.016) |
| `neg_silent_black_and_white` | Us (2019) -3.45 (0.031) | Crazy, Stupid, Love. (2011) -3.47 (0.030) | Rise of the Planet of the Apes (2011) -3.58 (0.027) |
| `neg_bollywood` | Ocean's Eight (2018) -2.30 (0.091) | Cha Cha Real Smooth (2022) -3.26 (0.037) | Coco (2017) -3.72 (0.024) |
| `neg_documentary` | Ocean's Eight (2018) -0.32 (0.422) | Flight (2012) -0.66 (0.341) | War for the Planet of the Apes (2017) -0.68 (0.336) |
| `neg_pirates` | Coco (2017) -0.43 (0.394) | Soul (2020) -0.90 (0.290) | Ocean's Eight (2018) -1.47 (0.187) |
| `neg_time_loop` | Ocean's Eight (2018) -1.33 (0.209) | Black Panther (2018) -1.50 (0.182) | Kingsman: The Golden Circle (2017) -1.72 (0.152) |
| `neg_horse_racing` | F1 (2025) -4.01 (0.018) | Coco (2017) -4.51 (0.011) | Black Panther (2018) -4.71 (0.009) |
| `neg_cold_war_espionage` | Arrival (2016) -3.52 (0.029) | Ocean's Eight (2018) -3.63 (0.026) | Past Lives (2023) -4.12 (0.016) |
| `neg_mountaineering` | Black Panther (2018) -1.51 (0.181) | Ocean's Eight (2018) -2.51 (0.075) | Big Hero 6 (2014) -2.52 (0.075) |
| `neg_chess` | Crazy, Stupid, Love. (2011) -3.28 (0.036) | Ocean's Eight (2018) -3.72 (0.024) | Black Panther (2018) -4.02 (0.018) |
| `neg_vineyard` | Black Panther (2018) -4.72 (0.009) | War for the Planet of the Apes (2017) -4.93 (0.007) | Kingsman: The Golden Circle (2017) -5.03 (0.007) |
| `neg_ballet` | Coco (2017) -4.06 (0.017) | Ocean's Eight (2018) -4.14 (0.016) | Black Panther (2018) -4.43 (0.012) |

### Threshold sweep

Exact sweep over 4695 candidate thresholds; best points:

- **(i) all positives alive:** t=-7.409 logit (sigmoid 0.001; live-gate 0.5002): negatives gated 0/12, positives alive 15/15, film P=0.040 R=0.900 F1=0.076
- **(ii) best balanced:** t=-3.169 logit (sigmoid 0.040; live-gate 0.5101): negatives gated 7/12, positives alive 8/15, film P=0.439 R=0.257 F1=0.324
- **best film F1:** t=-3.550 logit (sigmoid 0.028; live-gate 0.5070): negatives gated 4/12, positives alive 11/15, film P=0.431 R=0.314 F1=0.364
- **Full separation: NO** — highest negative top score -0.316, lowest positive best-relevant score -7.409 (gap -7.093)

| t (logit) | sigmoid | live-gate | neg gated | pos alive | film P | film R | F1 |
|---|---|---|---|---|---|---|---|
| -9.481 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.032 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.030 | 0.986 | 0.058 |
| -8.582 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.032 | 0.986 | 0.062 |
| -8.133 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.035 | 0.971 | 0.067 |
| -7.684 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.037 | 0.929 | 0.072 |
| -7.234 | 0.001 | 0.5002 | 0/12 | 14/15 | 0.040 | 0.857 | 0.077 |
| -6.785 | 0.001 | 0.5003 | 0/12 | 14/15 | 0.044 | 0.757 | 0.084 |
| -6.335 | 0.002 | 0.5004 | 0/12 | 14/15 | 0.053 | 0.686 | 0.099 |
| -5.886 | 0.003 | 0.5007 | 0/12 | 13/15 | 0.065 | 0.557 | 0.116 |
| -5.436 | 0.004 | 0.5011 | 0/12 | 12/15 | 0.093 | 0.514 | 0.158 |
| -4.987 | 0.007 | 0.5017 | 0/12 | 11/15 | 0.132 | 0.443 | 0.204 |
| -4.537 | 0.011 | 0.5026 | 1/12 | 11/15 | 0.206 | 0.386 | 0.269 |
| -4.088 | 0.016 | 0.5041 | 1/12 | 11/15 | 0.321 | 0.371 | 0.344 |
| -3.638 | 0.026 | 0.5064 | 4/12 | 11/15 | 0.415 | 0.314 | 0.358 |
| -3.189 | 0.040 | 0.5099 | 7/12 | 8/15 | 0.439 | 0.257 | 0.324 |
| -2.739 | 0.061 | 0.5152 | 7/12 | 6/15 | 0.407 | 0.157 | 0.227 |
| -2.290 | 0.092 | 0.5230 | 8/12 | 4/15 | 0.412 | 0.100 | 0.161 |
| -1.840 | 0.137 | 0.5342 | 8/12 | 4/15 | 0.455 | 0.071 | 0.123 |
| -1.391 | 0.199 | 0.5496 | 9/12 | 3/15 | 0.375 | 0.043 | 0.077 |
| -0.942 | 0.281 | 0.5697 | 10/12 | 2/15 | 0.667 | 0.029 | 0.055 |
| -0.492 | 0.379 | 0.5937 | 10/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| -0.043 | 0.489 | 0.6200 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +0.407 | 0.600 | 0.6457 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +0.856 | 0.702 | 0.6686 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +1.306 | 0.787 | 0.6871 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |

### Top score vs. margin (top − median over all films)

- Top score: AUC(pos vs neg queries) 0.606, separates=False, gap -5.966
- Margin: AUC(pos vs neg queries) 0.794, separates=False, gap -1.579

| query | type | top | median | margin |
|---|---|---|---|---|
| `pos_uneasy` | positive | +1.306 | -8.216 | +9.522 |
| `neg_documentary` | negative | -0.316 | -3.222 | +2.906 |
| `neg_pirates` | negative | -0.430 | -4.797 | +4.367 |
| `pos_unhinged_comedy` | positive | -0.896 | -6.063 | +5.168 |
| `pos_plot_twist` | positive | -0.902 | -6.279 | +5.377 |
| `pos_coming_of_age` | positive | -0.990 | -6.732 | +5.743 |
| `neg_time_loop` | negative | -1.329 | -5.085 | +3.756 |
| `pos_bittersweet_romance` | positive | -1.414 | -6.440 | +5.026 |
| `neg_mountaineering` | negative | -1.511 | -5.761 | +4.250 |
| `pos_hopeful` | positive | -1.848 | -5.950 | +4.101 |
| `pos_pleasant_surprise` | positive | -1.996 | -8.137 | +6.141 |
| `pos_mind_bending` | positive | -2.027 | -5.215 | +3.188 |
| `neg_bollywood` | negative | -2.302 | -6.897 | +4.594 |
| `pos_grief` | positive | -2.539 | -6.564 | +4.026 |
| `pos_found_family` | positive | -2.699 | -7.998 | +5.299 |
| `pos_feel_good` | positive | -2.766 | -6.213 | +3.447 |
| `pos_class_satire` | positive | -3.060 | -7.145 | +4.085 |
| `neg_chess` | negative | -3.284 | -6.881 | +3.596 |
| `neg_silent_black_and_white` | negative | -3.450 | -5.526 | +2.076 |
| `pos_visually_stunning` | positive | -3.474 | -8.164 | +4.690 |
| `neg_cold_war_espionage` | negative | -3.517 | -6.456 | +2.938 |
| `neg_werewolves` | negative | -3.663 | -6.672 | +3.009 |
| `neg_horse_racing` | negative | -4.007 | -7.211 | +3.204 |
| `neg_ballet` | negative | -4.063 | -7.378 | +3.316 |
| `pos_obsessive_ambition` | positive | -4.221 | -7.545 | +3.324 |
| `neg_vineyard` | negative | -4.722 | -8.173 | +3.450 |
| `pos_war` | positive | -6.282 | -9.298 | +3.015 |

## Variant: dedup

### Positive queries

| query | AUC | relevant ranks (of 174) | lowest relevant | highest irrelevant |
|---|---|---|---|---|
| `pos_uneasy` | 0.915 | Obsession (2025) #1, Companion (2025) #3, Longlegs (2024) #8, Weapons (2025) #9, Hereditary (2018) #26, Joker (2019) #72 | Joker (2019) -7.77 (0.000) | Nope (2022) -2.69 (0.063) |
| `pos_found_family` | 0.619 | Guardians of the Galaxy Vol. 2 (2017) #1, Coco (2017) #42, Guardians of the Galaxy (2014) #87, KPop Demon Hunters (2025) #99, Deadpool 2 (2018) #112 | Deadpool 2 (2018) -8.28 (0.000) | War for the Planet of the Apes (2017) -3.83 (0.021) |
| `pos_pleasant_surprise` | 0.817 | Roofman (2025) #13, Scott Pilgrim vs. the World (2010) #16, Crazy, Stupid, Love. (2011) #17, Flight (2012) #19, Bugonia (2025) #63, The Drama (2026) #82 | The Drama (2026) -7.95 (0.000) | Superman (2025) -2.86 (0.054) |
| `pos_plot_twist` | 0.720 | Avengers: Infinity War (2018) #1, Bugonia (2025) #19, The Hateful Eight (2015) #43, Parasite (2019) #137 | Parasite (2019) -6.98 (0.001) | War for the Planet of the Apes (2017) -3.98 (0.018) |
| `pos_unhinged_comedy` | 0.528 | 21 Jump Street (2012) #28, Deadpool (2016) #75, Ted 2 (2015) #98, Pineapple Express (2008) #103, Ted (2012) #114 | Ted (2012) -6.22 (0.002) | Bugonia (2025) -0.91 (0.286) |
| `pos_visually_stunning` | 0.826 | Doctor Strange (2016) #1, Past Lives (2023) #2, Project Hail Mary (2026) #17, Shang-Chi and the Legend of the Ten Rings (2021) #19, The Odyssey (2026) #26, Uncut Gems (2019) #42, Interstellar (2014) #128 | Interstellar (2014) -8.70 (0.000) | Coco (2017) -4.23 (0.014) |
| `pos_war` | 0.875 | War for the Planet of the Apes (2017) #3, The Odyssey (2026) #29, The Greatest Beer Run Ever (2022) #38 | The Greatest Beer Run Ever (2022) -8.17 (0.000) | Roofman (2025) -5.63 (0.004) |
| `pos_hopeful` | 0.750 | Chef (2014) #5, Coco (2017) #11, Soul (2020) #27, Everything Everywhere All at Once (2022) #137 | Everything Everywhere All at Once (2022) -6.60 (0.001) | The Upside (2017) -2.20 (0.100) |
| `pos_bittersweet_romance` | 1.000 | (500) Days of Summer (2009) #1, Crazy, Stupid, Love. (2011) #2, La La Land (2016) #3, Materialists (2025) #4, Past Lives (2023) #5 | Past Lives (2023) -2.83 (0.056) | Jujutsu Kaisen 0 (2021) -3.53 (0.028) |
| `pos_grief` | 0.959 | Midsommar (2019) #2, Life of Pi (2012) #4, Hereditary (2018) #6, Flight (2012) #26 | Flight (2012) -4.67 (0.009) | Arrival (2016) -2.48 (0.077) |
| `pos_coming_of_age` | 0.874 | CODA (2021) #1, A Silent Voice: The Movie (2016) #4, Cha Cha Real Smooth (2022) #37, No Hard Feelings (2023) #54 | No Hard Feelings (2023) -5.60 (0.004) | GOAT (2026) -1.94 (0.126) |
| `pos_class_satire` | 0.922 | Parasite (2019) #1, American Psycho (2000) #6, Bugonia (2025) #26, Anora (2024) #29 | Anora (2024) -5.21 (0.005) | Rise of the Planet of the Apes (2011) -3.35 (0.034) |
| `pos_feel_good` | 0.669 | I Love You, Man (2009) #1, How High (2001) #8, Despicable Me 4 (2024) #10, Mike and Dave Need Wedding Dates (2016) #85, Soul Plane (2004) #117, Wedding Crashers (2005) #136 | Wedding Crashers (2005) -6.77 (0.001) | The Switch (2010) -3.26 (0.037) |
| `pos_mind_bending` | 0.744 | Project Hail Mary (2026) #15, Arrival (2016) #29, Nope (2022) #71, Interstellar (2014) #72 | Interstellar (2014) -4.82 (0.008) | Rise of the Planet of the Apes (2011) -1.78 (0.144) |
| `pos_obsessive_ambition` | 0.554 | Whiplash (2014) #52, tick, tick... BOOM! (2021) #90, King Richard (2021) #92 | King Richard (2021) -7.43 (0.001) | Ford v Ferrari (2019) -2.51 (0.075) |

### Negative queries — top 3

| query | #1 | #2 | #3 |
|---|---|---|---|
| `neg_werewolves` | Ford v Ferrari (2019) -2.00 (0.119) | The Dark Knight (2008) -2.14 (0.105) | Green Book (2018) -2.44 (0.080) |
| `neg_silent_black_and_white` | Rise of the Planet of the Apes (2011) -1.39 (0.200) | Ford v Ferrari (2019) -2.08 (0.111) | War for the Planet of the Apes (2017) -2.37 (0.086) |
| `neg_bollywood` | Green Book (2018) -0.73 (0.324) | Cha Cha Real Smooth (2022) -2.19 (0.101) | Coco (2017) -2.63 (0.067) |
| `neg_documentary` | War for the Planet of the Apes (2017) +1.71 (0.847) | The Dark Knight (2008) +1.24 (0.775) | Rise of the Planet of the Apes (2011) +0.83 (0.696) |
| `neg_pirates` | Green Book (2018) +1.32 (0.790) | The Dark Knight (2008) +1.01 (0.734) | Ford v Ferrari (2019) +1.00 (0.731) |
| `neg_time_loop` | The Dark Knight (2008) +0.41 (0.602) | The Dark Knight Rises (2012) +0.38 (0.593) | Captain America: Civil War (2016) +0.25 (0.563) |
| `neg_horse_racing` | The Avengers (2012) -3.41 (0.032) | The Odyssey (2026) -3.53 (0.028) | War for the Planet of the Apes (2017) -3.53 (0.028) |
| `neg_cold_war_espionage` | The Dark Knight (2008) -3.46 (0.030) | Cha Cha Real Smooth (2022) -3.49 (0.030) | The Avengers (2012) -3.51 (0.029) |
| `neg_mountaineering` | Captain America: Civil War (2016) +0.32 (0.580) | The Avengers (2012) -0.02 (0.495) | Avengers: Age of Ultron (2015) -0.33 (0.418) |
| `neg_chess` | Crazy, Stupid, Love. (2011) -1.60 (0.168) | Uncut Gems (2019) -2.90 (0.052) | The Dark Knight (2008) -2.99 (0.048) |
| `neg_vineyard` | War for the Planet of the Apes (2017) -3.78 (0.022) | Rise of the Planet of the Apes (2011) -4.03 (0.018) | The Dark Knight (2008) -4.03 (0.017) |
| `neg_ballet` | Deadpool (2016) -3.63 (0.026) | Coco (2017) -3.87 (0.020) | Ford v Ferrari (2019) -3.97 (0.018) |

### Threshold sweep

Exact sweep over 4698 candidate thresholds; best points:

- **(i) all positives alive:** t=-6.404 logit (sigmoid 0.002; live-gate 0.5004): negatives gated 0/12, positives alive 15/15, film P=0.048 R=0.786 F1=0.090
- **(ii) best balanced:** t=-6.404 logit (sigmoid 0.002; live-gate 0.5004): negatives gated 0/12, positives alive 15/15, film P=0.048 R=0.786 F1=0.090
- **best film F1:** t=-3.604 logit (sigmoid 0.026; live-gate 0.5066): negatives gated 2/12, positives alive 11/15, film P=0.241 R=0.286 F1=0.261
- **Full separation: NO** — highest negative top score +1.708, lowest positive best-relevant score -6.404 (gap -8.112)

| t (logit) | sigmoid | live-gate | neg gated | pos alive | film P | film R | F1 |
|---|---|---|---|---|---|---|---|
| -9.482 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.016 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.030 | 1.000 | 0.059 |
| -8.549 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.032 | 0.986 | 0.062 |
| -8.083 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.034 | 0.957 | 0.065 |
| -7.617 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.034 | 0.886 | 0.066 |
| -7.151 | 0.001 | 0.5002 | 0/12 | 15/15 | 0.038 | 0.843 | 0.072 |
| -6.684 | 0.001 | 0.5003 | 0/12 | 15/15 | 0.043 | 0.814 | 0.082 |
| -6.218 | 0.002 | 0.5005 | 0/12 | 14/15 | 0.050 | 0.757 | 0.094 |
| -5.752 | 0.003 | 0.5008 | 0/12 | 13/15 | 0.055 | 0.629 | 0.101 |
| -5.286 | 0.005 | 0.5013 | 0/12 | 12/15 | 0.066 | 0.543 | 0.118 |
| -4.819 | 0.008 | 0.5020 | 0/12 | 12/15 | 0.085 | 0.457 | 0.144 |
| -4.353 | 0.013 | 0.5032 | 0/12 | 11/15 | 0.123 | 0.386 | 0.186 |
| -3.887 | 0.020 | 0.5050 | 0/12 | 11/15 | 0.197 | 0.329 | 0.246 |
| -3.421 | 0.032 | 0.5079 | 3/12 | 9/15 | 0.258 | 0.229 | 0.242 |
| -2.954 | 0.050 | 0.5124 | 4/12 | 5/15 | 0.297 | 0.157 | 0.206 |
| -2.488 | 0.077 | 0.5192 | 4/12 | 4/15 | 0.348 | 0.114 | 0.172 |
| -2.022 | 0.117 | 0.5292 | 4/12 | 4/15 | 0.500 | 0.086 | 0.146 |
| -1.556 | 0.174 | 0.5435 | 6/12 | 2/15 | 0.500 | 0.029 | 0.054 |
| -1.089 | 0.252 | 0.5626 | 7/12 | 1/15 | 0.500 | 0.014 | 0.028 |
| -0.623 | 0.349 | 0.5864 | 8/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| -0.157 | 0.461 | 0.6132 | 8/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +0.309 | 0.577 | 0.6403 | 8/12 | 0/15 | — | 0.000 | 0.000 |
| +0.776 | 0.685 | 0.6648 | 10/12 | 0/15 | — | 0.000 | 0.000 |
| +1.242 | 0.776 | 0.6848 | 10/12 | 0/15 | — | 0.000 | 0.000 |
| +1.708 | 0.847 | 0.6999 | 11/12 | 0/15 | — | 0.000 | 0.000 |

### Top score vs. margin (top − median over all films)

- Top score: AUC(pos vs neg queries) 0.372, separates=False, gap -7.341
- Margin: AUC(pos vs neg queries) 0.422, separates=False, gap -3.206

| query | type | top | median | margin |
|---|---|---|---|---|
| `neg_documentary` | negative | +1.708 | -3.130 | +4.838 |
| `neg_pirates` | negative | +1.324 | -4.265 | +5.588 |
| `neg_time_loop` | negative | +0.414 | -4.818 | +5.233 |
| `neg_mountaineering` | negative | +0.324 | -5.091 | +5.414 |
| `pos_uneasy` | positive | +0.004 | -8.059 | +8.063 |
| `neg_bollywood` | negative | -0.734 | -6.719 | +5.984 |
| `pos_unhinged_comedy` | positive | -0.913 | -5.842 | +4.930 |
| `neg_silent_black_and_white` | negative | -1.389 | -5.390 | +4.001 |
| `pos_bittersweet_romance` | positive | -1.520 | -6.120 | +4.600 |
| `pos_coming_of_age` | positive | -1.576 | -6.524 | +4.948 |
| `neg_chess` | negative | -1.602 | -6.612 | +5.010 |
| `pos_plot_twist` | positive | -1.619 | -6.189 | +4.570 |
| `pos_mind_bending` | positive | -1.783 | -5.118 | +3.336 |
| `neg_werewolves` | negative | -2.004 | -6.452 | +4.447 |
| `pos_hopeful` | positive | -2.202 | -5.737 | +3.536 |
| `pos_grief` | positive | -2.478 | -6.513 | +4.035 |
| `pos_obsessive_ambition` | positive | -2.506 | -7.201 | +4.695 |
| `pos_found_family` | positive | -2.610 | -7.649 | +5.039 |
| `pos_pleasant_surprise` | positive | -2.855 | -8.031 | +5.176 |
| `pos_feel_good` | positive | -3.199 | -5.978 | +2.778 |
| `pos_class_satire` | positive | -3.209 | -6.907 | +3.699 |
| `neg_horse_racing` | negative | -3.409 | -6.950 | +3.541 |
| `neg_cold_war_espionage` | negative | -3.464 | -6.425 | +2.960 |
| `pos_visually_stunning` | positive | -3.593 | -7.443 | +3.851 |
| `neg_ballet` | negative | -3.630 | -7.015 | +3.385 |
| `neg_vineyard` | negative | -3.776 | -7.986 | +4.210 |
| `pos_war` | positive | -5.633 | -9.252 | +3.618 |

## Live pipeline view (current variant)

Fused pool = RRF of top 20 vector + top 20 BM25; gate sees reranked top 8. Films missing from the pool are a retrieval-recall problem no threshold can fix.

- In fused pool: 75.7% of relevant films overall (mean per query 79.6%)
- In reranked top 8: 50.0% overall (mean per query 51.6%)
- Queries with no relevant film in pool: none
- Queries with no relevant film in top 8: ['pos_unhinged_comedy']

| query | relevant | pool size | in pool | in top 8 | top-8 ceiling | missing from pool |
|---|---|---|---|---|---|---|
| `pos_uneasy` | 6 | 27 | 5 (83%) | 3 (50%) | 100% | Joker |
| `pos_found_family` | 5 | 31 | 3 (60%) | 2 (40%) | 100% | Guardians of the Galaxy, KPop Demon Hunters |
| `pos_pleasant_surprise` | 6 | 36 | 2 (33%) | 2 (33%) | 100% | Bugonia, Flight, Scott Pilgrim vs. the World, The Drama |
| `pos_plot_twist` | 4 | 27 | 3 (75%) | 2 (50%) | 100% | Parasite |
| `pos_unhinged_comedy` | 5 | 34 | 3 (60%) | 0 (0%) | 100% | Deadpool, Ted |
| `pos_visually_stunning` | 7 | 21 | 4 (57%) | 3 (43%) | 100% | Interstellar, Past Lives, Shang-Chi and the Legend of the Ten Rings |
| `pos_war` | 3 | 32 | 3 (100%) | 2 (67%) | 100% | — |
| `pos_hopeful` | 4 | 34 | 4 (100%) | 2 (50%) | 100% | — |
| `pos_bittersweet_romance` | 5 | 28 | 5 (100%) | 5 (100%) | 100% | — |
| `pos_grief` | 4 | 34 | 4 (100%) | 3 (75%) | 100% | — |
| `pos_coming_of_age` | 4 | 32 | 4 (100%) | 2 (50%) | 100% | — |
| `pos_class_satire` | 4 | 30 | 4 (100%) | 4 (100%) | 100% | — |
| `pos_feel_good` | 6 | 33 | 3 (50%) | 2 (33%) | 100% | Despicable Me 4, Mike and Dave Need Wedding Dates, Soul Plane |
| `pos_mind_bending` | 4 | 23 | 3 (75%) | 2 (50%) | 100% | Arrival |
| `pos_obsessive_ambition` | 3 | 28 | 3 (100%) | 1 (33%) | 100% | — |
