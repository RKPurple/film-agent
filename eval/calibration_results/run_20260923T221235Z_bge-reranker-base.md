# Reranker threshold calibration — 20260923T221235Z — BAAI/bge-reranker-base

- Reranker: `BAAI/bge-reranker-base`, embedding: `BAAI/bge-base-en-v1.5`, sentence-transformers 3.4.1
- 174 films; 15 positive, 12 negative queries
- Settings: RETRIEVAL_N_RESULTS=20, RRF_K=60, RERANK_TOP_N=8, RERANK_CONFIDENCE_THRESHOLD=0.5, CHUNK_TONE_WEIGHT=2, CHUNK_REVIEW_WEIGHT=3
- All scores are true logits; *sigmoid* = sigmoid(logit); *live-gate* = sigmoid(sigmoid(logit)), what `retrieve()` currently compares to the threshold (CrossEncoder.predict already applies sigmoid).
- dedup==current assertion held for 0 films with no tone and no review; dedup re-expanded by the tone/review weights == current held for all films.

## Timing and memory

- model_load_seconds: 2.064917042000161
- device: mps:0
- max_length: None
- n_parameters: 278044417
- memory_method: resource.getrusage(RUSAGE_SELF).ru_maxrss (PEAK RSS; psutil not installed)
- peak_rss_mb_before_model_load: 449.703125
- peak_rss_mb_after_model_load: 892.6875
- peak_rss_mb_increase_from_model_load: 442.984375
- mps_allocated_mb_after_model_load: 1060.663818359375
- score_all_films_median_seconds: 5.975293667001097
- live_rerank_median_seconds: 0.9524911659991631
- live_rerank_pool_size_median: 28.0
- live_rerank_pool_size_range: [20, 36]

## Terminal summary

```
Calibration run 20260923T221235Z  reranker BAAI/bge-reranker-base  (174 films, 15 positive + 12 negative queries)
  device mps:0, 278M params, max_length None; load 2.1s; score all 174 films 5.975s/query; live rerank (median pool 28) 0.952s/query
  peak RSS after model load 893 MB (+443 MB during load); MPS allocated 1061 MB

== current ==
  mean AUC over positive queries: 0.849  (min 0.607)
  observed logit range: -10.197 .. +1.439
  (i)  100% positives alive, most negatives gated: t=-5.589 logit (sigmoid 0.004; live-gate 0.5009): negatives gated 1/12, positives alive 15/15, film P=0.079 R=0.743 F1=0.143
  (ii) best balanced ((neg gated + pos alive)/2 = 0.650): t=-1.842 logit (sigmoid 0.137; live-gate 0.5341): negatives gated 10/12, positives alive 7/15, film P=0.311 R=0.200 F1=0.243
  best film-level F1: t=-2.551 logit (sigmoid 0.072; live-gate 0.5181): negatives gated 8/12, positives alive 8/15, film P=0.286 R=0.314 F1=0.299
  full separation: NO -- max negative top +0.612 vs min positive best-relevant -5.589 (gap -6.202)
  query-level, top score: AUC 0.794, separates=False (gap -4.497)
  query-level, margin:    AUC 0.772, separates=False (gap -3.149)

== dedup ==
  mean AUC over positive queries: 0.792  (min 0.433)
  observed logit range: -10.197 .. +1.089
  (i)  100% positives alive, most negatives gated: t=-4.878 logit (sigmoid 0.008; live-gate 0.5019): negatives gated 2/12, positives alive 15/15, film P=0.094 R=0.586 F1=0.162
  (ii) best balanced ((neg gated + pos alive)/2 = 0.583): t=-4.878 logit (sigmoid 0.008; live-gate 0.5019): negatives gated 2/12, positives alive 15/15, film P=0.094 R=0.586 F1=0.162
  best film-level F1: t=-2.286 logit (sigmoid 0.092; live-gate 0.5231): negatives gated 7/12, positives alive 7/15, film P=0.250 R=0.214 F1=0.231
  full separation: NO -- max negative top +0.976 vs min positive best-relevant -4.855 (gap -5.830)
  query-level, top score: AUC 0.728, separates=False (gap -5.027)
  query-level, margin:    AUC 0.778, separates=False (gap -2.846)

== live pipeline (current variant, pool = RRF of top 20 vector + BM25, rerank top 8) ==
  relevant films in fused pool: 75.7% overall, mean per query 79.6%
  relevant films in reranked top 8: 52.9% overall, mean per query 57.0%
  queries with NO relevant film in pool: none
  queries with NO relevant film in top 8: ['pos_unhinged_comedy']

Note: live retrieve() confidence = sigmoid(sigmoid(logit)) in [0.5, 0.731] (CrossEncoder.predict already applies sigmoid), so threshold 0.5 can never gate anything today.
```

## Variant: current

### Positive queries

| query | AUC | relevant ranks (of 174) | lowest relevant | highest irrelevant |
|---|---|---|---|---|
| `pos_uneasy` | 0.885 | Obsession (2025) #1, Companion (2025) #4, Weapons (2025) #7, Longlegs (2024) #10, Hereditary (2018) #42, Joker (2019) #86 | Joker (2019) -7.84 (0.000) | Us (2019) -2.16 (0.103) |
| `pos_found_family` | 0.781 | Deadpool 2 (2018) #1, Coco (2017) #3, Guardians of the Galaxy Vol. 2 (2017) #5, Guardians of the Galaxy (2014) #71, KPop Demon Hunters (2025) #124 | KPop Demon Hunters (2025) -8.49 (0.000) | Ted 2 (2015) -4.67 (0.009) |
| `pos_pleasant_surprise` | 0.882 | Flight (2012) #5, Crazy, Stupid, Love. (2011) #6, Scott Pilgrim vs. the World (2010) #10, Bugonia (2025) #23, Roofman (2025) #30, The Drama (2026) #71 | The Drama (2026) -7.15 (0.001) | Superman (2025) -1.41 (0.196) |
| `pos_plot_twist` | 0.607 | Avengers: Infinity War (2018) #2, Bugonia (2025) #10, The Hateful Eight (2015) #100, Parasite (2019) #165 | Parasite (2019) -5.99 (0.002) | I Love You, Man (2009) -1.42 (0.195) |
| `pos_unhinged_comedy` | 0.816 | 21 Jump Street (2012) #15, Deadpool (2016) #23, Pineapple Express (2008) #29, Ted 2 (2015) #33, Ted (2012) #76 | Ted (2012) -5.29 (0.005) | Joker (2019) +1.34 (0.793) |
| `pos_visually_stunning` | 0.732 | Doctor Strange (2016) #1, Past Lives (2023) #6, Project Hail Mary (2026) #8, Shang-Chi and the Legend of the Ten Rings (2021) #9, The Odyssey (2026) #85, Uncut Gems (2019) #100, Interstellar (2014) #138 | Interstellar (2014) -9.42 (0.000) | Supergirl (2026) -3.88 (0.020) |
| `pos_war` | 0.955 | The Odyssey (2026) #6, War for the Planet of the Apes (2017) #8, The Greatest Beer Run Ever (2022) #15 | The Greatest Beer Run Ever (2022) -5.38 (0.005) | Captain America: Civil War (2016) -2.62 (0.068) |
| `pos_hopeful` | 0.750 | Chef (2014) #4, Coco (2017) #6, Soul (2020) #8, Everything Everywhere All at Once (2022) #162 | Everything Everywhere All at Once (2022) -7.68 (0.000) | Big Hero 6 (2014) +0.36 (0.590) |
| `pos_bittersweet_romance` | 0.999 | (500) Days of Summer (2009) #1, Crazy, Stupid, Love. (2011) #2, Materialists (2025) #3, La La Land (2016) #4, Past Lives (2023) #7 | Past Lives (2023) -1.40 (0.198) | Guardians of the Galaxy Vol. 3 (2023) -0.92 (0.284) |
| `pos_grief` | 0.985 | Midsommar (2019) #1, Hereditary (2018) #4, Life of Pi (2012) #5, Flight (2012) #10 | Flight (2012) -3.73 (0.023) | Arrival (2016) -0.95 (0.279) |
| `pos_coming_of_age` | 0.950 | A Silent Voice: The Movie (2016) #1, CODA (2021) #2, Cha Cha Real Smooth (2022) #11, No Hard Feelings (2023) #30 | No Hard Feelings (2023) -4.97 (0.007) | GOAT (2026) -0.14 (0.464) |
| `pos_class_satire` | 0.979 | Parasite (2019) #1, American Psycho (2000) #4, Bugonia (2025) #9, Anora (2024) #10 | Anora (2024) -3.95 (0.019) | Training Day (2001) -2.92 (0.051) |
| `pos_feel_good` | 0.822 | How High (2001) #1, I Love You, Man (2009) #7, Despicable Me 4 (2024) #9, Soul Plane (2004) #51, Wedding Crashers (2005) #63, Mike and Dave Need Wedding Dates (2016) #70 | Mike and Dave Need Wedding Dates (2016) -5.07 (0.006) | The VelociPastor (2017) -0.77 (0.317) |
| `pos_mind_bending` | 0.811 | Arrival (2016) #10, Project Hail Mary (2026) #11, Interstellar (2014) #40, Nope (2022) #79 | Nope (2022) -7.76 (0.000) | Doctor Strange (2016) -0.93 (0.283) |
| `pos_obsessive_ambition` | 0.781 | King Richard (2021) #3, Whiplash (2014) #12, tick, tick... BOOM! (2021) #104 | tick, tick... BOOM! (2021) -7.57 (0.001) | The Social Network (2010) -3.79 (0.022) |

### Negative queries — top 3

| query | #1 | #2 | #3 |
|---|---|---|---|
| `neg_werewolves` | Avengers: Age of Ultron (2015) -4.21 (0.015) | Ocean's Eight (2018) -4.63 (0.010) | Project Hail Mary (2026) -4.63 (0.010) |
| `neg_silent_black_and_white` | Green Book (2018) -1.99 (0.120) | The Wolf of Wall Street (2013) -2.19 (0.100) | I Love You, Man (2009) -2.27 (0.094) |
| `neg_bollywood` | I Love You, Man (2009) -3.21 (0.039) | Crazy, Stupid, Love. (2011) -3.66 (0.025) | Bugonia (2025) -4.13 (0.016) |
| `neg_documentary` | I Love You, Man (2009) +0.61 (0.648) | Django Unchained (2012) +0.39 (0.597) | The Dark Knight (2008) +0.37 (0.591) |
| `neg_pirates` | Black Panther (2018) -4.09 (0.016) | Guardians of the Galaxy (2014) -4.20 (0.015) | Weapons (2025) -4.30 (0.013) |
| `neg_time_loop` | Green Book (2018) -1.41 (0.196) | Supergirl (2026) -1.93 (0.127) | I Love You, Man (2009) -2.10 (0.109) |
| `neg_horse_racing` | F1 (2025) -4.72 (0.009) | Joker: Folie à Deux (2024) -5.25 (0.005) | Nope (2022) -5.31 (0.005) |
| `neg_cold_war_espionage` | 21 Jump Street (2012) -2.27 (0.094) | Longlegs (2024) -2.36 (0.086) | Flight (2012) -2.46 (0.079) |
| `neg_mountaineering` | How High (2001) -4.02 (0.018) | Ocean's Eight (2018) -5.00 (0.007) | Project Hail Mary (2026) -5.29 (0.005) |
| `neg_chess` | King Richard (2021) -3.43 (0.031) | F1 (2025) -4.01 (0.018) | HIM (2025) -4.05 (0.017) |
| `neg_vineyard` | Chef (2014) -7.49 (0.001) | Midsommar (2019) -8.02 (0.000) | The Menu (2022) -8.32 (0.000) |
| `neg_ballet` | Cha Cha Real Smooth (2022) -2.93 (0.050) | Django Unchained (2012) -3.14 (0.041) | King Richard (2021) -3.48 (0.030) |

### Threshold sweep

Exact sweep over 4694 candidate thresholds; best points:

- **(i) all positives alive:** t=-5.589 logit (sigmoid 0.004; live-gate 0.5009): negatives gated 1/12, positives alive 15/15, film P=0.079 R=0.743 F1=0.143
- **(ii) best balanced:** t=-1.842 logit (sigmoid 0.137; live-gate 0.5341): negatives gated 10/12, positives alive 7/15, film P=0.311 R=0.200 F1=0.243
- **best film F1:** t=-2.551 logit (sigmoid 0.072; live-gate 0.5181): negatives gated 8/12, positives alive 8/15, film P=0.286 R=0.314 F1=0.299
- **Full separation: NO** — highest negative top score +0.612, lowest positive best-relevant score -5.589 (gap -6.202)

| t (logit) | sigmoid | live-gate | neg gated | pos alive | film P | film R | F1 |
|---|---|---|---|---|---|---|---|
| -10.197 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.712 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.029 | 1.000 | 0.056 |
| -9.227 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.029 | 0.986 | 0.057 |
| -8.742 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.031 | 0.986 | 0.061 |
| -8.258 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.032 | 0.943 | 0.063 |
| -7.773 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.036 | 0.929 | 0.069 |
| -7.288 | 0.001 | 0.5002 | 1/12 | 15/15 | 0.040 | 0.886 | 0.077 |
| -6.803 | 0.001 | 0.5003 | 1/12 | 15/15 | 0.045 | 0.829 | 0.086 |
| -6.318 | 0.002 | 0.5004 | 1/12 | 15/15 | 0.054 | 0.814 | 0.101 |
| -5.834 | 0.003 | 0.5007 | 1/12 | 15/15 | 0.066 | 0.757 | 0.122 |
| -5.349 | 0.005 | 0.5012 | 1/12 | 14/15 | 0.084 | 0.700 | 0.150 |
| -4.864 | 0.008 | 0.5019 | 1/12 | 13/15 | 0.098 | 0.614 | 0.169 |
| -4.379 | 0.012 | 0.5031 | 2/12 | 12/15 | 0.112 | 0.529 | 0.185 |
| -3.894 | 0.020 | 0.5050 | 5/12 | 11/15 | 0.129 | 0.429 | 0.199 |
| -3.409 | 0.032 | 0.5080 | 6/12 | 9/15 | 0.160 | 0.357 | 0.221 |
| -2.925 | 0.051 | 0.5127 | 8/12 | 8/15 | 0.216 | 0.314 | 0.256 |
| -2.440 | 0.080 | 0.5200 | 8/12 | 7/15 | 0.299 | 0.286 | 0.292 |
| -1.955 | 0.124 | 0.5310 | 10/12 | 7/15 | 0.304 | 0.200 | 0.241 |
| -1.470 | 0.187 | 0.5466 | 10/12 | 5/15 | 0.294 | 0.143 | 0.192 |
| -0.985 | 0.272 | 0.5675 | 11/12 | 5/15 | 0.375 | 0.129 | 0.191 |
| -0.501 | 0.377 | 0.5932 | 11/12 | 3/15 | 0.357 | 0.071 | 0.119 |
| -0.016 | 0.496 | 0.6215 | 11/12 | 2/15 | 0.375 | 0.043 | 0.077 |
| +0.469 | 0.615 | 0.6491 | 11/12 | 2/15 | 0.500 | 0.029 | 0.054 |
| +0.954 | 0.722 | 0.6730 | 12/12 | 2/15 | 0.500 | 0.029 | 0.054 |
| +1.439 | 0.808 | 0.6917 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |

### Top score vs. margin (top − median over all films)

- Top score: AUC(pos vs neg queries) 0.794, separates=False, gap -4.497
- Margin: AUC(pos vs neg queries) 0.772, separates=False, gap -3.149

| query | type | top | median | margin |
|---|---|---|---|---|
| `pos_bittersweet_romance` | positive | +1.439 | -7.065 | +8.504 |
| `pos_uneasy` | positive | +1.399 | -7.847 | +9.246 |
| `pos_unhinged_comedy` | positive | +1.344 | -5.531 | +6.875 |
| `neg_documentary` | negative | +0.612 | -4.168 | +4.780 |
| `pos_hopeful` | positive | +0.365 | -6.225 | +6.590 |
| `pos_coming_of_age` | positive | -0.057 | -6.417 | +6.359 |
| `pos_pleasant_surprise` | positive | -0.326 | -7.623 | +7.297 |
| `pos_feel_good` | positive | -0.616 | -5.493 | +4.877 |
| `pos_grief` | positive | -0.699 | -6.001 | +5.302 |
| `pos_mind_bending` | positive | -0.928 | -7.928 | +7.000 |
| `neg_time_loop` | negative | -1.414 | -5.559 | +4.145 |
| `pos_plot_twist` | positive | -1.421 | -3.831 | +2.410 |
| `neg_silent_black_and_white` | negative | -1.991 | -4.769 | +2.777 |
| `neg_cold_war_espionage` | negative | -2.267 | -5.386 | +3.119 |
| `pos_class_satire` | positive | -2.478 | -6.020 | +3.543 |
| `pos_war` | positive | -2.618 | -8.775 | +6.156 |
| `neg_ballet` | negative | -2.935 | -7.219 | +4.284 |
| `neg_bollywood` | negative | -3.205 | -8.219 | +5.013 |
| `neg_chess` | negative | -3.435 | -8.993 | +5.558 |
| `pos_obsessive_ambition` | positive | -3.786 | -7.228 | +3.442 |
| `pos_visually_stunning` | positive | -3.873 | -8.351 | +4.477 |
| `pos_found_family` | positive | -3.884 | -7.438 | +3.554 |
| `neg_mountaineering` | negative | -4.024 | -8.796 | +4.772 |
| `neg_pirates` | negative | -4.090 | -6.861 | +2.771 |
| `neg_werewolves` | negative | -4.206 | -6.980 | +2.774 |
| `neg_horse_racing` | negative | -4.717 | -9.249 | +4.532 |
| `neg_vineyard` | negative | -7.487 | -10.195 | +2.708 |

## Variant: dedup

### Positive queries

| query | AUC | relevant ranks (of 174) | lowest relevant | highest irrelevant |
|---|---|---|---|---|
| `pos_uneasy` | 0.889 | Obsession (2025) #1, Companion (2025) #5, Weapons (2025) #7, Longlegs (2024) #14, Hereditary (2018) #24, Joker (2019) #95 | Joker (2019) -7.87 (0.000) | Nope (2022) -2.42 (0.081) |
| `pos_found_family` | 0.649 | Coco (2017) #8, Guardians of the Galaxy Vol. 2 (2017) #13, Deadpool 2 (2018) #31, Guardians of the Galaxy (2014) #100, KPop Demon Hunters (2025) #164 | KPop Demon Hunters (2025) -10.15 (0.000) | How to Train Your Dragon (2010) -4.23 (0.014) |
| `pos_pleasant_surprise` | 0.839 | Flight (2012) #5, Crazy, Stupid, Love. (2011) #8, Roofman (2025) #11, Scott Pilgrim vs. the World (2010) #16, Bugonia (2025) #71, The Drama (2026) #77 | The Drama (2026) -7.58 (0.001) | Superman (2025) -2.43 (0.081) |
| `pos_plot_twist` | 0.433 | Avengers: Infinity War (2018) #28, Bugonia (2025) #67, The Hateful Eight (2015) #130, Parasite (2019) #169 | Parasite (2019) -6.84 (0.001) | I Love You, Man (2009) -0.94 (0.281) |
| `pos_unhinged_comedy` | 0.569 | 21 Jump Street (2012) #22, Deadpool (2016) #31, Ted 2 (2015) #82, Pineapple Express (2008) #109, Ted (2012) #139 | Ted (2012) -6.95 (0.001) | Joker (2019) +1.09 (0.748) |
| `pos_visually_stunning` | 0.807 | Doctor Strange (2016) #1, Project Hail Mary (2026) #5, Shang-Chi and the Legend of the Ten Rings (2021) #9, The Odyssey (2026) #24, Past Lives (2023) #28, Uncut Gems (2019) #42, Interstellar (2014) #150 | Interstellar (2014) -9.83 (0.000) | Supergirl (2026) -4.29 (0.014) |
| `pos_war` | 0.943 | War for the Planet of the Apes (2017) #4, The Odyssey (2026) #5, The Greatest Beer Run Ever (2022) #26 | The Greatest Beer Run Ever (2022) -5.96 (0.003) | Captain America: Civil War (2016) -1.75 (0.149) |
| `pos_hopeful` | 0.740 | Chef (2014) #5, Coco (2017) #8, Soul (2020) #12, Everything Everywhere All at Once (2022) #162 | Everything Everywhere All at Once (2022) -8.04 (0.000) | Big Hero 6 (2014) -0.38 (0.406) |
| `pos_bittersweet_romance` | 0.995 | (500) Days of Summer (2009) #1, Crazy, Stupid, Love. (2011) #2, Materialists (2025) #3, La La Land (2016) #7, Past Lives (2023) #8 | Past Lives (2023) -2.26 (0.095) | Guardians of the Galaxy Vol. 3 (2023) -1.61 (0.166) |
| `pos_grief` | 0.990 | Midsommar (2019) #2, Life of Pi (2012) #3, Hereditary (2018) #5, Flight (2012) #7 | Flight (2012) -3.43 (0.031) | Arrival (2016) -0.59 (0.357) |
| `pos_coming_of_age` | 0.900 | CODA (2021) #2, A Silent Voice: The Movie (2016) #3, Cha Cha Real Smooth (2022) #34, No Hard Feelings (2023) #39 | No Hard Feelings (2023) -4.94 (0.007) | GOAT (2026) -0.09 (0.479) |
| `pos_class_satire` | 0.928 | Parasite (2019) #1, American Psycho (2000) #6, Anora (2024) #21, Bugonia (2025) #30 | Bugonia (2025) -4.66 (0.009) | I Love You, Man (2009) -2.74 (0.061) |
| `pos_feel_good` | 0.828 | I Love You, Man (2009) #2, How High (2001) #3, Despicable Me 4 (2024) #5, Soul Plane (2004) #53, Mike and Dave Need Wedding Dates (2016) #61, Wedding Crashers (2005) #71 | Wedding Crashers (2005) -5.10 (0.006) | Crazy, Stupid, Love. (2011) -0.33 (0.417) |
| `pos_mind_bending` | 0.664 | Project Hail Mary (2026) #9, Arrival (2016) #45, Interstellar (2014) #89, Nope (2022) #97 | Nope (2022) -8.06 (0.000) | Doctor Strange (2016) -1.09 (0.252) |
| `pos_obsessive_ambition` | 0.704 | Whiplash (2014) #16, King Richard (2021) #43, tick, tick... BOOM! (2021) #100 | tick, tick... BOOM! (2021) -7.76 (0.000) | Green Book (2018) -2.63 (0.067) |

### Negative queries — top 3

| query | #1 | #2 | #3 |
|---|---|---|---|
| `neg_werewolves` | Project Hail Mary (2026) -3.57 (0.027) | Deadpool (2016) -3.59 (0.027) | The Avengers (2012) -3.75 (0.023) |
| `neg_silent_black_and_white` | Roofman (2025) -1.12 (0.245) | Green Book (2018) -1.27 (0.219) | Project Hail Mary (2026) -2.09 (0.110) |
| `neg_bollywood` | I Love You, Man (2009) -2.73 (0.061) | Crazy, Stupid, Love. (2011) -2.82 (0.056) | Green Book (2018) -3.29 (0.036) |
| `neg_documentary` | Rise of the Planet of the Apes (2011) +0.98 (0.726) | The Dark Knight (2008) +0.70 (0.668) | Deadpool (2016) +0.65 (0.657) |
| `neg_pirates` | Green Book (2018) -3.56 (0.028) | Hereditary (2018) -3.71 (0.024) | Project Hail Mary (2026) -3.80 (0.022) |
| `neg_time_loop` | Green Book (2018) +0.24 (0.561) | I Love You, Man (2009) -0.89 (0.291) | Supergirl (2026) -1.18 (0.234) |
| `neg_horse_racing` | Ford v Ferrari (2019) -5.04 (0.006) | F1 (2025) -5.33 (0.005) | The Dark Knight (2008) -5.48 (0.004) |
| `neg_cold_war_espionage` | Longlegs (2024) -1.92 (0.128) | Flight (2012) -2.09 (0.110) | 21 Jump Street (2012) -2.65 (0.066) |
| `neg_mountaineering` | Project Hail Mary (2026) -3.39 (0.033) | How High (2001) -3.43 (0.031) | Ant-Man (2015) -4.25 (0.014) |
| `neg_chess` | Ford v Ferrari (2019) -3.03 (0.046) | HIM (2025) -3.36 (0.034) | Uncut Gems (2019) -3.57 (0.027) |
| `neg_vineyard` | Chef (2014) -7.00 (0.001) | Midsommar (2019) -7.41 (0.001) | Green Book (2018) -8.08 (0.000) |
| `neg_ballet` | Green Book (2018) -1.97 (0.123) | Supergirl (2026) -2.30 (0.091) | Coco (2017) -2.34 (0.088) |

### Threshold sweep

Exact sweep over 4686 candidate thresholds; best points:

- **(i) all positives alive:** t=-4.878 logit (sigmoid 0.008; live-gate 0.5019): negatives gated 2/12, positives alive 15/15, film P=0.094 R=0.586 F1=0.162
- **(ii) best balanced:** t=-4.878 logit (sigmoid 0.008; live-gate 0.5019): negatives gated 2/12, positives alive 15/15, film P=0.094 R=0.586 F1=0.162
- **best film F1:** t=-2.286 logit (sigmoid 0.092; live-gate 0.5231): negatives gated 7/12, positives alive 7/15, film P=0.250 R=0.214 F1=0.231
- **Full separation: NO** — highest negative top score +0.976, lowest positive best-relevant score -4.855 (gap -5.830)

| t (logit) | sigmoid | live-gate | neg gated | pos alive | film P | film R | F1 |
|---|---|---|---|---|---|---|---|
| -10.197 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.727 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.028 | 0.971 | 0.055 |
| -9.256 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.030 | 0.971 | 0.057 |
| -8.786 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.032 | 0.971 | 0.061 |
| -8.316 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.034 | 0.971 | 0.066 |
| -7.846 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.036 | 0.900 | 0.069 |
| -7.375 | 0.001 | 0.5002 | 0/12 | 15/15 | 0.040 | 0.843 | 0.076 |
| -6.905 | 0.001 | 0.5003 | 1/12 | 15/15 | 0.046 | 0.814 | 0.088 |
| -6.435 | 0.002 | 0.5004 | 1/12 | 15/15 | 0.054 | 0.771 | 0.101 |
| -5.965 | 0.003 | 0.5006 | 1/12 | 15/15 | 0.063 | 0.714 | 0.115 |
| -5.494 | 0.004 | 0.5010 | 1/12 | 15/15 | 0.076 | 0.643 | 0.136 |
| -5.024 | 0.007 | 0.5016 | 2/12 | 15/15 | 0.091 | 0.614 | 0.159 |
| -4.554 | 0.010 | 0.5026 | 2/12 | 13/15 | 0.096 | 0.486 | 0.161 |
| -4.083 | 0.017 | 0.5041 | 2/12 | 12/15 | 0.121 | 0.443 | 0.190 |
| -3.613 | 0.026 | 0.5066 | 2/12 | 9/15 | 0.133 | 0.343 | 0.192 |
| -3.143 | 0.041 | 0.5103 | 5/12 | 7/15 | 0.153 | 0.271 | 0.196 |
| -2.673 | 0.065 | 0.5161 | 7/12 | 7/15 | 0.200 | 0.243 | 0.219 |
| -2.202 | 0.100 | 0.5249 | 7/12 | 6/15 | 0.245 | 0.186 | 0.211 |
| -1.732 | 0.150 | 0.5375 | 9/12 | 5/15 | 0.294 | 0.143 | 0.192 |
| -1.262 | 0.221 | 0.5549 | 9/12 | 4/15 | 0.318 | 0.100 | 0.152 |
| -0.792 | 0.312 | 0.5773 | 10/12 | 3/15 | 0.286 | 0.057 | 0.095 |
| -0.321 | 0.420 | 0.6036 | 10/12 | 3/15 | 0.429 | 0.043 | 0.078 |
| +0.149 | 0.537 | 0.6312 | 10/12 | 2/15 | 0.667 | 0.029 | 0.055 |
| +0.619 | 0.650 | 0.6570 | 11/12 | 0/15 | 0.000 | 0.000 | 0.000 |
| +1.089 | 0.748 | 0.6788 | 12/12 | 0/15 | 0.000 | 0.000 | 0.000 |

### Top score vs. margin (top − median over all films)

- Top score: AUC(pos vs neg queries) 0.728, separates=False, gap -5.027
- Margin: AUC(pos vs neg queries) 0.778, separates=False, gap -2.846

| query | type | top | median | margin |
|---|---|---|---|---|
| `pos_unhinged_comedy` | positive | +1.089 | -5.874 | +6.963 |
| `neg_documentary` | negative | +0.976 | -4.100 | +5.076 |
| `pos_uneasy` | positive | +0.277 | -7.791 | +8.069 |
| `pos_bittersweet_romance` | positive | +0.257 | -7.393 | +7.651 |
| `neg_time_loop` | negative | +0.243 | -6.138 | +6.382 |
| `pos_coming_of_age` | positive | -0.085 | -6.665 | +6.579 |
| `pos_feel_good` | positive | -0.335 | -5.732 | +5.398 |
| `pos_hopeful` | positive | -0.379 | -6.396 | +6.017 |
| `pos_grief` | positive | -0.588 | -6.346 | +5.758 |
| `pos_plot_twist` | positive | -0.937 | -4.473 | +3.536 |
| `pos_pleasant_surprise` | positive | -1.046 | -7.810 | +6.764 |
| `pos_mind_bending` | positive | -1.090 | -7.914 | +6.824 |
| `neg_silent_black_and_white` | negative | -1.124 | -5.121 | +3.997 |
| `pos_war` | positive | -1.746 | -8.774 | +7.029 |
| `neg_cold_war_espionage` | negative | -1.918 | -5.723 | +3.804 |
| `neg_ballet` | negative | -1.965 | -7.217 | +5.252 |
| `pos_class_satire` | positive | -2.286 | -6.610 | +4.324 |
| `pos_obsessive_ambition` | positive | -2.632 | -7.494 | +4.862 |
| `neg_bollywood` | negative | -2.733 | -7.962 | +5.229 |
| `neg_chess` | negative | -3.033 | -8.668 | +5.635 |
| `neg_mountaineering` | negative | -3.389 | -8.532 | +5.143 |
| `neg_pirates` | negative | -3.560 | -6.778 | +3.218 |
| `neg_werewolves` | negative | -3.566 | -6.967 | +3.401 |
| `pos_visually_stunning` | positive | -3.805 | -8.526 | +4.721 |
| `pos_found_family` | positive | -4.052 | -7.635 | +3.584 |
| `neg_horse_racing` | negative | -5.037 | -8.903 | +3.866 |
| `neg_vineyard` | negative | -6.999 | -10.195 | +3.196 |

## Live pipeline view (current variant)

Fused pool = RRF of top 20 vector + top 20 BM25; gate sees reranked top 8. Films missing from the pool are a retrieval-recall problem no threshold can fix.

- In fused pool: 75.7% of relevant films overall (mean per query 79.6%)
- In reranked top 8: 52.9% overall (mean per query 57.0%)
- Queries with no relevant film in pool: none
- Queries with no relevant film in top 8: ['pos_unhinged_comedy']

| query | relevant | pool size | in pool | in top 8 | top-8 ceiling | missing from pool |
|---|---|---|---|---|---|---|
| `pos_uneasy` | 6 | 27 | 5 (83%) | 3 (50%) | 100% | Joker |
| `pos_found_family` | 5 | 31 | 3 (60%) | 3 (60%) | 100% | Guardians of the Galaxy, KPop Demon Hunters |
| `pos_pleasant_surprise` | 6 | 36 | 2 (33%) | 1 (17%) | 100% | Bugonia, Flight, Scott Pilgrim vs. the World, The Drama |
| `pos_plot_twist` | 4 | 27 | 3 (75%) | 2 (50%) | 100% | Parasite |
| `pos_unhinged_comedy` | 5 | 34 | 3 (60%) | 0 (0%) | 100% | Deadpool, Ted |
| `pos_visually_stunning` | 7 | 21 | 4 (57%) | 2 (29%) | 100% | Interstellar, Past Lives, Shang-Chi and the Legend of the Ten Rings |
| `pos_war` | 3 | 32 | 3 (100%) | 3 (100%) | 100% | — |
| `pos_hopeful` | 4 | 34 | 4 (100%) | 3 (75%) | 100% | — |
| `pos_bittersweet_romance` | 5 | 28 | 5 (100%) | 5 (100%) | 100% | — |
| `pos_grief` | 4 | 34 | 4 (100%) | 3 (75%) | 100% | — |
| `pos_coming_of_age` | 4 | 32 | 4 (100%) | 3 (75%) | 100% | — |
| `pos_class_satire` | 4 | 30 | 4 (100%) | 4 (100%) | 100% | — |
| `pos_feel_good` | 6 | 33 | 3 (50%) | 2 (33%) | 100% | Despicable Me 4, Mike and Dave Need Wedding Dates, Soul Plane |
| `pos_mind_bending` | 4 | 23 | 3 (75%) | 1 (25%) | 100% | Arrival |
| `pos_obsessive_ambition` | 3 | 28 | 3 (100%) | 2 (67%) | 100% | — |
