# Reranker threshold calibration — 20260923T220619Z — BAAI/bge-reranker-v2-m3

- Reranker: `BAAI/bge-reranker-v2-m3`, embedding: `BAAI/bge-base-en-v1.5`, sentence-transformers 3.4.1
- 174 films; 15 positive, 12 negative queries
- Settings: RETRIEVAL_N_RESULTS=20, RRF_K=60, RERANK_TOP_N=8, RERANK_CONFIDENCE_THRESHOLD=0.5, CHUNK_TONE_WEIGHT=2, CHUNK_REVIEW_WEIGHT=3
- All scores are true logits; *sigmoid* = sigmoid(logit); *live-gate* = sigmoid(sigmoid(logit)), what `retrieve()` currently compares to the threshold (CrossEncoder.predict already applies sigmoid).
- dedup==current assertion held for 0 films with no tone and no review; dedup re-expanded by the tone/review weights == current held for all films.

## Timing and memory

- model_load_seconds: 2.9118257499994797
- device: mps:0
- max_length: None
- n_parameters: 567755777
- memory_method: resource.getrusage(RUSAGE_SELF).ru_maxrss (PEAK RSS; psutil not installed)
- peak_rss_mb_before_model_load: 450.65625
- peak_rss_mb_after_model_load: 853.859375
- peak_rss_mb_increase_from_model_load: 403.203125
- mps_allocated_mb_after_model_load: 2165.942138671875
- score_all_films_median_seconds: 19.97944279100011
- live_rerank_median_seconds: 3.1965561669994713
- live_rerank_pool_size_median: 28.0
- live_rerank_pool_size_range: [20, 36]

## Terminal summary

```
Calibration run 20260923T220619Z  reranker BAAI/bge-reranker-v2-m3  (174 films, 15 positive + 12 negative queries)
  device mps:0, 568M params, max_length None; load 2.9s; score all 174 films 19.979s/query; live rerank (median pool 28) 3.197s/query
  peak RSS after model load 854 MB (+403 MB during load); MPS allocated 2166 MB

== current ==
  mean AUC over positive queries: 0.781  (min 0.485)
  observed logit range: -10.935 .. +1.665
  (i)  100% positives alive, most negatives gated: t=-6.621 logit (sigmoid 0.001; live-gate 0.5003): negatives gated 0/12, positives alive 15/15, film P=0.032 R=0.914 F1=0.061
  (ii) best balanced ((neg gated + pos alive)/2 = 0.717): t=-2.650 logit (sigmoid 0.066; live-gate 0.5165): negatives gated 10/12, positives alive 9/15, film P=0.286 R=0.200 F1=0.235
  best film-level F1: t=-2.650 logit (sigmoid 0.066; live-gate 0.5165): negatives gated 10/12, positives alive 9/15, film P=0.286 R=0.200 F1=0.235
  full separation: NO -- max negative top -1.944 vs min positive best-relevant -6.614 (gap -4.670)
  query-level, top score: AUC 0.806, separates=False (gap -2.838)
  query-level, margin:    AUC 0.894, separates=False (gap -0.919)

== dedup ==
  mean AUC over positive queries: 0.754  (min 0.444)
  observed logit range: -10.986 .. +1.256
  (i)  100% positives alive, most negatives gated: t=-5.155 logit (sigmoid 0.006; live-gate 0.5014): negatives gated 2/12, positives alive 15/15, film P=0.037 R=0.786 F1=0.071
  (ii) best balanced ((neg gated + pos alive)/2 = 0.650): t=-2.105 logit (sigmoid 0.109; live-gate 0.5271): negatives gated 10/12, positives alive 7/15, film P=0.204 R=0.143 F1=0.168
  best film-level F1: t=-2.152 logit (sigmoid 0.104; live-gate 0.5260): negatives gated 9/12, positives alive 7/15, film P=0.204 R=0.157 F1=0.177
  full separation: NO -- max negative top -1.404 vs min positive best-relevant -5.136 (gap -3.732)
  query-level, top score: AUC 0.806, separates=False (gap -2.870)
  query-level, margin:    AUC 0.883, separates=False (gap -0.593)

== live pipeline (current variant, pool = RRF of top 20 vector + BM25, rerank top 8) ==
  relevant films in fused pool: 75.7% overall, mean per query 79.6%
  relevant films in reranked top 8: 45.7% overall, mean per query 48.3%
  queries with NO relevant film in pool: none
  queries with NO relevant film in top 8: ['pos_unhinged_comedy']

Note: live retrieve() confidence = sigmoid(sigmoid(logit)) in [0.5, 0.731] (CrossEncoder.predict already applies sigmoid), so threshold 0.5 can never gate anything today.
```

## Variant: current

### Positive queries

| query | AUC | relevant ranks (of 174) | lowest relevant | highest irrelevant |
|---|---|---|---|---|
| `pos_uneasy` | 0.944 | Obsession (2025) #1, Weapons (2025) #3, Companion (2025) #6, Longlegs (2024) #15, Joker (2019) #17, Hereditary (2018) #49 | Hereditary (2018) -5.10 (0.006) | Nope (2022) -2.26 (0.095) |
| `pos_found_family` | 0.606 | Guardians of the Galaxy Vol. 2 (2017) #2, Deadpool 2 (2018) #51, Coco (2017) #92, Guardians of the Galaxy (2014) #96, KPop Demon Hunters (2025) #107 | KPop Demon Hunters (2025) -6.62 (0.001) | The Social Network (2010) -3.61 (0.026) |
| `pos_pleasant_surprise` | 0.707 | Scott Pilgrim vs. the World (2010) #10, Crazy, Stupid, Love. (2011) #14, Roofman (2025) #30, Flight (2012) #46, The Drama (2026) #95, Bugonia (2025) #126 | Bugonia (2025) -8.53 (0.000) | Superman (2025) -4.15 (0.016) |
| `pos_plot_twist` | 0.485 | Avengers: Infinity War (2018) #10, The Hateful Eight (2015) #88, Bugonia (2025) #108, Parasite (2019) #152 | Parasite (2019) -5.20 (0.005) | Black Panther (2018) -1.76 (0.147) |
| `pos_unhinged_comedy` | 0.595 | Deadpool (2016) #12, 21 Jump Street (2012) #38, Ted 2 (2015) #70, Ted (2012) #93, Pineapple Express (2008) #147 | Pineapple Express (2008) -5.94 (0.003) | Pizza Movie (2026) +0.09 (0.523) |
| `pos_visually_stunning` | 0.759 | Doctor Strange (2016) #1, Past Lives (2023) #3, The Odyssey (2026) #35, Uncut Gems (2019) #41, Shang-Chi and the Legend of the Ten Rings (2021) #63, Project Hail Mary (2026) #67, Interstellar (2014) #102 | Interstellar (2014) -5.81 (0.003) | Black Panther (2018) -2.63 (0.067) |
| `pos_war` | 0.867 | War for the Planet of the Apes (2017) #2, The Greatest Beer Run Ever (2022) #9, The Odyssey (2026) #63 | The Odyssey (2026) -7.15 (0.001) | Eternals (2021) -4.78 (0.008) |
| `pos_hopeful` | 0.820 | Chef (2014) #4, Coco (2017) #14, Soul (2020) #16, Everything Everywhere All at Once (2022) #99 | Everything Everywhere All at Once (2022) -5.60 (0.004) | Big Hero 6 (2014) -0.59 (0.357) |
| `pos_bittersweet_romance` | 0.983 | (500) Days of Summer (2009) #1, La La Land (2016) #2, Crazy, Stupid, Love. (2011) #3, Materialists (2025) #9, Past Lives (2023) #14 | Past Lives (2023) -3.57 (0.027) | How to Train Your Dragon (2010) -3.04 (0.046) |
| `pos_grief` | 0.888 | Life of Pi (2012) #1, Midsommar (2019) #4, Hereditary (2018) #9, Flight (2012) #73 | Flight (2012) -5.23 (0.005) | Arrival (2016) -3.32 (0.035) |
| `pos_coming_of_age` | 0.874 | CODA (2021) #1, A Silent Voice: The Movie (2016) #2, No Hard Feelings (2023) #35, Cha Cha Real Smooth (2022) #58 | Cha Cha Real Smooth (2022) -3.66 (0.025) | Black Panther (2018) -2.18 (0.102) |
| `pos_class_satire` | 0.777 | Parasite (2019) #1, Anora (2024) #44, American Psycho (2000) #46, Bugonia (2025) #69 | Bugonia (2025) -4.32 (0.013) | The Social Network (2010) -2.80 (0.057) |
| `pos_feel_good` | 0.739 | How High (2001) #1, I Love You, Man (2009) #4, Mike and Dave Need Wedding Dates (2016) #42, Soul Plane (2004) #48, Wedding Crashers (2005) #57, Despicable Me 4 (2024) #130 | Despicable Me 4 (2024) -6.18 (0.002) | The Switch (2010) -1.77 (0.146) |
| `pos_mind_bending` | 0.729 | Interstellar (2014) #13, Project Hail Mary (2026) #46, Arrival (2016) #47, Nope (2022) #91 | Nope (2022) -4.11 (0.016) | Avengers: Infinity War (2018) -1.82 (0.139) |
| `pos_obsessive_ambition` | 0.941 | Whiplash (2014) #1, King Richard (2021) #8, tick, tick... BOOM! (2021) #28 | tick, tick... BOOM! (2021) -5.67 (0.003) | The Social Network (2010) -4.17 (0.015) |

### Negative queries — top 3

| query | #1 | #2 | #3 |
|---|---|---|---|
| `neg_werewolves` | King Richard (2021) -3.58 (0.027) | The Social Network (2010) -3.81 (0.022) | Spider-Man: Homecoming (2017) -4.07 (0.017) |
| `neg_silent_black_and_white` | A Bronx Tale (1993) -2.03 (0.117) | The Dark Knight (2008) -2.24 (0.096) | Spider-Man: Homecoming (2017) -2.33 (0.089) |
| `neg_bollywood` | Encanto (2021) -2.83 (0.056) | Harold & Kumar Go to White Castle (2004) -2.95 (0.050) | Spider-Man: Into the Spider-Verse (2018) -3.48 (0.030) |
| `neg_documentary` | Spider-Man: Homecoming (2017) -1.94 (0.125) | The Social Network (2010) -2.18 (0.102) | Captain America: Civil War (2016) -2.19 (0.101) |
| `neg_pirates` | King Richard (2021) -2.67 (0.065) | Black Panther (2018) -3.55 (0.028) | The Dark Knight (2008) -3.70 (0.024) |
| `neg_time_loop` | La La Land (2016) -4.18 (0.015) | Jujutsu Kaisen 0 (2021) -4.22 (0.014) | The Social Network (2010) -4.30 (0.013) |
| `neg_horse_racing` | Thor: Ragnarok (2017) -3.63 (0.026) | Thor (2011) -4.01 (0.018) | Captain America: Civil War (2016) -4.06 (0.017) |
| `neg_cold_war_espionage` | Deadpool (2016) -2.84 (0.055) | Iron Man (2008) -2.90 (0.052) | Thor: Ragnarok (2017) -3.07 (0.045) |
| `neg_mountaineering` | Iron Man 2 (2010) -4.20 (0.015) | Deadpool (2016) -4.28 (0.014) | Captain America: Civil War (2016) -4.29 (0.014) |
| `neg_chess` | Big Hero 6 (2014) -5.34 (0.005) | Eternals (2021) -5.43 (0.004) | Spider-Man: Brand New Day (2026) -5.74 (0.003) |
| `neg_vineyard` | Thor (2011) -5.77 (0.003) | The Social Network (2010) -6.02 (0.002) | Thor: The Dark World (2013) -6.05 (0.002) |
| `neg_ballet` | Whiplash (2014) -3.08 (0.044) | Wonder Woman (2017) -4.31 (0.013) | Encanto (2021) -4.35 (0.013) |

### Threshold sweep

Exact sweep over 4696 candidate thresholds; best points:

- **(i) all positives alive:** t=-6.621 logit (sigmoid 0.001; live-gate 0.5003): negatives gated 0/12, positives alive 15/15, film P=0.032 R=0.914 F1=0.061
- **(ii) best balanced:** t=-2.650 logit (sigmoid 0.066; live-gate 0.5165): negatives gated 10/12, positives alive 9/15, film P=0.286 R=0.200 F1=0.235
- **best film F1:** t=-2.650 logit (sigmoid 0.066; live-gate 0.5165): negatives gated 10/12, positives alive 9/15, film P=0.286 R=0.200 F1=0.235
- **Full separation: NO** — highest negative top score -1.944, lowest positive best-relevant score -6.614 (gap -4.670)

| t (logit) | sigmoid | live-gate | neg gated | pos alive | film P | film R | F1 |
|---|---|---|---|---|---|---|---|
| -10.935 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -10.410 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.885 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.360 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.028 | 1.000 | 0.054 |
| -8.835 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.028 | 1.000 | 0.054 |
| -8.310 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.028 | 0.986 | 0.055 |
| -7.785 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.029 | 0.971 | 0.056 |
| -7.260 | 0.001 | 0.5002 | 0/12 | 15/15 | 0.030 | 0.957 | 0.058 |
| -6.735 | 0.001 | 0.5003 | 0/12 | 15/15 | 0.031 | 0.914 | 0.060 |
| -6.210 | 0.002 | 0.5005 | 0/12 | 14/15 | 0.034 | 0.857 | 0.065 |
| -5.685 | 0.003 | 0.5008 | 1/12 | 14/15 | 0.037 | 0.800 | 0.071 |
| -5.160 | 0.006 | 0.5014 | 2/12 | 14/15 | 0.045 | 0.714 | 0.084 |
| -4.635 | 0.010 | 0.5024 | 2/12 | 13/15 | 0.056 | 0.614 | 0.102 |
| -4.110 | 0.016 | 0.5040 | 4/12 | 12/15 | 0.071 | 0.471 | 0.124 |
| -3.585 | 0.027 | 0.5067 | 5/12 | 10/15 | 0.102 | 0.343 | 0.157 |
| -3.060 | 0.045 | 0.5112 | 7/12 | 10/15 | 0.172 | 0.243 | 0.201 |
| -2.535 | 0.073 | 0.5184 | 10/12 | 7/15 | 0.268 | 0.157 | 0.198 |
| -2.010 | 0.118 | 0.5295 | 11/12 | 5/15 | 0.357 | 0.071 | 0.119 |
| -1.485 | 0.185 | 0.5460 | 12/12 | 4/15 | 0.571 | 0.057 | 0.104 |
| -0.960 | 0.277 | 0.5688 | 12/12 | 1/15 | 0.333 | 0.014 | 0.027 |
| -0.435 | 0.393 | 0.5970 | 12/12 | 1/15 | 0.500 | 0.014 | 0.028 |
| +0.090 | 0.523 | 0.6277 | 12/12 | 1/15 | 0.500 | 0.014 | 0.028 |
| +0.615 | 0.649 | 0.6568 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +1.140 | 0.758 | 0.6809 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +1.665 | 0.841 | 0.6987 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |

### Top score vs. margin (top − median over all films)

- Top score: AUC(pos vs neg queries) 0.806, separates=False, gap -2.838
- Margin: AUC(pos vs neg queries) 0.894, separates=False, gap -0.919

| query | type | top | median | margin |
|---|---|---|---|---|
| `pos_uneasy` | positive | +1.665 | -5.567 | +7.233 |
| `pos_unhinged_comedy` | positive | +0.091 | -5.146 | +5.237 |
| `pos_hopeful` | positive | -0.591 | -5.363 | +4.772 |
| `pos_visually_stunning` | positive | -1.161 | -5.575 | +4.413 |
| `pos_bittersweet_romance` | positive | -1.360 | -4.855 | +3.494 |
| `pos_coming_of_age` | positive | -1.387 | -4.155 | +2.768 |
| `pos_feel_good` | positive | -1.686 | -5.430 | +3.744 |
| `pos_plot_twist` | positive | -1.762 | -4.318 | +2.556 |
| `pos_mind_bending` | positive | -1.824 | -4.049 | +2.225 |
| `neg_documentary` | negative | -1.944 | -3.623 | +1.679 |
| `neg_silent_black_and_white` | negative | -2.025 | -4.063 | +2.037 |
| `pos_class_satire` | positive | -2.300 | -4.603 | +2.304 |
| `pos_grief` | positive | -2.300 | -5.452 | +3.152 |
| `neg_pirates` | negative | -2.667 | -5.450 | +2.784 |
| `neg_bollywood` | negative | -2.832 | -5.290 | +2.458 |
| `neg_cold_war_espionage` | negative | -2.840 | -4.730 | +1.890 |
| `neg_ballet` | negative | -3.076 | -6.219 | +3.144 |
| `pos_pleasant_surprise` | positive | -3.471 | -8.019 | +4.548 |
| `neg_werewolves` | negative | -3.579 | -5.841 | +2.263 |
| `pos_found_family` | positive | -3.608 | -6.382 | +2.773 |
| `neg_horse_racing` | negative | -3.631 | -5.911 | +2.281 |
| `pos_obsessive_ambition` | positive | -3.779 | -6.601 | +2.822 |
| `neg_time_loop` | negative | -4.179 | -6.476 | +2.297 |
| `neg_mountaineering` | negative | -4.204 | -6.267 | +2.063 |
| `pos_war` | positive | -4.781 | -7.753 | +2.971 |
| `neg_chess` | negative | -5.344 | -7.555 | +2.211 |
| `neg_vineyard` | negative | -5.770 | -8.256 | +2.486 |

## Variant: dedup

### Positive queries

| query | AUC | relevant ranks (of 174) | lowest relevant | highest irrelevant |
|---|---|---|---|---|
| `pos_uneasy` | 0.925 | Obsession (2025) #1, Companion (2025) #8, Weapons (2025) #11, Longlegs (2024) #21, Joker (2019) #25, Hereditary (2018) #46 | Hereditary (2018) -4.54 (0.011) | Nope (2022) -1.41 (0.197) |
| `pos_found_family` | 0.672 | Guardians of the Galaxy Vol. 2 (2017) #3, Deadpool 2 (2018) #29, Guardians of the Galaxy (2014) #51, Coco (2017) #94, KPop Demon Hunters (2025) #117 | KPop Demon Hunters (2025) -6.61 (0.001) | The Social Network (2010) -3.35 (0.034) |
| `pos_pleasant_surprise` | 0.655 | Crazy, Stupid, Love. (2011) #5, Scott Pilgrim vs. the World (2010) #22, Roofman (2025) #37, Flight (2012) #50, The Drama (2026) #127, Bugonia (2025) #132 | Bugonia (2025) -8.22 (0.000) | Superman (2025) -3.72 (0.024) |
| `pos_plot_twist` | 0.444 | Avengers: Infinity War (2018) #12, The Hateful Eight (2015) #85, Bugonia (2025) #130, Parasite (2019) #159 | Parasite (2019) -5.15 (0.006) | Black Panther (2018) -1.58 (0.171) |
| `pos_unhinged_comedy` | 0.544 | Deadpool (2016) #13, 21 Jump Street (2012) #51, Ted 2 (2015) #94, Ted (2012) #107, Pineapple Express (2008) #138 | Pineapple Express (2008) -5.55 (0.004) | Joker (2019) -0.66 (0.340) |
| `pos_visually_stunning` | 0.769 | Doctor Strange (2016) #1, Past Lives (2023) #21, The Odyssey (2026) #24, Project Hail Mary (2026) #32, Uncut Gems (2019) #49, Shang-Chi and the Legend of the Ten Rings (2021) #51, Interstellar (2014) #124 | Interstellar (2014) -5.44 (0.004) | Black Panther (2018) -2.00 (0.119) |
| `pos_war` | 0.842 | War for the Planet of the Apes (2017) #2, The Greatest Beer Run Ever (2022) #20, The Odyssey (2026) #65 | The Odyssey (2026) -6.68 (0.001) | Avengers: Infinity War (2018) -4.27 (0.014) |
| `pos_hopeful` | 0.774 | Chef (2014) #5, Coco (2017) #14, Soul (2020) #20, Everything Everywhere All at Once (2022) #125 | Everything Everywhere All at Once (2022) -5.54 (0.004) | Big Hero 6 (2014) -1.02 (0.266) |
| `pos_bittersweet_romance` | 0.987 | (500) Days of Summer (2009) #1, La La Land (2016) #3, Crazy, Stupid, Love. (2011) #4, Materialists (2025) #5, Past Lives (2023) #13 | Past Lives (2023) -3.00 (0.047) | Spider-Man: No Way Home (2021) -1.87 (0.134) |
| `pos_grief` | 0.870 | Life of Pi (2012) #1, Midsommar (2019) #19, Hereditary (2018) #22, Flight (2012) #56 | Flight (2012) -4.46 (0.011) | Arrival (2016) -2.93 (0.051) |
| `pos_coming_of_age` | 0.834 | CODA (2021) #1, A Silent Voice: The Movie (2016) #9, No Hard Feelings (2023) #51, Cha Cha Real Smooth (2022) #62 | Cha Cha Real Smooth (2022) -3.22 (0.038) | Ricky Stanicky (2024) -1.72 (0.152) |
| `pos_class_satire` | 0.720 | Parasite (2019) #1, Anora (2024) #58, American Psycho (2000) #66, Bugonia (2025) #74 | Bugonia (2025) -3.87 (0.020) | The Social Network (2010) -2.49 (0.077) |
| `pos_feel_good` | 0.646 | I Love You, Man (2009) #2, How High (2001) #4, Mike and Dave Need Wedding Dates (2016) #68, Wedding Crashers (2005) #76, Soul Plane (2004) #81, Despicable Me 4 (2024) #144 | Despicable Me 4 (2024) -5.75 (0.003) | The Switch (2010) -1.31 (0.213) |
| `pos_mind_bending` | 0.774 | Project Hail Mary (2026) #16, Interstellar (2014) #19, Arrival (2016) #61, Nope (2022) #71 | Nope (2022) -3.04 (0.045) | Doctor Strange in the Multiverse of Madness (2022) -1.32 (0.210) |
| `pos_obsessive_ambition` | 0.864 | Whiplash (2014) #2, King Richard (2021) #10, tick, tick... BOOM! (2021) #64 | tick, tick... BOOM! (2021) -5.84 (0.003) | The Social Network (2010) -2.84 (0.055) |

### Negative queries — top 3

| query | #1 | #2 | #3 |
|---|---|---|---|
| `neg_werewolves` | Ford v Ferrari (2019) -2.96 (0.049) | Green Book (2018) -3.23 (0.038) | The Social Network (2010) -3.48 (0.030) |
| `neg_silent_black_and_white` | The Dark Knight (2008) -1.52 (0.179) | Green Book (2018) -1.72 (0.151) | Spider-Man: Homecoming (2017) -1.82 (0.139) |
| `neg_bollywood` | Harold & Kumar Go to White Castle (2004) -2.47 (0.078) | Encanto (2021) -2.60 (0.069) | War for the Planet of the Apes (2017) -2.65 (0.066) |
| `neg_documentary` | Spider-Man: Homecoming (2017) -1.40 (0.197) | Ford v Ferrari (2019) -1.65 (0.162) | Life of Pi (2012) -1.69 (0.156) |
| `neg_pirates` | Ford v Ferrari (2019) -2.70 (0.063) | King Richard (2021) -2.77 (0.059) | Black Panther (2018) -2.83 (0.056) |
| `neg_time_loop` | La La Land (2016) -3.80 (0.022) | The Social Network (2010) -3.83 (0.021) | Fight Club (1999) -3.96 (0.019) |
| `neg_horse_racing` | Thor: Ragnarok (2017) -3.37 (0.033) | Black Panther (2018) -3.66 (0.025) | The Dark Knight (2008) -3.68 (0.025) |
| `neg_cold_war_espionage` | Ford v Ferrari (2019) -2.11 (0.109) | Deadpool (2016) -2.36 (0.086) | Iron Man (2008) -2.55 (0.073) |
| `neg_mountaineering` | The Social Network (2010) -3.59 (0.027) | Shang-Chi and the Legend of the Ten Rings (2021) -3.76 (0.023) | Spider-Man: Across the Spider-Verse (2023) -3.81 (0.022) |
| `neg_chess` | Fight Club (1999) -5.25 (0.005) | Spider-Man: Brand New Day (2026) -5.34 (0.005) | Ant-Man (2015) -5.36 (0.005) |
| `neg_vineyard` | Black Panther (2018) -5.52 (0.004) | Thor (2011) -5.57 (0.004) | Deadpool (2016) -5.73 (0.003) |
| `neg_ballet` | Whiplash (2014) -3.37 (0.033) | Black Panther (2018) -3.98 (0.018) | Shang-Chi and the Legend of the Ten Rings (2021) -4.17 (0.015) |

### Threshold sweep

Exact sweep over 4699 candidate thresholds; best points:

- **(i) all positives alive:** t=-5.155 logit (sigmoid 0.006; live-gate 0.5014): negatives gated 2/12, positives alive 15/15, film P=0.037 R=0.786 F1=0.071
- **(ii) best balanced:** t=-2.105 logit (sigmoid 0.109; live-gate 0.5271): negatives gated 10/12, positives alive 7/15, film P=0.204 R=0.143 F1=0.168
- **best film F1:** t=-2.152 logit (sigmoid 0.104; live-gate 0.5260): negatives gated 9/12, positives alive 7/15, film P=0.204 R=0.157 F1=0.177
- **Full separation: NO** — highest negative top score -1.404, lowest positive best-relevant score -5.136 (gap -3.732)

| t (logit) | sigmoid | live-gate | neg gated | pos alive | film P | film R | F1 |
|---|---|---|---|---|---|---|---|
| -10.986 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -10.476 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.966 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -9.456 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.027 | 1.000 | 0.053 |
| -8.946 | 0.000 | 0.5000 | 0/12 | 15/15 | 0.028 | 1.000 | 0.054 |
| -8.436 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.028 | 1.000 | 0.055 |
| -7.925 | 0.000 | 0.5001 | 0/12 | 15/15 | 0.028 | 0.971 | 0.054 |
| -7.415 | 0.001 | 0.5002 | 0/12 | 15/15 | 0.029 | 0.971 | 0.056 |
| -6.905 | 0.001 | 0.5003 | 0/12 | 15/15 | 0.030 | 0.971 | 0.059 |
| -6.395 | 0.002 | 0.5004 | 0/12 | 15/15 | 0.031 | 0.929 | 0.060 |
| -5.885 | 0.003 | 0.5007 | 0/12 | 15/15 | 0.033 | 0.886 | 0.064 |
| -5.375 | 0.005 | 0.5012 | 1/12 | 15/15 | 0.034 | 0.786 | 0.066 |
| -4.865 | 0.008 | 0.5019 | 2/12 | 14/15 | 0.040 | 0.729 | 0.075 |
| -4.355 | 0.013 | 0.5032 | 2/12 | 13/15 | 0.046 | 0.629 | 0.086 |
| -3.845 | 0.021 | 0.5052 | 2/12 | 12/15 | 0.055 | 0.514 | 0.099 |
| -3.335 | 0.034 | 0.5086 | 6/12 | 11/15 | 0.071 | 0.386 | 0.121 |
| -2.825 | 0.056 | 0.5140 | 7/12 | 10/15 | 0.098 | 0.243 | 0.139 |
| -2.315 | 0.090 | 0.5225 | 9/12 | 7/15 | 0.153 | 0.157 | 0.155 |
| -1.805 | 0.141 | 0.5353 | 10/12 | 6/15 | 0.200 | 0.086 | 0.120 |
| -1.295 | 0.215 | 0.5536 | 12/12 | 4/15 | 0.444 | 0.057 | 0.101 |
| -0.785 | 0.313 | 0.5777 | 12/12 | 2/15 | 0.667 | 0.029 | 0.055 |
| -0.275 | 0.432 | 0.6063 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +0.236 | 0.559 | 0.6361 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +0.746 | 0.678 | 0.6633 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |
| +1.256 | 0.778 | 0.6853 | 12/12 | 1/15 | 1.000 | 0.014 | 0.028 |

### Top score vs. margin (top − median over all films)

- Top score: AUC(pos vs neg queries) 0.806, separates=False, gap -2.870
- Margin: AUC(pos vs neg queries) 0.883, separates=False, gap -0.593

| query | type | top | median | margin |
|---|---|---|---|---|
| `pos_uneasy` | positive | +1.256 | -4.952 | +6.208 |
| `pos_visually_stunning` | positive | -0.390 | -4.668 | +4.278 |
| `pos_unhinged_comedy` | positive | -0.662 | -4.581 | +3.919 |
| `pos_coming_of_age` | positive | -0.848 | -3.612 | +2.764 |
| `pos_hopeful` | positive | -1.016 | -4.836 | +3.820 |
| `pos_class_satire` | positive | -1.150 | -4.006 | +2.857 |
| `pos_feel_good` | positive | -1.307 | -4.523 | +3.217 |
| `pos_mind_bending` | positive | -1.324 | -3.326 | +2.003 |
| `neg_documentary` | negative | -1.404 | -3.132 | +1.728 |
| `neg_silent_black_and_white` | negative | -1.521 | -3.523 | +2.002 |
| `pos_bittersweet_romance` | positive | -1.549 | -4.371 | +2.822 |
| `pos_plot_twist` | positive | -1.577 | -3.873 | +2.295 |
| `neg_cold_war_espionage` | negative | -2.106 | -4.339 | +2.233 |
| `neg_bollywood` | negative | -2.469 | -4.589 | +2.120 |
| `pos_grief` | positive | -2.577 | -5.006 | +2.429 |
| `neg_pirates` | negative | -2.695 | -5.161 | +2.465 |
| `pos_obsessive_ambition` | positive | -2.841 | -6.193 | +3.352 |
| `neg_werewolves` | negative | -2.957 | -5.552 | +2.595 |
| `pos_pleasant_surprise` | positive | -3.075 | -7.446 | +4.370 |
| `pos_found_family` | positive | -3.346 | -5.992 | +2.646 |
| `neg_ballet` | negative | -3.368 | -5.813 | +2.445 |
| `neg_horse_racing` | negative | -3.375 | -5.724 | +2.349 |
| `neg_mountaineering` | negative | -3.586 | -5.962 | +2.376 |
| `neg_time_loop` | negative | -3.799 | -6.121 | +2.322 |
| `pos_war` | positive | -4.274 | -7.414 | +3.140 |
| `neg_chess` | negative | -5.254 | -7.274 | +2.020 |
| `neg_vineyard` | negative | -5.520 | -8.083 | +2.562 |

## Live pipeline view (current variant)

Fused pool = RRF of top 20 vector + top 20 BM25; gate sees reranked top 8. Films missing from the pool are a retrieval-recall problem no threshold can fix.

- In fused pool: 75.7% of relevant films overall (mean per query 79.6%)
- In reranked top 8: 45.7% overall (mean per query 48.3%)
- Queries with no relevant film in pool: none
- Queries with no relevant film in top 8: ['pos_unhinged_comedy']

| query | relevant | pool size | in pool | in top 8 | top-8 ceiling | missing from pool |
|---|---|---|---|---|---|---|
| `pos_uneasy` | 6 | 27 | 5 (83%) | 3 (50%) | 100% | Joker |
| `pos_found_family` | 5 | 31 | 3 (60%) | 2 (40%) | 100% | Guardians of the Galaxy, KPop Demon Hunters |
| `pos_pleasant_surprise` | 6 | 36 | 2 (33%) | 1 (17%) | 100% | Bugonia, Flight, Scott Pilgrim vs. the World, The Drama |
| `pos_plot_twist` | 4 | 27 | 3 (75%) | 1 (25%) | 100% | Parasite |
| `pos_unhinged_comedy` | 5 | 34 | 3 (60%) | 0 (0%) | 100% | Deadpool, Ted |
| `pos_visually_stunning` | 7 | 21 | 4 (57%) | 3 (43%) | 100% | Interstellar, Past Lives, Shang-Chi and the Legend of the Ten Rings |
| `pos_war` | 3 | 32 | 3 (100%) | 2 (67%) | 100% | — |
| `pos_hopeful` | 4 | 34 | 4 (100%) | 1 (25%) | 100% | — |
| `pos_bittersweet_romance` | 5 | 28 | 5 (100%) | 5 (100%) | 100% | — |
| `pos_grief` | 4 | 34 | 4 (100%) | 3 (75%) | 100% | — |
| `pos_coming_of_age` | 4 | 32 | 4 (100%) | 2 (50%) | 100% | — |
| `pos_class_satire` | 4 | 30 | 4 (100%) | 2 (50%) | 100% | — |
| `pos_feel_good` | 6 | 33 | 3 (50%) | 2 (33%) | 100% | Despicable Me 4, Mike and Dave Need Wedding Dates, Soul Plane |
| `pos_mind_bending` | 4 | 23 | 3 (75%) | 2 (50%) | 100% | Arrival |
| `pos_obsessive_ambition` | 3 | 28 | 3 (100%) | 3 (100%) | 100% | — |
