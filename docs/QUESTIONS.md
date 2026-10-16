# Questions for the supervisor, with provisional decisions

Written 16 October 2026, at the end of weeks 5 and 6. The work did not wait
for answers. Each question below has the decision taken so the next step
could run, the reason, and what it would cost to undo. Undoing is never more
than a re-run: every decision is a constant or a switch in the code, and the
caches are keyed on the data, so nothing has to be rebuilt by hand.

Questions 1 to 19 concern work that is done (weeks 3 to 6). Questions 20 to
32 concern weeks 7 to 16; their decisions are plans, and an answer before the
week in question is enough. This file replaces section 7 of
`docs/MEASURE.md`. The most important are 7 and 9.

Cost to undo: **none** (a reporting choice), **cheap** (minutes, from cached
results), **moderate** (hours of computing), **expensive** (redo a week).

## Summary

| # | question | provisional decision | to undo |
|---:|---|---|---|
| 1 | Is the development/benchmark split acceptable? | used as proposed: 5 development, 15 benchmark datasets | moderate |
| 2 | Resampling null instead of the proposal's permutation null? | half-reference resampling per point; permutation for the chance level only | moderate |
| 3 | Budget: top 5% or the number of labelled anomalies? | top 5%; the anomaly count reported as a second analysis | cheap |
| 4 | Shuttle as a control? | kept, as the case where local and global dimension differ | none |
| 5 | Lattice data (integer features)? | kept and reported; LOF not used on it; a sensitivity analysis without heavily tied datasets | cheap |
| 6 | Ranks or normalised scores? | ranks | expensive |
| 7 | Adopt the multi-scale reading (candidate version 2)? | not adopted, by the rule fixed beforehand; run beside version 1 on the benchmark | cheap |
| 8 | If it is adopted, which scales? | k = 20, 50 and 100 | moderate |
| 9 | The verdicts depend on the detector. Which one? | kNN, named in every statement; LOF on the continuous benchmark datasets in week 7 | cheap to add |
| 10 | Correct the per-point calls for multiple testing? | no; the false-call rate is measured instead, and p-values are kept | cheap |
| 11 | When is the measure redundant with continuity and trustworthiness? | per-point AUC above 0.9 on most runs; fixed before week 11 | none |
| 12 | Which projection methods, at which settings? | PCA, random projection, UMAP, t-SNE, PaCMAP, TriMap 1.1.4, library defaults | moderate |
| 13 | How many seeds? | 3 in week 6 (5 on the development set); 5 from week 8 | moderate |
| 14 | Cap of 20,000 points? | yes; backdoor at full size once, after the freeze | moderate |
| 15 | Which statistics for the method comparison? | dataset as the unit; Friedman, Nemenyi, Wilcoxon with Holm | cheap |
| 16 | Which outcome is primary? | masking rate at the top 5%; phantom rate beside it, never combined | none |
| 17 | Random projection in the comparison? | yes, as the uninformed floor | cheap |
| 18 | Run the multi-scale candidate on the benchmark too? | yes, as a secondary analysis declared before the run | cheap |
| 19 | One random projection, or several? | three seeds, with its per-point calls read as seed-dependent | none |
| 20 | Which target dimensions in week 7? | 2, 3, 5 and 10; t-SNE at 2 and 3 only | additive |
| 21 | Which hyperparameters in week 8? | one neighbourhood-size parameter per method at three values, plus UMAP's min_dist | additive |
| 22 | How to summarise seed variance? | spread of the rates, Jaccard of the masked sets, and how often each point gets the same call | none |
| 23 | How to test Question 3? | within-dataset model of masked against retained points, LID against density | moderate |
| 24 | Which existing metrics in week 11? | trustworthiness, continuity, MRRE, per point and aggregate, plus the Schubert et al. weighted correlation | additive |
| 25 | What does "frozen" mean in week 12? | a tagged commit; any later change means the whole final run again | none |
| 26 | Full-size backdoor as the security case study? | yes, once, with the frozen definition | moderate |
| 27 | Clustered anomalies (a denial-of-service burst)? | outside the definition; described with a per-dataset statistic | none |
| 28 | Categorical features? | taken as ADBench encodes them, standardised like the rest | expensive |
| 29 | Visual salience: a perceptual study? | no; the gap is stated as a limitation | none |
| 30 | A clump at the edge of a picture: outlying or not? | part of question 29; both readings reported | none |
| 31 | Title? | propose "Outlier faithfulness of low-dimensional projections" | none |
| 32 | Results that do not reproduce bit for bit across machines? | the numbers the text cites are tracked in git | none |

## Weeks 3 and 4, still open

**1. The split.** `results/splits.json` is marked "proposed; pending
supervisor confirmation": shuttle, wbc, ionosphere, fault and magic for
development, fifteen datasets for the benchmark, backdoor among them as the
security case.
- *Decision:* used as proposed. Week 6 ran all fifteen.
- *Why:* the screen applied criteria stated before any measure existed, and
  nothing in weeks 4 to 6 argues against the split.
- *To undo:* moderate. A benchmark dataset costs two to five minutes of
  computing at 20,000 points and under a minute below that, once its
  projections exist; dropping one is free, adding one is a run of that
  dataset. A different
  development set would leave the week-4 and week-5 decisions resting on
  other evidence.

**2. The null model.** The proposal named a permutation null after scDEED.
The measure uses a resampling null per point: 100 re-rankings of the
original space, each against a random half of the data. Permutation is kept
where it fits, for the chance level of the shortlist overlap.
- *Decision:* keep.
- *Why:* the scDEED-style null calls almost nothing; a point has to fall 7 to
  63 times deeper in the ranking to count (MEASURE.md 5.5). The resampling
  null is calibrated: comparing two re-measurements of the same data calls
  2 to 3% of crossings at the 5% level (MEASURE.md 4 and 9.3).
- *To undo:* moderate. The null is computed once per dataset and cached.

**3. The budget.** The top 5% of points, which needs no labels, rather than
the number of labelled anomalies (the P@n convention).
- *Decision:* top 5%. The anomaly-count budget is computed on every
  benchmark run and compared in the same way (BENCHMARK.md).
- *Why:* the measure asks what the projection preserves of what the detector
  sees, which does not involve labels. Three development datasets are about
  35% anomalies, so the anomaly count would call a third of the data "top".
- *To undo:* cheap; both are in `results/week6_benchmark.json`.

**4. Shuttle.** Expected to be a control, but PCA to 2D hides 57% of its top
5%, because shuttle spans six linear dimensions while its local dimension is
about two (MEASURE.md 5.1).
- *Decision:* kept in the development set as the example of local and global
  dimension differing.
- *To undo:* none.

**5. Lattice data.** Every feature of shuttle and wbc is a whole number. Two
consequences: tied scores, and pictures that shatter into islands of
near-identical points (MEASURE.md 5.9).
- *Decision:* keep both. Record for every benchmark dataset how grid-like it
  is (integer features, distinct values per feature, distinct scores,
  islands per picture). Do not use LOF on lattice data, where its result
  depends on how tied distances are ordered. Repeat the method comparison
  without the datasets whose original-space scores are less than half
  distinct; this rule was fixed before the benchmark run.
- *Why:* replacing the datasets would hide a property that network data
  shares (ports, flags, counts).
- *To undo:* cheap.

**6. Ranks or normalised scores.** Ranks treat a point far ahead of the next
and a point barely ahead alike; a score-based displacement would not.
- *Decision:* ranks.
- *Why:* scores do not carry over between spaces of different dimension
  (MEASURE.md 9.9). The kNN distance grows like density to the power -1/d,
  so under PCA the median score ratio of the top 5% grows from 5.1 at nine
  dimensions to 20.1 at two on wbc, and every point both spaces agree on
  would show one to two bits of displacement from the change of dimension
  alone. Gaussian scaling (Kriegel et al. 2011) saturates at the top on
  three of five datasets and drifts on a fourth.
- *To undo:* expensive; the whole measure is rank-based.

## Week 5

**7. The multi-scale candidate.** UMAP and PaCMAP gather isolated outliers
into tight clumps, which look dense at k = 20 and apart at k = 50. The
candidate (version 2) reads both spaces at k = 20, 50 and 100 and keeps each
point's most outlying reading (MEASURE.md 9.2). Should it replace version 1?
- *Decision:* not adopted, because it missed two of the four criteria fixed
  before the results were read (MEASURE.md 9.1 and 9.11), both narrowly. It
  moved one linear-method verdict by 0.101 against a bound of 0.10 (shuttle
  under PCA), and its masked sets agree across seeds slightly less often
  (mean Jaccard 0.741 against 0.797, a drop of 0.056 against a tolerance of
  0.05). It passes every known-answer check and is far less sensitive to its
  own scales than version 1 is to k (89% of runs within 0.10 against 71%).
  The benchmark runs both readings, version 1 as the primary analysis.
- *For the supervisor to weigh:* the candidate reads a clump made by the
  projection the way a person probably would (version 1 calls all 40
  outliers of a far-away clump hidden, the candidate none), and on wbc the
  neighbour-based methods' masking falls from 77-95% to 17-32% under it.
  The bound in the first criterion was set after the shuttle line had been
  printed, which the record states. The second criterion cannot tell
  measurement noise from real differences between seeds' pictures: a clump
  that sits apart under one seed and beside the bulk under another should be
  read differently. Both arguments came after the results, which is why the
  rule was followed rather than set aside.
- *To undo:* cheap. The benchmark results exist for both readings, and
  show how much the method comparison depends on the reading
  (BENCHMARK.md 4.3 and 4.5): under the candidate, t-SNE and TriMap move
  level with PCA, and the pictures that rank the anomalies below the normal
  points fall from 24 dataset-method pairs to 3. Those results came after
  the decision and did not change it.

**8. The scales,** if the candidate is adopted: 20 (version 1's k), 50, and
100 (the neighbour graph's stored depth).
- *Decision:* these three.
- *Why:* three other sets, {20, 50}, {20, 100} and {10, 20, 50, 100}, change
  the masking rate by a median of 0.010 (MEASURE.md 9.5).
- *To undo:* moderate.

**9. The detector.** The proposal lists distance-based, density-based,
isolation-based and dimensionality-aware detectors. The definition takes any
of them without change, and LOF stays close to calibrated through the same
code. But the verdicts depend on the detector: on the three continuous
development datasets, LOF orders the six methods differently from kNN (rank
correlation +0.85, -0.43 and +0.26). On magic, kNN finds the linear methods
keep most of the top 5% and the neighbour-based methods lose most of it; LOF
finds every method loses about the same (MEASURE.md 9.6).
- *Decision:* kNN is the detector of the benchmark, and every statement
  about methods names it. LOF runs in week 7 on the benchmark datasets whose
  original-space kNN scores are at least 95% distinct (all but celeba), on
  the projections already cached, so the comparison can be read for two
  detectors before week 8. The 95% rule separates the development set's
  lattice datasets (42% and 81%), where LOF depends on how ties are ordered,
  from its continuous ones (99 to 100%). Isolation forest has no neighbour graph
  and its own randomness, so its noise null would be its seed variance; it
  is added in week 12 if time allows. A dimensionality-aware detector
  (Anderberg et al. 2024) belongs to Question 3 and is tried in week 10.
- *Why:* one detector keeps the week-6 comparison about the projections; the
  disagreement is a finding to report, not a reason to pick the detector
  that tells the tidier story.
- *To undo:* adding a detector is additive.

**10. Multiple testing.** Each point gets its own call at the 5% level, with
no correction across points.
- *Decision:* no correction.
- *Why:* the rates are compared between methods at the same level, and the
  calibration check measures the false-call rate directly (2 to 3% for kNN).
  Per-point p-values are stored for every benchmark run, so a
  Benjamini-Hochberg version can be computed without re-running.
- *To undo:* cheap.

**11. Redundancy.** Fixed before week 11: if per-point continuity separated
masked from retained points with an AUC above 0.9 on most runs (and
trustworthiness phantoms from retained points), the per-point calls would add
little. Week 5's first look (MEASURE.md 9.8): for linear projections
continuity finds the masked points well (median AUC 0.90); for the
neighbour-based methods it does not, and under version 1 it points the wrong
way (median 0.24): the points they hide keep their original neighbours
better than the points they keep.
- *Decision:* keep the criterion; week 11 applies it to every metric, on the
  benchmark.
- *To undo:* none.

## Week 6

**12. Methods and settings.** PCA (exact SVD), a Gaussian random projection,
UMAP, t-SNE (openTSNE), PaCMAP and TriMap, at library defaults.
- *Decision:* as listed. TriMap is pinned to 1.1.4, because 1.2.0 requires
  torch and its CUDA stack. 1.1.4 reproduces only with its random generator
  seeded inside numba and on one thread, which the code does; it needs
  annoy, which compiles from source on Windows (README).
- *To undo:* moderate (re-embedding; t-SNE is the slow one, about a minute per
  embedding at 20,000 points).

**13. Seeds.** 3 per stochastic method on the benchmark (0, 1, 2), 5 on the
development set.
- *Decision:* 3 in week 6; week 8 adds seeds 3 and 4.
- *To undo:* moderate.

**14. The cap.** Every dataset is capped at 20,000 points (stratified by
label, seed 0), after duplicate removal.
- *Decision:* keep the cap for the benchmark; analyse backdoor once at full
  size (question 26).
- *Why:* the noise null grows with n squared. At 20,000 points the whole
  benchmark, both readings, ran in 18 minutes on two cores, after about 45
  minutes of projections.
- *To undo:* moderate.

**15. Statistics.** The unit is the dataset: the mean over seeds per dataset
and method. Friedman's test with the Iman-Davenport correction, Nemenyi's
critical difference (Demsar 2006), and pairwise Wilcoxon signed-rank tests
with Holm's correction, which Benavoli, Corani and Mangili (2016) recommend
because mean-rank post-hoc tests depend on which other methods are compared.
Differences below 0.02 are counted as within the noise floor (September).
- *Decision:* as stated.
- *To undo:* cheap.

**16. Primary outcome.** The masking rate at the top 5%, version 1: the
failure that costs detections. The phantom rate is reported beside it and
never combined with it, as the proposal asks. AUC retained is the proposal's
anchor.
- *Decision:* as stated.
- *To undo:* none.

**17. Random projection.** Included in the comparison as the uninformed
floor, so "better than chance" has a concrete meaning.
- *Decision:* included in every test.
- *To undo:* cheap.

**18. The candidate on the benchmark.** Running two readings on benchmark
data invites choosing between them on benchmark results.
- *Decision:* run it, as a secondary analysis declared before any benchmark
  score existed, with version 1 fixed as primary at the same time.
- *Why:* question 7 is open, and the supervisor's answer should not need a
  re-run.
- *To undo:* cheap.

**19. Random projections.** Different random maps hide different points: over
five seeds, only 16% of the original top 5% gets the same call every time
(MEASURE.md 9.7).
- *Decision:* three seeds, as for the other methods; its rates are averaged,
  and its per-point calls are not read one seed at a time.
- *To undo:* none.

## Weeks 7 to 12

**20. Week 7, more than two dimensions.**
- *Decision:* d = 2, 3, 5 and 10 (only d < D), for PCA, random projection,
  UMAP, PaCMAP and TriMap; t-SNE at 2 and 3 only, with Barnes-Hut at 3
  (openTSNE's interpolation scheme stops at two dimensions, and Barnes-Hut
  beyond three is unsupported). Seeds 0 to 2. The noise null is a property of
  the original space, so it is reused. LOF as in question 9.
- *Why:* the proposal names two and three dimensions for viewing and "more
  than three" for analytics. Question 2 asks for at least one method across
  dimensions; PCA and UMAP give a linear and a neighbour-based one.
- *To undo:* additive.

**21. Week 8, hyperparameters.**
- *Decision:* one neighbourhood-size parameter per method at three values,
  the default in the middle: UMAP n_neighbors 5, 15, 50; t-SNE perplexity 5,
  30, 100; PaCMAP n_neighbors 5, 10, 30; TriMap n_inliers 6, 12, 24. Plus
  UMAP min_dist 0.0, 0.1, 0.5, because how tightly UMAP packs points is what
  makes the clumps of question 7. 2D, seeds 0 to 4.
- *To undo:* additive.

**22. Week 8, seed variance.**
- *Decision:* report the spread of the rates over seeds, the mean Jaccard
  overlap of the masked sets, and the share of the original top 5% masked
  under every seed, under some and under none (as in
  `figures/week5_seeds.png`).
- *To undo:* none.

**23. Week 10, Question 3.**
- *Decision:* within each dataset and method, among the points of the
  original top 5%, model masked (1) against retained (0) on the point's LID
  (MLE, k = 20) and its local density (log k-distance), by logistic
  regression. The hypothesis predicts a positive LID effect on masking at
  fixed density. Report the sign and size of the LID coefficient per dataset
  and method, and a sign test across datasets. LID is already stored per
  point for every benchmark dataset.
- *Why:* the proposal tests Question 3 within datasets, since both predictor
  and outcome are per point.
- *To undo:* moderate.

**24. Week 11, existing metrics.**
- *Decision:* trustworthiness and continuity (k = 20), and MRRE in both
  directions, per point and aggregate; the Schubert et al. weighted rank
  correlation; Spearman. ZADU's cluster-level measures (steadiness and
  cohesiveness) only in aggregate. Criterion as in question 11.
- *To undo:* additive.

**25. Week 12, the freeze.**
- *Decision:* the definition, its constants (scales, budget, alpha, B), the
  dataset list and the seeds are fixed in a tagged commit. The final run is
  computed from empty caches with that commit.
- *To undo:* none.

## Beyond week 12

**26. Backdoor at full size.** 87,020 points after duplicate removal.
- *Decision:* once, with the frozen definition, as the security case study.
  The null costs about 19 times the capped one.
- *To undo:* moderate.

**27. Clustered anomalies.** A denial-of-service burst is thousands of
near-identical records: a cluster, not an outlier. The measure inherits the
detector's notion of outlier, so a group much larger than k is not outlying
to it in either space.
- *Decision:* outside the definition. Each benchmark dataset records the
  median share of an anomaly's 20 nearest neighbours that are anomalies too,
  which describes how clustered its anomalies are (BENCHMARK.md).
- *To undo:* none.

**28. Categorical features.** ADBench gives every dataset as numbers (one-hot
or codes), which are standardised like the rest.
- *Decision:* take them as given.
- *To undo:* expensive (a different distance for mixed data changes every
  neighbour graph).

**29. Visual salience.** "Outlying in a projection" is a detector's reading of
the projected coordinates, not what a person sees.
- *Decision:* no perceptual study; the gap is a stated limitation.
- *To undo:* none.

**30. Clumps.** Version 1 calls a tight clump at the edge of a UMAP plot
inlying; the multi-scale candidate calls it outlying if it is apart at
k = 100. Which is right is a question about what a person sees.
- *Decision:* as question 29; both readings are reported (question 7).

**31. Title.** "Outlier detection on low-dimensional projections" describes
detection; the thesis builds a faithfulness measure.
- *Decision:* propose "Outlier faithfulness of low-dimensional projections".
- *To undo:* none.

**32. Reproducibility across machines.** UMAP, t-SNE and TriMap do not
reproduce bit for bit across machines.
- *Decision:* the result files the text cites are tracked in git (README).
- *To undo:* none.
