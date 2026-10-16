# The benchmark in two dimensions (week 6)

Week 6 of the plan: the measure, version 1 (`docs/MEASURE.md`), run on the
fifteen benchmark datasets at two dimensions, with the multi-scale candidate
of week 5 beside it. Code: `src/benchmark.py` (stages `embed`, `run`,
`report`, `figures`). Numbers below come from `results/week6_benchmark.json`
and `results/week6_stats.json`; figures are `figures/week6_*.png`. The open
questions, each with the decision taken, are in `docs/QUESTIONS.md` (12 to 19
concern this week).

## 1. Protocol

Fixed before any outlier score was computed on benchmark data. The
projections were computed first, while week 5 was still testing the measure;
they do not depend on it. Week 5 then kept version 1 (MEASURE.md 9.11), and
version 1 was fixed as the primary analysis. The multi-scale candidate runs
beside it as a secondary analysis, declared at the same time, so the open
question 7 can be answered without re-running anything.

| | |
|---|---|
| datasets | the 15 benchmark datasets of `results/splits.json`, duplicates removed, each capped at 20,000 points (stratified by label, seed 0), standardised |
| methods | PCA (exact SVD); a Gaussian random projection, the uninformed floor; UMAP, t-SNE (openTSNE), PaCMAP and TriMap 1.1.4, at library defaults |
| seeds | 0, 1 and 2 for every stochastic method; PCA once |
| dimension | 2 |
| measure | version 1: kNN at k = 20 in both spaces, top 5% budget, noise null of 100 half-reference replicates, alpha = 0.05 |
| primary outcome | the masking rate at the top 5% |
| also reported | phantom rate, adjusted overlap, AUC in both spaces (the proposal's anchor) and what the failures cost in labelled anomalies |
| secondary analyses | the masking rate at the anomaly-count budget (the P@n convention); the comparison without datasets whose original-space scores are less than half distinct (heavy ties); the multi-scale candidate (k = 20, 50 and 100) |
| unit of comparison | the dataset: one value per dataset and method, the mean over seeds |
| tests | Friedman's test with the Iman-Davenport F; Nemenyi's critical difference (Demsar 2006); pairwise Wilcoxon signed-rank tests with Holm's correction (Benavoli, Corani and Mangili 2016), with differences below 0.02 counted as within the noise floor |
| descriptors | per dataset: integer-valued features, distinct values per feature, distinct scores, and how clustered the anomalies are (median share of an anomaly's 20 nearest neighbours that are anomalies); per picture: islands of the 10-NN graph, as in week 4 |

Per-point output (categories, displacement and p-value under both readings,
original-space ranks, LID) is saved for every run in
`results/week6_points_<dataset>.npz`, so weeks 8, 10 and 11 can reuse it.

## 2. Expectations, written before the results were read

Written while the run was in progress. One line of its log had been seen
(backdoor, TriMap, seed 2); nothing else. Each expectation comes from the
development set (MEASURE.md 5 and 9).

| | expectation | from |
|---|---|---|
| B1 | under version 1 the neighbour-based methods mask more of the top 5% than both linear methods on most datasets, and Friedman's test rejects equal masking rates | development set: 4 of 5 datasets |
| B2 | the random projection masks less than PCA on about half the datasets, not on all | development set: less on shuttle and ionosphere, more on wbc |
| B3 | at 2D the phantom rate stays within 0.05 of the masking rate for most runs | most 2D crossings are far beyond noise (MEASURE.md 6) |
| B4 | AUC retained separates the methods less clearly than the masking rate (a smaller Friedman statistic) | aggregate summaries miss the shortlist (MEASURE.md 5.2) |
| B5 | the anomaly-count budget and the comparison without heavily tied datasets give much the same order of methods (rank correlation of the average ranks at least 0.8) | development set, budget sweep (5.4, 9.10) |
| B6 | the multi-scale candidate lowers the neighbour-based methods' masking rates on some datasets, moves the linear methods' little, and keeps the neighbour-based methods above the linear ones on average | development set (9.4) |
| B7 | on datasets with heavily tied scores the neighbour-based pictures break into many islands, as shuttle's did | MEASURE.md 5.9 |

## 3. The datasets as run

| dataset | n | D | anomalies | integer features | distinct scores | AUC | anomalies' neighbours that are anomalies | islands, neighbour-based |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cover | 20,000 | 10 | 1.0% | 10 of 10 | 99% | 0.91 | 0.85 | 41 |
| pageblocks | 5,393 | 10 | 9.5% | 6 of 10 | 99% | 0.91 | 0.85 | 36 |
| vowels | 1,452 | 12 | 3.2% | 0 of 12 | 100% | 0.96 | 0.25 | 5 |
| pendigits | 6,870 | 16 | 2.3% | 0 of 16 | 99% | 0.81 | 1.00 | 52 |
| cardio | 1,822 | 21 | 9.6% | 0 of 21 | 99% | 0.79 | 0.90 | 18 |
| waveform | 3,443 | 21 | 2.9% | 0 of 21 | 99% | 0.74 | 0.15 | 4 |
| fraud | 20,000 | 29 | 0.2% | 0 of 29 | 99% | 0.97 | 0.55 | 202 |
| wdbc | 367 | 30 | 2.7% | 0 of 30 | 99% | 0.98 | 0.38 | 1 |
| letter | 1,598 | 32 | 6.3% | 32 of 32 | 98% | 0.84 | 0.10 | 30 |
| satimage2 | 5,801 | 36 | 1.2% | 36 of 36 | 99% | 0.98 | 1.00 | 15 |
| celeba | 20,000 | 39 | 2.6% | 39 of 39 | 59% | 0.74 | 0.20 | 90 |
| campaign | 20,000 | 62 | 11.3% | 52 of 62 | 99% | 0.74 | 0.25 | 429 |
| mnist | 7,603 | 100 | 9.2% | 10 of 100 | 99% | 0.86 | 0.75 | 25 |
| musk | 3,062 | 166 | 3.2% | 166 of 166 | 98% | 0.94 | 1.00 | 98 |
| backdoor | 20,000 | 196 | 2.2% | 158 of 196 | 99% | 0.91 | 0.15 | 251 |

"Distinct scores" is the share of distinct kNN scores in the original space,
"AUC" the detector's AUC there, "anomalies' neighbours" the median share of
an anomaly's 20 nearest neighbours that are anomalies too (1.00: the
anomalies form their own group), and "islands" the median number of islands
in the neighbour-based pictures (a linear picture has a median of 4 or 5).

- **No dataset has heavily tied scores.** Five have only integer features,
  but after standardisation their scores are 98 to 99% distinct, except
  celeba (39 binary attributes, 59%). Nothing falls below the threshold of
  50% fixed before the run, so the sensitivity analysis had nothing to
  remove.
- **The neighbour-based pictures shatter on the large datasets with many
  discrete fields:** campaign (t-SNE: 1,069 islands at seed 0), backdoor and
  fraud. Shuttle's pictures did the same in week 4.
- **On seven datasets the anomalies form their own group** (median share
  0.75 or more: cover, pageblocks, pendigits, cardio, satimage2, mnist,
  musk). These are
  the clustered anomalies of `docs/QUESTIONS.md` 27, present in the data, not
  made by the projection.

## 4. Results (version 1)

### 4.1 Masking rate per dataset and method

Mean over seeds, top 5%; the lowest in each row in bold
(`figures/week6_rates.png` has the phantom rates beside them).

| dataset | PCA | random | UMAP | t-SNE | PaCMAP | TriMap |
|---|---:|---:|---:|---:|---:|---:|
| cover | **0.58** | 0.66 | 0.85 | 0.79 | 0.82 | 0.77 |
| pageblocks | 0.34 | **0.31** | 0.85 | 0.87 | 0.84 | 0.55 |
| vowels | 0.79 | 0.79 | 0.72 | **0.54** | 0.82 | 0.75 |
| pendigits | **0.68** | 0.70 | 0.91 | 0.81 | 0.88 | 0.71 |
| cardio | 0.49 | **0.46** | 0.83 | 0.78 | 0.99 | 0.93 |
| waveform | 0.72 | 0.80 | 0.85 | 0.78 | **0.69** | 0.72 |
| fraud | **0.43** | 0.54 | 0.89 | 0.87 | 0.89 | 0.93 |
| wdbc | **0.05** | 0.37 | 0.60 | 0.58 | 0.44 | 0.49 |
| letter | 0.68 | 0.78 | 0.84 | **0.53** | 0.87 | 0.86 |
| satimage2 | 0.56 | **0.54** | 0.70 | 0.57 | 0.68 | 0.60 |
| celeba | **0.64** | 0.81 | 0.76 | 0.76 | 0.75 | 0.65 |
| campaign | **0.57** | 0.67 | 0.84 | 0.75 | 0.91 | 0.83 |
| mnist | 0.83 | 0.62 | 0.78 | 0.77 | 0.81 | **0.41** |
| musk | 0.58 | 0.70 | 0.64 | 0.52 | 0.54 | **0.47** |
| backdoor | 0.54 | **0.33** | 0.58 | 0.62 | 0.81 | 0.64 |
| mean | 0.57 | 0.60 | 0.78 | 0.70 | 0.78 | 0.69 |
| average rank | 2.27 | 2.67 | 4.73 | 3.33 | 4.60 | 3.40 |

- **Every method hides most of the top 5% on most datasets.** PCA and the
  random projection hide more than half on 11 of 15, UMAP and t-SNE on all
  15. The best case is PCA on wdbc (5%), as it was on wbc in week 4.
- **The neighbour-based methods hide more.** UMAP hides 58 to 91% on every
  dataset, PaCMAP 44 to 99%. t-SNE and TriMap sit between them and the linear
  methods, and each is the best method on two datasets (t-SNE on vowels and
  letter, TriMap on mnist and musk).
- **No method is best everywhere.** PCA hides least on six datasets, the
  random projection on four (backdoor among them), t-SNE and TriMap on two
  each, PaCMAP on one, UMAP on none.
- **The phantom rate follows the masking rate** within 0.05 in 90% of runs
  (largest gap 0.14, musk under t-SNE), as expected at 2D, where nearly every
  crossing is far beyond noise: the masking rate is 0.020 below the plain
  exit rate on average, as in week 4.
- **Some shortlists are no better than chance:** PaCMAP and TriMap on cardio
  under every seed (adjusted overlap -0.05 to +0.03, permutation p above
  0.05), and some seeds elsewhere (UMAP and PaCMAP on pendigits, PaCMAP on
  letter, TriMap on fraud).

### 4.2 Comparing the methods

| | Friedman chi-squared | p | Iman-Davenport F | p |
|---|---:|---:|---:|---:|
| masking rate | 21.4 | 0.0007 | 5.6 | 0.0002 |
| phantom rate | 22.6 | 0.0004 | 6.0 | 0.0001 |
| adjusted overlap | 20.5 | 0.001 | 5.3 | 0.0004 |
| AUC retained | 33.6 | 0.000003 | 11.4 | 0.00000005 |

Six methods, fifteen datasets: Nemenyi's critical difference is 1.95. Pairs
that differ in a Wilcoxon signed-rank test after Holm's correction, for the
masking rate:

| better | worse | datasets where better | Holm p | median difference |
|---|---|---:|---:|---:|
| PCA | UMAP | 13 of 15 | 0.013 | 0.17 |
| PCA | PaCMAP | 12 of 15 | 0.028 | 0.20 |
| random | UMAP | 12 of 15 | 0.034 | 0.16 |
| TriMap | PaCMAP | 12 of 15 | 0.034 | 0.07 |
| t-SNE | UMAP | 13 of 15 | 0.037 | 0.06 |

Nemenyi's test finds the first three of these pairs and no others. The
phantom rate and the adjusted overlap give
the same two groups (`figures/week6_ranks.png`): **UMAP and PaCMAP hide
significantly more than PCA, and UMAP more than the random projection; t-SNE
and TriMap cannot be told from the linear methods with fifteen datasets.**
This is a statement about the kNN detector at k = 20 (MEASURE.md 9.6).

### 4.3 The anchor: detection in the picture

The proposal's anchor runs the same detector on the picture and compares
AUC with the original space (mean original AUC 0.87).

| | PCA | random | UMAP | t-SNE | PaCMAP | TriMap |
|---|---:|---:|---:|---:|---:|---:|
| AUC in the picture, mean | 0.83 | 0.73 | 0.51 | 0.55 | 0.53 | 0.73 |
| datasets with AUC below 0.5 | 0 | 0 | 9 | 8 | 7 | 0 |
| same, multi-scale reading | 0 | 0 | 1 | 0 | 2 | 0 |
| AUC in the picture, multi-scale reading, mean | 0.84 | 0.73 | 0.74 | 0.76 | 0.71 | 0.80 |

- **Read at k = 20, UMAP, t-SNE and PaCMAP pictures rank the anomalies
  *below* the normal points on about half the datasets.** The kNN distance in
  those pictures is shorter for anomalies than for the rest: they sit in
  dense spots. Read at several scales the inversions all but vanish, which
  is the clumping of MEASURE.md 9.2 at the level of whole datasets.
- **The anchor separates the methods more sharply than the masking rate**
  (Friedman 33.6 against 21.4), because of these inversions.
- **The anchor and the masking rate disagree on single runs**
  (`figures/week6_auc_vs_masking.png`): PCA hides 83% of mnist's top 5% and
  keeps 82% of its AUC; TriMap hides 93% of cardio's and keeps 77%.

### 4.4 What the failures cost

Labelled anomalies among the original top 5%, summed over datasets, mean
over seeds:

| | PCA | random | UMAP | t-SNE | PaCMAP | TriMap |
|---|---:|---:|---:|---:|---:|---:|
| hidden by the picture (of 1,496) | 752 (50%) | 700 (47%) | 1,214 (81%) | 1,091 (73%) | 1,263 (84%) | 971 (65%) |
| promoted as phantoms | 422 | 356 | 211 | 226 | 228 | 317 |

The extremes are datasets whose anomalies form their own group: PCA hides 4
of satimage2's 63 anomalies in the top 5% and 2 of pendigits' 31; UMAP hides
62 and 31.

### 4.5 Secondary analyses

- **The anomaly-count budget** (P@n) gives the same order: average ranks
  agree with the top-5% ranking at rank correlation 0.94. Friedman p =
  0.0004; PCA beats PaCMAP and UMAP, the random projection beats UMAP.
- **Heavily tied datasets**: none (section 3), so the repeat was not needed.
- **The multi-scale candidate** (MEASURE.md 9.2) lowers the neighbour-based
  methods' masking rates by 0.09 to 0.11 on average and moves the linear
  ones by a median of 0.012 (largest: PCA on musk, 0.58 to 0.34). Means:
  PCA 0.55, random 0.60, t-SNE 0.59, TriMap 0.60, UMAP 0.67, PaCMAP 0.68. The
  ranking changes (rank correlation with version 1's average ranks 0.77):
  t-SNE and TriMap move level with PCA, Friedman's p rises to 0.021, and
  only TriMap against PaCMAP stays significant after Holm. The largest
  changes are where the anomalies form groups: mnist (UMAP 0.78 to 0.27,
  TriMap 0.41 to 0.20), pendigits (UMAP 0.91 to 0.55) and musk. On letter
  it goes the other way (UMAP 0.84 to 0.97, PaCMAP 0.87 to 1.00).
  `figures/week6_readings.png`.

### 4.6 Seeds

The masking rate moves over three seeds by a median of 0.02 (UMAP, t-SNE),
0.04 (TriMap), 0.05 (PaCMAP) and 0.10 (random projection). The largest moves
are on wdbc, whose 19-point shortlist turns 5 points into 0.26, and for the
random projection on pageblocks (0.25). Week 8.

## 5. Expected against observed

| | expectation | observed |
|---|---|---|
| B1 | neighbour-based methods above both linear ones on most datasets; Friedman rejects | held: 10 of 15; p = 0.0007 |
| B2 | random projection below PCA on about half the datasets | held: 6 of 15 |
| B3 | phantom rate within 0.05 of the masking rate in most runs | held: 90% |
| B4 | AUC retained separates the methods less clearly | not held: it separates them more (33.6 against 21.4), because of the inversions in 4.3 |
| B5 | same order under the anomaly-count budget and without heavy ties | held for the budget (0.94); no heavily tied dataset |
| B6 | the candidate lowers the neighbour-based rates, moves the linear ones little, keeps the neighbour-based above on average | held (0.63 against 0.58 on average), though above both linear methods on only 7 datasets instead of 10 |
| B7 | heavily tied datasets shatter into islands | not testable: no heavily tied dataset; the large datasets with many discrete fields shatter instead |

## 6. What this shows, and what it does not

- Fifteen datasets separate the worst methods from the best, not the middle.
  The critical difference (1.95 ranks) is wide.
- Everything is for one detector, kNN at k = 20, at library defaults, three
  seeds and 2D. LOF on the continuous datasets comes with week 7, the
  hyperparameters with week 8 (`docs/QUESTIONS.md` 9, 20, 21).
- How a picture is read decides part of the answer: the multi-scale reading
  narrows the gap between neighbour-based and linear methods and removes the
  AUC inversions. Which reading to keep is open (`docs/QUESTIONS.md` 7). The
  primary analysis was fixed before the results, and they do not change it.
- Labels enter only the anchor and the cost; the measure never uses them.
