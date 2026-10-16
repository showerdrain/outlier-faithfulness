# Outlier faithfulness: definition, version 1

Week 4 (16 October 2026). Built and tested on the five development datasets
only (`results/splits.json`: shuttle, wbc, ionosphere, fault, magic). No
benchmark dataset was loaded while the definition was being made. Version 1
will change in week 5. It is frozen in week 12.

Code: `src/measure.py` (the definition), `src/explore_measure.py` and
`src/scale_check.py` (the tests in section 5), `src/validate.py` sections F and
H (known-answer checks). Numbers below come from `results/week4_measure.json`,
`results/week4_nulls.json` and `results/week4_scale.json`.

## 1. The question

An analyst runs a detector on the original data, gets a ranked list of
suspicious points, and looks at a 2D projection to judge them. For each point:
does the projection keep it where the detector put it?

Two things can go wrong, and they cost different things:

- **masked**: the detector ranks the point among the most outlying, the
  projection does not. The analyst looks at the plot and the point sits in the
  crowd. If it is a real anomaly, it is missed because of the projection.
- **phantom**: the projection ranks the point among the most outlying, the
  detector does not. The plot shows an isolated point that is ordinary in the
  data. Analyst time is spent on nothing.

## 2. Definition

Let `a` be the detector's scores in the original space and `b` the same
detector's scores on the projected coordinates, for `n` points. Let `m` be the
shortlist budget: how many points the analyst inspects.

**Ranks.** `rho(i)` is the descending rank of point `i`, 1 = most outlying.
Scores within a relative tolerance of `1e-9` count as tied, and tied points
share the average of the positions they span.

**Displacement**, in bits:

    lambda(i) = log2( rho_a(i) / rho_b(i) )

Negative: the projection moved the point toward the crowd. Positive: toward
the edge. `|lambda|` counts rank doublings, so `lambda = -3` means the point sits
eight times deeper in the projection's ranking than in the original's.

**Shortlists.** `S_a = {i : rho_a(i) <= m}` and `S_b = {i : rho_b(i) <= m}`.

**Noise null.** For each point, the distribution of the displacement it would
show between two equally valid measurements of the original space, with no
projection involved. `B = 100` replicate rankings are made. Replicate `r` draws
a random half `H_r` of the points and scores every point by its distance to
the `(k/2)`-th nearest neighbour within `H_r`, itself excluded. Ranks are taken
over all `n` points. The null for point `i` is the set of differences
`D_i = log2 rho_r(i) - log2 rho_s(i)` over all replicate pairs `r < s`.

**Calls**, with `p(i) = P_null(|D_i| >= |lambda(i)|) / 2`, the one-sided p-value
in the direction the point moved, and `alpha = 0.05`:

| category | rule | meaning |
|---|---|---|
| retained | in `S_a` and in `S_b` | kept on the shortlist |
| masked | in `S_a`, not in `S_b`, `p < alpha` | hidden, beyond noise |
| exit, ambiguous | in `S_a`, not in `S_b`, `p >= alpha` | left the shortlist within noise |
| phantom | in `S_b`, not in `S_a`, `p < alpha` | promoted, beyond noise |
| entry, ambiguous | in `S_b`, not in `S_a`, `p >= alpha` | entered within noise |
| background | in neither | |

**Aggregates**, per dataset, projection and budget:

| | definition | reads as |
|---|---|---|
| masking rate MR | `n_masked / size(S_a)` | share of the detector's shortlist the projection hides |
| phantom rate PR | `n_phantom / size(S_b)` | share of the projection's shortlist that is not outlying in the data |
| adjusted overlap AO | `(O - E) / (min(size(S_a), size(S_b)) - E)` | 1 = same shortlist, 0 = no better than chance |

`O = size(S_a & S_b)` and `E = size(S_a) size(S_b) / n`, the overlap
expected when the projected scores are permuted across points. Under that
permutation the overlap is hypergeometric, which gives `p_perm`.

**Defaults:** detector kNN distance with `k = 20` in both spaces; budget top
5% (`m = ceil(0.05 n)`); `alpha = 0.05`; `B = 100`; datasets capped at 20,000
points (stratified, seed 0), the cap proposed for the benchmark.

## 3. Why each choice, and what it replaced

**Same detector, same parameters, on both sides.** As in the proposal.
"Outlying in a projection" therefore means "ranked high by the detector run on
the projected coordinates". It is a reproducible proxy, not visual salience,
and the gap between the two is still open (proposal, possible extensions).
Nothing in the definition depends on the detector being kNN.

**Log-rank displacement instead of `(rank_LD - rank_HD) / n`.** The placeholder
scored a move from rank 1 to rank 64 and a move from rank 1001 to rank 1064
the same, 63/n. An analyst cares about the first and never sees the second, so
the placeholder was dominated by the bulk of inliers. That is the objection
Schubert, Wojdanowski, Zimek and Kriegel (SDM 2012) raise against Spearman and
Kendall for outlier rankings: they weight every position equally and ignore
the class imbalance. In bits, the first move is -6 and the second -0.09. Ranks
also make the measure invariant to any monotone rescoring of either space,
including the scale of the projection.

**Ties are resolved by the data.** Average ranks, so a tied group shares one
rank and is wholly in or wholly out of a shortlist. A shortlist can then differ
from `m` by at most half a tied group. This replaces the stable sort, which
made the September cut reproducible but arbitrary among tied points (shuttle:
four points on one boundary score). Ties are detected with a tolerance,
because exact float equality already gave different answers on Windows and
Linux three times. Schubert et al. (2012) note that ties, especially among
clear inliers, are to be expected in outlier rankings.

**The budget is the top 5%, not the number of labelled anomalies.** The
measure asks whether the projection preserves what the detector sees, which
does not involve labels, so the budget should not either. The development
datasets also make the old convention unusable: three of the five have about
35% labelled anomalies, so a budget equal to the anomaly count calls a third
of the data "the top ranks". The P@n budget is still computed and reported as
a variant (section 5.4), and the budget sweep shows how much the choice
matters.

**Masked and phantom counts are no longer forced equal.** With equal budgets,
every point that leaves a shortlist is replaced by one that enters, so the
placeholder's two counts were identical by arithmetic. Under v1 a crossing
only counts as a failure if it is beyond noise, and exits and entries are
judged separately. Section 5 shows the counts separating where they should.

**Why a noise null, and why this one.** A point's rank is only as precise as
the sample it was estimated from. Without a null every boundary crossing is a
failure, including a point that moves from rank `m` to rank `m + 1`. The null
asks how far the point's rank would move anyway, between two equally valid
measurements of the same data. Four design decisions:

- *The competitor set stays fixed.* The real comparison ranks the same `n`
  points in both spaces, so the null does too. Only the reference set used to
  score them is resampled. Resampling the points themselves would add the
  noise of which competitors happen to be present, which is part of detection
  error and plays no part in the comparison.
- *Half-sampling.* For smooth statistics, the spread between a half-sample
  estimate and the full-sample estimate matches the full-sample standard
  error, as the bootstrap's does. The bootstrap itself would duplicate points,
  putting zero distances into the neighbour graph.
- *`k/2` neighbours in a half-sized reference.* This keeps the neighbourhood
  holding the same share of the data as `k` does in the full set.
- *The null is a property of the original space only.* It does not depend on
  the projection, so it is computed once per dataset, cached, and shared by
  every method, seed and dimension.

`validate.py` checks its calibration directly. Compare two fresh replicates
with each other, which is the null hypothesis itself, and the share of
boundary crossings called significant should stay at or below `alpha`. It is
2-3% on wbc, ionosphere and fault.

**Why not the permutation null of scDEED.** The proposal named a permutation
null, with scDEED (Xia et al. 2024) as precedent. scDEED permutes every
feature independently, re-embeds the permuted data with the same method, and
cuts the pooled scores at the 5th and 95th percentiles. Built the same way here
(section 5.5), it calls almost nothing. On data without dependence structure a
projection destroys the outlier ranking almost completely, so "worse than on
structureless data" is far too low a bar. Its cut-offs require a flagged point
to fall between 7 times (wbc) and 63 times (ionosphere) deeper in the ranking
before it counts. On ionosphere it calls one masked point where PCA's 2D
shortlist shares nothing at all with the original's.

**Permutation still sets the chance level.** It appears where it fits: in the
aggregate. Shuffling the projected scores across points gives the expected
overlap of two unrelated shortlists in closed form, so AO is 0 at chance and 1
when the shortlists agree. This has the same form as the chance adjustment
Campos et al. (2016) apply to P@n. Section H of `validate.py` checks that
shuffled scores give AO of about 0.

**Labels condition nothing in the measure.** A labelled anomaly that is not
outlying in the original space cannot be masked, because there is nothing for
the projection to hide. That is the conditioning on original-space outlyingness
the proposal asked for, and it is built into the definition. Labels come in
afterwards, to say what the failures cost:

- *masked anomalies:* detections the projection loses.
- *phantom anomalies:* labelled anomalies the projection promotes but the
  detector ranked low.

**Reported alongside, for the redundancy question.** Two aggregates are
reported next to v1, to show what they miss:

- the weighted correlation of Schubert et al. (2012), the closest existing
  measure of agreement between two outlier rankings, with their class-balancing
  weights (the union of the two shortlists against the rest), on
  rank-normalised scores;
- plain Spearman.

## 4. Checks with known answers

`python src/validate.py`, section H, plus the arithmetic in section F. All pass.

| check | result |
|---|---|
| identity projection: nothing moves (section C) | 0 points moved on six datasets |
| PCA to full dimension, a rotation: nothing moves | max abs(lambda) = 0 on wbc and ionosphere |
| swapping the two spaces negates lambda and swaps masked with phantom | exact |
| a monotone rescoring of either space changes nothing | identical categories |
| projected scores shuffled across points: AO about 0 | mean -0.000 over 200 shuffles |
| calibration: two fresh re-measurements of the original space compared | 2-3% of crossings called, at alpha = 5% |
| planted outliers in the directions PCA discards | 17-20 of 20 called masked, over five seeds |
| planted outliers in the plane PCA keeps | 20 of 20 retained, none masked |
| ties: tied scores share a rank; a tied group is wholly in or out | exact |

## 5. Results on the development set

Detector kNN, k = 20. Projections to 2D, top 5% budget, unless stated. PCA is
deterministic. Random projection uses 5 seeds; UMAP, t-SNE and PaCMAP 3 seeds
each, at library defaults. Figures are in `figures/week4_*.png`.

### 5.1 Expected against observed

The expectations come from the September handover (section 6 there), written
before v1 existed.

| expectation | observed | |
|---|---|---|
| identity and rotation move nothing | zero on all five datasets | held |
| failures fall as PCA keeps more dimensions, and vanish at full dimension | yes on all five (5.3) | held |
| wbc is a clean negative control | PCA to 2D: masking 0%, phantom 0%, AO +0.91 | held, **for PCA only**: t-SNE and PaCMAP mask 81% and 97% |
| ionosphere is damaged most | PCA to 2D: masking 100%, AO -0.05, chance level | held |
| shuttle is a control: "nothing is lost" | PCA to 2D hides 57% of its top 5% | **did not hold** (below) |
| UMAP absorbs outliers (Herrmann et al.) | UMAP masks 61-100% and sits at chance on shuttle and ionosphere | consistent, with a scale caveat (5.6) |

**Shuttle was never a control at the top of the ranking.** Its September
status rested on AUC retained (100%) and Spearman (0.79). Both are dominated by
the bulk of inliers, which is exactly what the thesis argues. Its median LID of
about 2 is local. Globally shuttle spans six linear dimensions: PCA needs five
components for 89% of the variance and six for all of it. PCA to 2D therefore
lays separate low-dimensional pieces on top of each other. Masking falls from
57% at d = 2 to 2% at d = 5 and 0% at d = 6. Local and global dimensionality are
different things, and only the global one matters to a linear projection. That
belongs in the discussion of Question 3.

### 5.2 Methods at 2D

Mean over seeds, with the range in brackets. "Bits, masked" is the median
displacement of masked points: -3.4 means about ten times deeper in the ranking.

| dataset | method | masking rate | phantom rate | adjusted overlap | bits, masked | weighted corr. | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| shuttle | PCA | 0.57 | 0.50 | +0.34 | -3.4 | 0.77 | 0.70 |
| shuttle | random | 0.40 (0.37-0.45) | 0.36 (0.28-0.44) | +0.49 | -2.9 | 0.82 | 0.74 |
| shuttle | UMAP | 0.91 (0.90-0.92) | 0.88 (0.88-0.90) | +0.02 | -3.9 | 0.25 | 0.34 |
| shuttle | t-SNE | 0.73 (0.73-0.74) | 0.77 (0.76-0.77) | +0.18 | -3.3 | 0.50 | 0.52 |
| shuttle | PaCMAP | 0.85 (0.84-0.86) | 0.84 (0.82-0.85) | +0.07 | -4.1 | 0.22 | 0.30 |
| wbc | PCA | 0.00 | 0.00 | +0.91 | | 0.87 | 0.71 |
| wbc | random | 0.37 (0.25-0.58) | 0.38 (0.25-0.58) | +0.58 | -2.0 | 0.82 | 0.68 |
| wbc | UMAP | 0.61 (0.42-1.00) | 0.61 (0.42-1.00) | +0.35 | -2.5 | 0.54 | 0.42 |
| wbc | t-SNE | 0.81 (0.75-0.83) | 0.81 (0.75-0.83) | +0.15 | -3.2 | 0.81 | 0.76 |
| wbc | PaCMAP | 0.97 (0.92-1.00) | 0.97 (0.92-1.00) | -0.03 | -4.8 | -0.07 | 0.00 |
| ionosphere | PCA | 1.00 | 1.00 | -0.05 | -3.4 | 0.55 | 0.56 |
| ionosphere | random | 0.66 (0.56-0.78) | 0.63 (0.50-0.78) | +0.31 | -3.1 | 0.79 | 0.76 |
| ionosphere | UMAP | 1.00 | 1.00 | -0.05 | -4.4 | 0.09 | 0.10 |
| ionosphere | t-SNE | 0.85 (0.67-0.94) | 0.85 (0.67-0.94) | +0.10 | -3.4 | 0.62 | 0.60 |
| ionosphere | PaCMAP | 1.00 | 1.00 | -0.05 | -4.7 | -0.21 | -0.08 |
| fault | PCA | 0.58 | 0.59 | +0.37 | -3.3 | 0.57 | 0.48 |
| fault | random | 0.59 (0.52-0.79) | 0.56 (0.47-0.78) | +0.36 | -3.2 | 0.57 | 0.46 |
| fault | UMAP | 0.82 (0.79-0.85) | 0.83 (0.77-0.87) | +0.11 | -4.2 | 0.22 | 0.36 |
| fault | t-SNE | 0.70 (0.68-0.72) | 0.70 (0.67-0.73) | +0.25 | -3.4 | 0.65 | 0.64 |
| fault | PaCMAP | 0.98 (0.97-0.98) | 0.97 (0.95-0.98) | -0.03 | -4.6 | 0.07 | 0.28 |
| magic | PCA | 0.39 | 0.39 | +0.56 | -2.0 | 0.81 | 0.63 |
| magic | random | 0.38 (0.33-0.44) | 0.38 (0.34-0.44) | +0.58 | -2.4 | 0.72 | 0.54 |
| magic | UMAP | 0.85 (0.83-0.86) | 0.85 (0.83-0.86) | +0.10 | -4.6 | 0.10 | 0.14 |
| magic | t-SNE | 0.93 (0.92-0.94) | 0.93 (0.92-0.94) | +0.02 | -4.5 | 0.09 | 0.19 |
| magic | PaCMAP | 0.89 (0.87-0.93) | 0.89 (0.86-0.93) | +0.06 | -4.6 | 0.09 | 0.14 |

What the table says:

- **The neighbour-based methods hide more of the top 5% than either linear
  method on four of the five datasets**: UMAP 61-100%, t-SNE 70-93%, PaCMAP
  85-100%, against 0-100% for PCA and 37-66% for a random projection. The
  exception is ionosphere, where PCA already hides everything, t-SNE hides
  85%, and only the random projection does clearly better. PaCMAP is at chance
  level (AO between -0.05 and +0.07) on all five datasets, and UMAP is on
  shuttle and ionosphere. Section 5.6 qualifies this for wbc and fault.
- **A random projection keeps more than PCA** on shuttle (40% masked against
  57%) and ionosphere (66% against 100%), as found in October with AUC. It
  loses on wbc. PCA throws away low-variance directions whatever they hold.
- **Aggregate correlations miss the shortlist.** On ionosphere, PCA keeps a
  Spearman of 0.56 and a weighted correlation of 0.55 while its 2D shortlist
  shares nothing with the original's. On wbc, t-SNE keeps 0.76 and 0.81 while
  hiding 81% of the shortlist. Even the outlier-weighted correlation of
  Schubert et al. does not capture what an analyst inspects. That is early
  evidence for the non-redundancy half of Question 1. The full test against
  trustworthiness, continuity and the rest is week 11.
- **The two failure modes are measured separately, and can differ.** At 2D
  most crossings are far beyond noise, so masking and phantom rates are close.
  They separate where the mechanism differs. Shuttle under PCA to 3D hides 32%
  of the original top 5% while only 7% of its own top 5% are phantoms: the
  points that enter the shortlist come from just below the boundary and are
  within noise, while the points that leave come from high up and fall far.

### 5.3 PCA target dimension

Masking / phantom rate, top 5%:

| d | shuttle (D=9) | wbc (D=9) | ionosphere (D=32) | fault (D=27) | magic (D=10) |
|---:|---|---|---|---|---|
| 1 | 0.69 / 0.67 | 0.17 / 0.17 | 1.00 / 0.94 | 0.77 / 0.77 | 0.39 / 0.40 |
| 2 | 0.57 / 0.50 | 0.00 / 0.00 | 1.00 / 1.00 | 0.58 / 0.59 | 0.39 / 0.39 |
| 3 | 0.32 / 0.07 | 0.00 / 0.00 | 0.94 / 0.94 | 0.46 / 0.50 | 0.29 / 0.28 |
| 4 | 0.18 / 0.02 | 0.00 / 0.00 | 0.94 / 0.94 | 0.43 / 0.43 | 0.20 / 0.19 |
| 6 | 0.00 / 0.00 | 0.00 / 0.00 | 0.44 / 0.44 | 0.34 / 0.21 | 0.06 / 0.03 |
| 8 | 0.00 / 0.00 | 0.00 / 0.00 | 0.44 / 0.44 | 0.19 / 0.09 | 0.01 / 0.00 |
| 16 | | | 0.11 / 0.11 | 0.01 / 0.00 | |
| 24 | | | 0.00 / 0.00 | 0.00 / 0.00 | |
| D | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

Failures fall with dimension and vanish before or at full dimension on all
five. Not strictly monotone: ionosphere goes 0.11 at d = 16, 0.22 at d = 20
(4 points of 18 against 2), then 0 from d = 24.

The null matters most here. At d = 6 on shuttle, 46 of the original top 5%
still drop out of the PCA shortlist, by a median of 0.18 bits. All 46 are
within noise, so the masking rate is 0. The placeholder would have counted
them all as failures of a projection that keeps all the variance.

### 5.4 The budget

Masking / phantom rate at 2D, seed 0:

| dataset | method | top 1% | 2% | 5% | 10% | 20% | anomaly count (share of n) |
|---|---|---|---|---|---|---|---|
| shuttle | PCA | 0.47 / 0.45 | 0.59 / 0.50 | 0.57 / 0.50 | 0.49 / 0.47 | 0.38 / 0.35 | 0.53 / 0.50 (7%) |
| shuttle | UMAP | 1.00 / 1.00 | 1.00 / 0.95 | 0.91 / 0.90 | 0.78 / 0.80 | 0.61 / 0.65 | 0.84 / 0.84 (7%) |
| wbc | PCA | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.09 / 0.09 | 0.22 / 0.22 | 0.10 / 0.10 (4%) |
| wbc | UMAP | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.71 / 0.71 | 1.00 / 1.00 (4%) |
| ionosphere | PCA | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.89 / 0.89 | 0.74 / 0.73 | 0.47 / 0.46 (36%) |
| ionosphere | UMAP | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.83 / 0.83 | 0.57 / 0.55 (36%) |
| fault | PCA | 0.40 / 0.40 | 0.59 / 0.62 | 0.58 / 0.59 | 0.59 / 0.58 | 0.53 / 0.50 | 0.41 / 0.41 (35%) |
| fault | UMAP | 1.00 / 1.00 | 0.97 / 0.95 | 0.83 / 0.87 | 0.68 / 0.66 | 0.59 / 0.59 | 0.43 / 0.45 (35%) |
| magic | PCA | 0.57 / 0.57 | 0.53 / 0.51 | 0.39 / 0.39 | 0.28 / 0.28 | 0.29 / 0.28 | 0.30 / 0.29 (35%) |
| magic | UMAP | 0.94 / 0.94 | 0.90 / 0.91 | 0.86 / 0.86 | 0.76 / 0.77 | 0.65 / 0.65 | 0.56 / 0.56 (35%) |

The level moves with the budget, generally down as the shortlist grows, but
the ordering does not: UMAP is at or above PCA at every budget on every
dataset. The anomaly-count budget gives the lowest rates on the three datasets
with about 35% anomalies, because there a third of the data counts as "top".
That is the reason for fixing the budget as a share of the data instead.

### 5.5 Null models

Shortlist crossings called significant, PCA to 2D, top 5%. "Agreement" is the
Jaccard overlap of the masked set with v1's.

| dataset | v1: half-reference, B = 100 | same, B = 50, other seed | k-variation, k = 10..40 | feature permutation (scDEED-style) | none: every crossing |
|---|---|---|---|---|---|
| shuttle, 626 exits | 569 | 576 (0.97) | 562 (0.93) | 84 (0.15) | 626 (0.91) |
| wbc, 1 exit | 0 | 0 | 1 | 0 | 1 |
| ionosphere, 18 exits | 18 | 18 (1.00) | 18 (1.00) | 1 (0.06) | 18 (1.00) |
| fault, 59 exits | 57 | 57 (1.00) | 58 (0.95) | 0 (0.00) | 59 (0.97) |
| magic, 392 exits | 369 | 369 (0.98) | 383 (0.96) | 12 (0.03) | 392 (0.94) |

The v1 null is stable to its own Monte Carlo and agrees closely with a quite
different notion of noise, the detector's sensitivity to k. The scDEED-style
permutation null calls almost nothing (cut-offs of 2.9 to 6.0 bits).

At 2D the null reclassifies few crossings as ambiguous: 9% on shuttle, 6% on
magic, 3% on fault, none on ionosphere, and the single crossing on wbc. At
higher dimension, where the projection is nearly faithful, it reclassifies
most of them (5.3). In other words, it does its job at the boundary and leaves
large displacements alone.

### 5.6 The scale of the projection-side reading

v1 reads the projection with the same detector at the same k = 20. The UMAP
pictures (`figures/week4_maps.png`) show why that is not innocent. UMAP and
PaCMAP gather isolated outliers into tight clumps. On wbc, the 12 points of
the original top 5% end up in small groups at the edge of the UMAP picture.
Scored at k = 5 they look like inliers (picture ranks 29-223). At k = 20 they
are middling (25-222). At k = 50, nine of them are back in the top 14.

Masking rate at 2D, seed 0, with the original space fixed at k = 20 and the
projection read at k_LD (`src/scale_check.py`, `results/week4_scale.json`):

| dataset | method | k_LD 5 | 10 | 20 (v1) | 50 | 100 | both sides at 50 |
|---|---|---:|---:|---:|---:|---:|---:|
| shuttle | PCA | 0.53 | 0.53 | 0.57 | 0.61 | 0.63 | 0.53 |
| shuttle | UMAP | 0.83 | 0.81 | 0.91 | 0.87 | 0.86 | 0.89 |
| shuttle | PaCMAP | 0.87 | 0.86 | 0.84 | 0.84 | 0.85 | 0.89 |
| wbc | PCA | 0.17 | 0.17 | 0.00 | 0.00 | 0.00 | 0.08 |
| wbc | UMAP | 1.00 | 1.00 | 1.00 | 0.33 | 0.33 | 0.33 |
| wbc | t-SNE | 0.83 | 0.83 | 0.83 | 0.33 | 0.33 | 0.33 |
| wbc | PaCMAP | 0.83 | 0.83 | 0.92 | 0.17 | 0.17 | 0.17 |
| ionosphere | PCA | 0.89 | 0.94 | 1.00 | 0.94 | 0.94 | 0.94 |
| ionosphere | UMAP | 0.72 | 0.78 | 1.00 | 1.00 | 1.00 | 1.00 |
| fault | PCA | 0.59 | 0.58 | 0.58 | 0.59 | 0.58 | 0.64 |
| fault | UMAP | 0.91 | 0.81 | 0.83 | 0.66 | 0.80 | 0.63 |
| fault | PaCMAP | 0.95 | 0.96 | 0.97 | 0.67 | 0.70 | 0.63 |
| magic | PCA | 0.43 | 0.41 | 0.39 | 0.38 | 0.35 | 0.37 |
| magic | UMAP | 0.90 | 0.88 | 0.86 | 0.72 | 0.65 | 0.73 |
| magic | t-SNE | 0.95 | 0.94 | 0.92 | 0.88 | 0.81 | 0.89 |

The full table, with random projection and every method on every dataset, is
in the JSON. The linear methods give the same verdict at every scale. The
neighbour-based methods do not. On shuttle, ionosphere and magic they hide
most of the top 5% at every scale tried. On wbc and fault their excess
masking shrinks or disappears once the picture is read at a scale larger than
the clumps they create.

This is the projection turning isolated outliers into small clusters: the
clustered-anomaly problem, created by the projection rather than present in
the data. Whether a tight clump at the edge of a UMAP plot counts as
"outlying" to a person is a question about visual salience, the open item
from the proposal. v1 keeps k equal on both sides, which is the proposal's
rule, and this scale profile is reported alongside it. It is the main thing
to settle in week 5 (section 7).

### 5.7 What the failures cost

Labels only enter here. Means over seeds, 2D, top 5%.

| dataset | method | anomalies in the original top 5% | of those, masked | share of anomalies, original top 5% | same, projection's top 5% | phantom anomalies |
|---|---|---:|---:|---:|---:|---:|
| shuttle | PCA | 247 | 84 | 0.25 | 0.21 | 44 |
| shuttle | UMAP | 247 | 235 | 0.25 | 0.09 | 71 |
| wbc | PCA | 9 | 0 | 0.75 | 0.67 | 0 |
| wbc | PaCMAP | 9 | 9 | 0.75 | 0.00 | 0 |
| ionosphere | PCA | 18 | 18 | 1.00 | 0.28 | 5 |
| ionosphere | random | 18 | 11.8 | 1.00 | 0.97 | 10.8 |
| ionosphere | t-SNE | 18 | 15.3 | 1.00 | 0.89 | 13.3 |
| fault | PCA | 63 | 38 | 0.64 | 0.44 | 19 |
| magic | PCA | 914 | 353 | 0.97 | 0.88 | 270 |
| magic | t-SNE | 914 | 855 | 0.97 | 0.57 | 480 |

Two patterns:

- **Masking costs real detections.** UMAP hides 235 of the 247 labelled
  anomalies in shuttle's top 5%; t-SNE hides 855 of 914 on magic.
- **Some pictures find different anomalies rather than none.** A random
  projection of ionosphere hides two thirds of the detector's shortlist, yet
  its own top 5% is still 97% anomalies. t-SNE does the same at 89%. This is the
  September musk pattern (AUC up, agreement down) seen at the level of single
  points. The measure counts those points as phantoms, which is right for
  preservation: the picture disagrees with the detector. The label breakdown
  shows when a phantom happens to be a real anomaly.

### 5.8 LID, exploratory only

Median LID of masked against retained points shows no consistent difference
at 2D. Examples: magic under PCA, 6.4 against 5.9; under t-SNE, 6.0 against 6.6.
On shuttle, masked points have higher LID than retained ones under PCA (2.8
against 2.4) and UMAP (2.8 against 1.4), lower under t-SNE (2.5 against 3.4).
This is not a test of Question 3, which needs per-point modelling (week 10).
The planted fixture still shows the mechanism in its clean form: outliers in
the discarded directions have LID about 25 against about 5 for those PCA keeps.

## 6. What v1 does not do

- **The projection-side scale** (5.6) is the largest open problem.
- **Ranks only.** A point far ahead of the next one and a point barely ahead
  are treated alike. A score-based variant, with scores put on a common scale
  as in Kriegel, Kröger, Schubert and Zimek (SDM 2011), is the obvious v2
  candidate.
- **One detector.** Everything here is kNN at k = 20. The definition takes any
  detector, but that has not been shown.
- **One embedding at a time.** A per-point call refers to one seed. On small
  datasets the seed matters a lot: UMAP's masking rate on wbc runs from 42% to
  100% over three seeds. Week 8.
- **No multiplicity correction** for per-point calls. The calibration check
  puts the false-call rate at 2-3%, so rates are not inflated, but a single
  call is a 5%-level statement.
- **The noise null covers the original space only.** Noise a projection adds
  counts against it, deliberately: the analyst sees one picture.
- **Shuttle is subsampled** to 20,000 points, the proposed benchmark cap.

## 7. For the supervisor

1. **Null model.** The proposal named a permutation null. v1 uses a resampling
   null per point and keeps permutation for the chance level of the aggregate,
   because the scDEED-style permutation null calls almost nothing (5.5). Is
   that acceptable?
2. **Budget.** Top 5% of points, which needs no labels, instead of the number
   of labelled anomalies. Three of five development datasets have about 35%
   anomalies (5.4).
3. **Scale of the projection-side reading** (5.6). Keep the proposal's rule
   (same detector, same k), read the projection at several scales and report
   the profile, or define a multi-scale reading? This decides how the UMAP
   result is stated.
4. **Shuttle.** Not a control for PCA (5.1). Keep it as the case where local
   and global dimension differ?
5. **Ranks or normalised scores** for v2 (section 6)?

## 8. Week 5

- Try two projection-side readings against v1 on the development set: several
  fixed scales, and a multi-scale score (most outlying at any of k, 2k, 5k).
- A score-based variant, compared with v1 point by point.
- A second detector (LOF) through the same code, to show the definition is
  detector-agnostic.
- Seed stability of per-point calls on the small datasets.
- A first per-point look against trustworthiness and continuity (redundancy).

## Changelog

- **v1, 16 October 2026.** Replaces the September placeholder,
  `(rank_LD - rank_HD) / n` with a hard top-k cut at the anomaly count. Changes:
  displacement in bits; average ranks with tolerance for ties; a per-point
  noise null; masked and phantom called separately; chance-adjusted overlap;
  budget of the top 5%.
