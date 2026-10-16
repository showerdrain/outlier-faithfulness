# Outlier faithfulness: definition, version 1

Version 1 was defined in week 4. Week 5 put it through sanity checks and
tested a revision, a multi-scale reading (candidate version 2), against
adoption criteria fixed before the results were read. The candidate missed
two of the four criteria, both narrowly, so version 1 stands (section 9). Both
weeks ran on 16 October 2026, on the five development datasets only
(`results/splits.json`: shuttle, wbc, ionosphere, fault, magic), which the
plan in the proposal calls the exploration set. No benchmark dataset was
scored while the definition was being made. The benchmark, run with version 1
and with the multi-scale reading beside it, is in `docs/BENCHMARK.md`. The
definition is frozen in week 12.

Sections 1 to 8 are the week-4 text, unchanged except where marked; section 9
is week 5. Open questions, each with the decision taken, are in
`docs/QUESTIONS.md`, which replaces section 7.

Code: `src/measure.py` (the definition) and `src/detectors.py` (the detector
readings), `src/explore_measure.py`, `src/scale_check.py` and
`src/lattice_check.py` (section 5), `src/revise_measure.py` and
`src/score_check.py` (section 9), `src/validate.py` sections F, H and I
(known-answer checks). Numbers come from `results/week4_*.json` (section 5) and
`results/week5_*.json` (section 9). The week-4 definition in LaTeX is
`docs/week4.tex`; week 5 and the benchmark are in `docs/week5_6.tex`.

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
the edge. `|lambda|` counts rank doublings, so `lambda = -3` means the point
sits eight times deeper in the projection's ranking than in the original's.

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
points (stratified, seed 0), the cap used for the benchmark.

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

For the kNN score the half-sampling argument can be made directly, to first
order. Each of a point's neighbours lands in a random half with probability
1/2, so its `(k/2)`-th neighbour within the half is its `j`-th neighbour in the
full data, where `j` has mean `k` and variance about `k`. The full-data `k`-th
neighbour distance is itself uncertain by about the same amount: the ball it
defines holds `k ± sqrt(k)` points' worth of the underlying distribution (a
Poisson count). So a replicate scatters around the full-data ranking about as
much as the full-data ranking scatters around the truth, and the difference
of two replicates has the spread of the difference of two independent
full-data measurements. That difference is what `lambda` is when the
projection keeps the ranking up to sampling noise.

`validate.py` checks the null's size. Compare two fresh replicates with each
other, which is the null hypothesis itself, and the share of boundary
crossings called significant should stay at or below `alpha`. It is 2-3% on
wbc, ionosphere and fault. This confirms the calls are not inflated; it cannot
confirm that sampling noise is the right notion of "equally valid", which is
an argument (above), not a test.

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

Week 5 added section I, twelve more checks (56 in all, all passing), for the
multi-scale candidate, LOF and per-point trustworthiness and continuity
(9.3).

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
belongs in the discussion of Question 3. Shuttle is also lattice data, which
shapes its neighbour-based pictures (5.9).

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
  85%, and only the random projection does clearly better. PaCMAP's shortlist
  is close to chance on all five datasets (mean AO between -0.05 and +0.07):
  at chance on wbc, ionosphere and fault, and above it on shuttle and magic
  only by a little (p_perm < 0.01, AO at most 0.09). UMAP is at chance on
  ionosphere and close to it on shuttle (AO 0.00 to 0.03 over three seeds).
  Section 5.6 qualifies this for wbc and fault.
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
permutation null calls almost nothing: a point must fall 2.9 to 6.0 bits to
count as masked, and rise 1.9 to 6.1 bits to count as a phantom.

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

### 5.9 Two of the five are lattice data

Every feature of shuttle and of wbc is a whole number. Shuttle's nine take a
median of 86 distinct values each; wbc's nine are 1-10 scores, a median of 9
values each. Ionosphere, fault and magic are continuous. Two consequences
(`src/lattice_check.py`, `results/week4_lattice.json`):

- **Ties.** Shuttle's 20,000 points take only 8,351 distinct kNN scores, 42%
  as many as there are points, against 81% on wbc and 99-100% on the three
  continuous datasets. Distances on a grid repeat. This is why the tie policy
  (section 3) matters on shuttle and hardly anywhere else.
- **Shattered pictures.** Shuttle's 2D pictures break into small islands:
  groups of points whose 10 nearest neighbours all lie inside the group
  (strongly connected components of the 10-NN graph, seed 0). There are
  1,188 under PCA, 1,191 under PaCMAP, 1,480 under t-SNE and 1,511 under UMAP,
  and the largest holds under 2% of the points. A random projection smears
  the grid most: 169 islands, the largest holding 71%. Magic, with about as
  many points, gives 5 to 65 islands under every method, with 87% or more of
  its points in the largest. Tight groups of near-identical points become
  separate neighbourhoods, and a picture made of a thousand islands has no
  stable notion of "apart from the rest". Shuttle's results for UMAP, t-SNE
  and PaCMAP describe the islands as much as the methods.

Network data is full of discrete fields (ports, flags, counts), so the same
will come back with backdoor. Question 6 in section 7.

## 6. What v1 does not do

Week-4 text; what week 5 found is added in italics.

- **The projection-side scale** (5.6) is the largest open problem. *Week 5:
  a multi-scale reading fixes the clumps but missed its adoption criteria;
  it is reported beside version 1 (9.2, 9.11).*
- **At 2D it is close to a top-m overlap.** When nearly every boundary
  crossing is far beyond noise, as at 2D, the masking rate sits close to the
  plain exit rate `1 - O/m`: 0.02 below it on average over every 2D run, at
  most 0.10 below except on one wbc run with a 12-point shortlist (0.17). AO
  is the chance-adjusted top-m overlap, in the form of adjusted precision at
  n. The null earns its place near the boundary and in nearly faithful
  projections (5.3). What v1 adds over a top-m overlap is the per-point call,
  the two failure modes kept apart, and the noise null. Whether that adds
  anything over trustworthiness and continuity is the week-11 test. *Week 5:
  a first look says it does for the neighbour-based methods (9.8).*
- **Ranks only.** A point far ahead of the next one and a point barely ahead
  are treated alike. A score-based variant, with scores put on a common scale
  as in Kriegel, Kröger, Schubert and Zimek (SDM 2011), is the obvious v2
  candidate. *Week 5: neither common scale carries over between spaces of
  different dimension, so ranks stay (9.9).*
- **One detector.** Everything here is kNN at k = 20. The definition takes any
  detector, but that has not been shown. *Week 5: LOF runs through the same
  code and stays close to calibrated, but orders the methods differently
  (9.6).*
- **One embedding at a time.** A per-point call refers to one seed. On small
  datasets the seed matters a lot: UMAP's masking rate on wbc runs from 42% to
  100% over three seeds. Week 8. *Week 5: five seeds; a random projection's
  calls depend on the seed above all (9.7).*
- **No multiplicity correction** for per-point calls. The calibration check
  puts the false-call rate at 2-3%, so rates are not inflated, but a single
  call is a 5%-level statement.
- **The noise null covers the original space only.** Noise a projection adds
  counts against it, deliberately: the analyst sees one picture.
- **Shuttle is subsampled** to 20,000 points, the proposed benchmark cap.

## 7. For the supervisor

Superseded by `docs/QUESTIONS.md`, which carries these six questions forward
(as 2, 3, 7, 4, 6 and 5) with the decision taken on each, and adds the
questions of weeks 5 to 16.

## 8. Week 5

The week-4 plan, each item done in section 9:

- Two projection-side readings against v1 on the development set: several
  fixed scales, and a multi-scale score (9.2 to 9.5).
- A score-based variant, compared with v1 (9.9).
- A second detector (LOF) through the same code (9.6).
- Seed stability of per-point calls (9.7).
- A first per-point look against trustworthiness and continuity (9.8).

## 9. Week 5: sanity checks and a multi-scale candidate

Everything here ran on the development datasets, at 2D with the top 5%
budget unless stated, with five seeds (0 to 4) for every stochastic method and
TriMap added as a sixth method. Code: `src/revise_measure.py` (`--report`
prints every table below), `src/score_check.py`, `src/validate.py` section I.
Results: `results/week5_revision.json`, `results/week5_redundancy_v1.json`,
`results/week5_score_check.json`. Figures: `figures/week5_*.png`. In the code
and the result files the candidate is called v2.

### 9.1 Expectations, written before the results were read

Written while `src/revise_measure.py` was running. The first lines for shuttle
(PCA and random projection) had been printed; nothing else had been seen.

| | expectation | why |
|---|---|---|
| E1 | identity and rotation move nothing under v2 | as for v1 |
| E2 | linear methods (PCA, random): v2 masking rate within 0.10 of v1's on every dataset | the week-4 scale profile (5.6): linear methods read the same at every scale |
| E3 | neighbour-based methods: v2 masks clearly less than v1 on wbc and fault, and still most of the top 5% on shuttle, ionosphere and magic | 5.6: the clumps those methods create are seen at k = 50 |
| E4 | the neighbour-based methods still mask more than both linear methods on at least three of the five datasets | if not, the week-4 ordering was an artefact of k = 20 |
| E5 | the verdict depends on the choice of scales less than v1's depended on k: v2 against three other scale sets differs by at most 0.10 in most runs | v1 varied by up to 0.67 over k (5.6) |
| E6 | LOF, on the three continuous datasets, orders the methods as kNN does | the definition does not depend on the detector |
| E7 | masking rate varies over five seeds by at most 0.05 on the two datasets with 20,000 points, much more on the small ones; v2's per-point calls are at least as stable as v1's | larger samples, steadier pictures |
| E8 | per-point continuity tells masked from retained points only moderately (AUC below 0.8), and trustworthiness phantoms from retained points likewise | if the AUC were above 0.9 on most runs, the per-point calls would add little to continuity |

Decision rule, fixed with the expectations: adopt v2 if it passes the
known-answer checks (`validate.py`, section I), E2 and E5 hold, and E7's
comparison does not go against it (the mean Jaccard overlap of the masked
sets across seeds no more than 0.05 below v1's). Otherwise keep v1 and report
the scale profile next to it.

### 9.2 The candidate: a multi-scale reading

Version 1 reads each space with one neighbourhood size, k = 20, and week 4
found that the verdict on the neighbour-based methods depends on it (5.6).
UMAP and PaCMAP gather isolated outliers into small tight clumps, and a member
of a clump of 40 has its 20th neighbour inside the clump. Read at k = 20 the
clump looks dense; read at k = 50 it is seen to be apart. Version 1's verdict
on UMAP for wbc moved from 100% masked to 33% with the scale.

The candidate reads every space at the scales `K = {20, 50, 100}`. With `d_k(i)`
the distance from point `i` to its `k`-th nearest neighbour,

    score(i) = max over k in K of  d_k(i) / median_j d_k(j)

how many times farther from its `k`-th neighbour point `i` is than a typical
point, at the scale where that is largest. The same rule scores both spaces;
the noise null scores each replicate at `k/2` per scale and combines them the
same way. Everything else is version 1, which is the case `K = {20}`. The
precedent is Breunig, Kriegel, Ng and Sander (2000), who rank points by their
largest LOF over a range of MinPts. Two details:

- *Ratios are combined, not ranks.* Dividing by the median gives a number
  with the same meaning at every scale. An earlier candidate took each
  point's best rank over the scales; on a test with a known answer (9.3) it
  let the scales split the shortlist, so half of a far-away clump still
  counted as hidden.
- *The largest scale is 100*, the depth of the stored neighbour graph and
  0.5% of a 20,000-point dataset. Scales above `(n - 1) / 2` would be dropped
  so the null stays defined; no dataset in this project loses one.

### 9.3 Checks with known answers

All 56 checks of `python src/validate.py` pass; section I, new this week,
holds 12 of them.

| check | result |
|---|---|
| LOF read off the neighbour graph equals scikit-learn's | largest relative error 6.5e-16, at k = 20 and 50 |
| per-point trustworthiness and continuity (`src/quality.py`) equal ZADU's | largest difference 0 |
| the generalised null is version 1's null when given one scale | bit for bit |
| candidate, kNN and LOF: identity and rotation move nothing | 0 points moved |
| candidate: swapping the spaces negates lambda; a monotone rescoring changes nothing | exact |
| candidate under the null itself: two fresh re-measurements compared | kNN 1.7% masked, 2.0% phantom; LOF 5.9% and 5.7%, at alpha = 5% |
| candidate, planted fixture | hidden outliers masked 17 to 20 of 20 (kNN), 19 to 20 (LOF); visible ones all retained |
| clumped fixture, far-away clump | version 1 masks 40 of 40; the candidate retains 40 of 40 |
| clumped fixture, scattered and absorbed outliers | the candidate retains 40 of 40 scattered ones and masks at least 38 of 40 absorbed ones |

The clumped fixture was built this week to separate a clump made by the
projection from a real loss. Forty outliers, each alone in its own direction
in ten dimensions, are placed in three pictures: together in one tight clump
far from the bulk (what UMAP and PaCMAP do), scattered on a wide circle, or
inside the bulk. LOF's calibration sits slightly above alpha; kNN's below it.

### 9.4 Version 1 against the candidate (E1 to E4)

Masking rate, mean over five seeds (PCA once), version 1 then the candidate:

| method | shuttle | wbc | ionosphere | fault | magic |
|---|---|---|---|---|---|
| PCA | 0.57, 0.47 | 0.00, 0.08 | 1.00, 1.00 | 0.58, 0.61 | 0.39, 0.37 |
| random | 0.40, 0.38 | 0.37, 0.38 | 0.66, 0.66 | 0.59, 0.56 | 0.38, 0.38 |
| UMAP | 0.90, 0.90 | 0.77, 0.32 | 0.98, 1.00 | 0.81, 0.71 | 0.85, 0.71 |
| t-SNE | 0.73, 0.81 | 0.77, 0.27 | 0.84, 0.86 | 0.69, 0.77 | 0.93, 0.86 |
| PaCMAP | 0.85, 0.82 | 0.95, 0.17 | 1.00, 1.00 | 0.97, 0.62 | 0.89, 0.87 |
| TriMap | 0.61, 0.63 | 0.82, 0.22 | 0.97, 0.99 | 0.75, 0.72 | 0.94, 0.94 |

Version 1's means use five seeds here and three in 5.2, which matters on wbc:
UMAP masks 61% of its top 5% over seeds 0 to 2 and 77% over seeds 0 to 4.

- **E1 held.** The identity and the rotation move nothing.
- **E2 missed, narrowly.** Shuttle under PCA moves from 0.569 to 0.468,
  0.101 against a bound of 0.10; the other nine linear runs move by 0.083 at
  most (one point of wbc's twelve). The candidate also changes which points
  form the original top 5%, so a linear method's verdict can move although
  its picture is the same.
- **E3 held, except for t-SNE.** On wbc the candidate cuts the neighbour-based
  methods' masking from 77-95% to 17-32%; in `figures/week5_maps.png` the
  points version 1 calls masked sit in tight clumps at the edge of the
  picture. On fault, PaCMAP falls from 0.97 to 0.62 and UMAP from 0.81 to
  0.71, while t-SNE rises from 0.69 to 0.77. On shuttle, ionosphere and magic
  the neighbour-based methods still hide 63 to 100%. t-SNE hides more at the
  larger scales on shuttle as well (0.73 to 0.81).
- **E4 held, at the bound.** The neighbour-based methods, on average, mask
  more than both linear methods on three datasets under the candidate
  (shuttle, fault, magic) and four under version 1 (wbc too). On wbc the
  candidate puts them (17-32%) between PCA (8%) and the random projection
  (38%); on ionosphere everything except the random projection hides 86 to
  100%.

### 9.5 The choice of scales (E5)

The candidate with three other scale sets, {20, 50}, {20, 100} and
{10, 20, 50, 100}, seed 0, against week 4's version 1 read at other k (5.6):

| | runs | median change | within 0.10 | largest change |
|---|---:|---:|---:|---:|
| candidate, other scale sets | 90 | 0.010 | 89% | 0.33 |
| version 1, other k | 125 | 0.041 | 71% | 0.75 |

(9.1 put version 1's largest change at 0.67, from UMAP on wbc; it is 0.75,
from PaCMAP on wbc.)

**E5 held.** The largest changes come from wbc, where one point is 8% of the
12-point shortlist (TriMap with {20, 100}: 0.17 to 0.50), and from TriMap and
t-SNE on fault (0.49 to 0.60, 0.71 to 0.59, with {20, 50}).

### 9.6 A second detector (E6)

LOF through the same code, read at the candidate's scales, on the three
continuous datasets (on lattice data LOF depends on how tied distances are
ordered). Masking rate, mean over seeds 0 to 2:

| dataset | detector | PCA | random | UMAP | t-SNE | PaCMAP | TriMap | rank correlation over methods |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ionosphere | LOF | 0.83 | 0.57 | 1.00 | 0.59 | 0.98 | 0.87 | +0.85 |
| | kNN | 1.00 | 0.63 | 1.00 | 0.85 | 1.00 | 1.00 | |
| fault | LOF | 0.41 | 0.74 | 0.63 | 0.71 | 0.79 | 0.67 | -0.43 |
| | kNN | 0.61 | 0.61 | 0.65 | 0.74 | 0.59 | 0.70 | |
| magic | LOF | 0.78 | 0.68 | 0.72 | 0.68 | 0.72 | 0.79 | +0.26 |
| | kNN | 0.37 | 0.40 | 0.71 | 0.87 | 0.87 | 0.94 | |

**E6 did not hold.** The definition takes LOF without change and stays close
to calibrated (9.3), so the measure itself does not depend on the detector.
The verdicts do. On magic, kNN finds that the linear methods keep most of the
top 5% (37-40% masked) and the neighbour-based methods lose most of it
(71-94%); LOF finds that every method loses about the same share (68-79%).
On fault they order the methods partly in reverse (rank correlation -0.43).
Which methods preserve outlyingness depends on which detector's notion of
outlyingness is to be preserved, so every comparison of methods in this
project is a statement about a named detector (`docs/QUESTIONS.md`, 9).

### 9.7 Seed stability (E7)

Per method, mean over the five datasets: the Jaccard overlap of the masked
sets between seeds, the spread of the masking rate over the five seeds, and
the share of the original top 5% that gets the same call under every seed
(masked under all five, or under none).

| method | Jaccard, v1 / candidate | spread, v1 / candidate | same call, v1 / candidate |
|---|---|---|---|
| random | 0.38 / 0.37 | 0.20 / 0.23 | 0.16 / 0.16 |
| UMAP | 0.88 / 0.87 | 0.15 / 0.11 | 0.79 / 0.80 |
| t-SNE | 0.89 / 0.80 | 0.11 / 0.13 | 0.82 / 0.73 |
| PaCMAP | 0.94 / 0.93 | 0.05 / 0.06 | 0.88 / 0.87 |
| TriMap | 0.89 / 0.74 | 0.07 / 0.16 | 0.82 / 0.72 |

**E7 did not hold**, in either part.

- On the two large datasets the masking rate moves over five seeds by up to
  0.15 (random projection, shuttle, candidate), 0.11 (UMAP and PaCMAP on
  magic, candidate) and 0.10 (TriMap on shuttle, version 1). t-SNE is the
  steadiest method, at 0.01 to 0.02 on both datasets.
- Over all runs the candidate's masked sets agree less across seeds: mean
  Jaccard 0.741 against 0.797, 0.056 lower, beyond the tolerance of 0.05. The
  loss is in t-SNE and TriMap; UMAP and PaCMAP are as stable under both. The
  Jaccard index falls when masked sets are small, as on wbc's 12-point
  shortlist, but the share of same calls, which does not depend on that,
  points the same way (0.656 against 0.695).
- A random projection's per-point calls depend on the seed above all: only
  16% of the top 5% gets the same call under all five random maps. One random
  projection says little about any single point.

### 9.8 Redundancy: per-point trustworthiness and continuity (E8)

Per-point trustworthiness t and continuity c (k = 20), seed 0. The AUC is for
1 - c separating the masked from the retained points of the original top 5%,
and for 1 - t separating the phantom from the retained points of the
projection's top 5%; 0.5 is no separation, below 0.5 the wrong way round.
Runs with no masked or no retained point have no AUC.

| | linear methods (PCA, random) | neighbour-based methods |
|---|---|---|
| 1 - c, masked, version 1 | median 0.90 (0.54-1.00); above 0.8 in 5 of 8 | median 0.24 (0.00-0.60); below 0.5 in 14 of 16 |
| 1 - c, masked, candidate | median 0.90 (0.54-1.00); above 0.8 in 7 of 9 | median 0.57 (0.07-1.00); below 0.5 in 7 of 16 |
| 1 - t, phantom, version 1 | median 0.45 | median 0.46 |
| 1 - t, phantom, candidate | median 0.60 | median 0.66 |

**E8 held for the neighbour-based methods and not for the linear ones.**

- When a linear map hides a point, it also replaces the point's neighbours,
  so continuity finds the masked points well (median AUC 0.90).
- The neighbour-based methods hide points that keep their original
  neighbours: under version 1, the points they hide have *higher* continuity
  than the points they keep in 14 of 16 runs, and on wbc every hidden point
  does (AUC 0 for t-SNE, PaCMAP and TriMap). That is what the clumps in
  `figures/week5_maps.png` look like: a group of outliers moved into the
  picture together, each with its neighbours, and packed tight.
- Trustworthiness says little about phantoms under either reading.
- In aggregate, continuity does not track the masking rate. On shuttle,
  fault and magic every method keeps a mean continuity of 0.90 to 1.00, and
  the neighbour-based methods, which mask the most, keep 0.97 to 1.00 (right
  panel of `figures/week5_redundancy.png`, candidate's calls).

The redundancy criterion fixed for week 11 (an AUC above 0.9 on most runs) is
not met: on this evidence the per-point calls are not redundant with
continuity, except that for linear projections the two come close. Week 11
repeats the test on the benchmark with every metric.

### 9.9 Ranks or scores

Version 1 compares ranks, which treat a point far ahead of the next and a
point barely ahead alike (section 6). A score-based displacement would
compare the size of each point's score in the two spaces after putting both
on a common scale. `src/score_check.py` tests the two obvious common scales
on the points both spaces put in their top 5%, where a displacement that is
about the point, and not about the space, should be small.

- *The ratio to the median*, the candidate's own score, drifts with the
  dimension of the space. Under PCA the median ratio of the top 5% grows as
  dimensions are removed: wbc 5.1 at D = 9, 9.2 at d = 4, 20.1 at d = 2 and
  34.2 at d = 1; fault 1.9 at D = 27 and 4.1 at d = 2; magic 3.7 and 5.9;
  shuttle 3.0 and 9.7. At 2D the agreed points would show +0.8 to +2.1 bits
  of displacement from the change of dimension alone. That is the kNN
  distance growing like density to the power -1/d: the same density contrast
  is a ratio of 1.29 in nine dimensions and 3.16 in two.
- *Gaussian scaling* (Kriegel, Kröger, Schubert and Zimek, SDM 2011)
  saturates. On wbc, ionosphere and magic the top 5% sits at 0.98 to 1.00 in
  both spaces, so no movement within the top can show. On shuttle the scale
  drifts instead: the top 5% sits at 0.06 in the original space and at 0.39
  in the PCA picture.

Ranks have neither problem, so they stay (`docs/QUESTIONS.md`, 6).

### 9.10 PCA dimension and budget, with the candidate

Both sweeps repeat week 4's (5.3, 5.4) with the candidate's reading.

- Failures fall with the PCA dimension and vanish by full dimension on all
  five datasets: shuttle 0.47 at d = 2 and 0 from d = 6; magic 0 from d = 8;
  fault 0.01 at d = 16 and 0 from d = 20; ionosphere 0.22 at d = 20 and 0 from
  d = 24. On wbc one point of twelve comes and goes (0.08 at d = 2, 4 and 5; 0
  at 3 and from 6).
- UMAP masks at least as much as PCA at every budget on every dataset, as
  under version 1.

### 9.11 Expected against observed, and the decision

| | expectation | observed |
|---|---|---|
| E1 | identity and rotation move nothing | held |
| E2 | linear methods within 0.10 of version 1 | missed: 9 of 10; shuttle PCA moves 0.101 |
| E3 | the candidate masks less on wbc and fault, still most on the other three | held, except t-SNE on fault (up from 0.69 to 0.77) |
| E4 | neighbour-based above both linear methods on 3 of 5 | held: 3 of 5 (version 1: 4) |
| E5 | less sensitive to the scales than version 1 to k | held: 89% of runs within 0.10, against 71% |
| E6 | LOF orders the methods as kNN does | not held: rank correlation +0.85, -0.43, +0.26 |
| E7 | seeds: rates within 0.05 on the large datasets; candidate as stable as version 1 | not held: spreads up to 0.15; Jaccard 0.056 lower |
| E8 | continuity separates masked from retained only moderately | held for neighbour-based methods, not for linear ones |

Recorded before any outlier score was computed on benchmark data.

The candidate passes every known-answer check (56 of 56) and E5, and misses
E2 and E7, both narrowly: shuttle under PCA moves from 0.569 to 0.468, 0.101
against a bound of 0.10 (the other nine linear runs move by 0.083 at most);
and the masked sets agree across seeds slightly less often under v2, mean
Jaccard 0.741 against 0.797, 0.056 lower against a tolerance of 0.05.
**Following the rule, version 1 stays the definition.** The multi-scale
reading runs beside it on the benchmark as a secondary analysis, declared
before the run, so the decision can be reversed at no cost.

What the candidate is still good for: it reads the clumps the neighbour-based
methods make as a person might, it depends less on its own parameters than
version 1 does on k, and it is cheap to carry. The benchmark reports it beside
version 1 (`docs/BENCHMARK.md`), and question 7 of `docs/QUESTIONS.md` asks
the supervisor to settle it.

## Changelog

- **v1, 16 October 2026.** Replaces the September placeholder,
  `(rank_LD - rank_HD) / n` with a hard top-k cut at the anomaly count. Changes:
  displacement in bits; average ranks with tolerance for ties; a per-point
  noise null; masked and phantom called separately; chance-adjusted overlap;
  budget of the top 5%.
- **v1, review before delivery, 16 October 2026.** No change to the
  definition or the code of the measure. Reproduced from a clean clone with
  empty caches. Added: section 5.9 and `src/lattice_check.py` (lattice data);
  the relation to a top-m overlap (section 6); the first-order argument for
  the half-sampling null (section 3); question 6; `docs/week4.tex`. Corrected:
  PaCMAP is close to chance on all five datasets, not at chance on all five.
- **v1 kept, week 5, 16 October 2026.** No change to the definition. A
  multi-scale reading (candidate version 2) was tested against adoption
  criteria fixed beforehand and missed two of four, narrowly; it is reported
  beside version 1 (section 9). Added: TriMap as a sixth method; five seeds
  on the development set; the code generalised to any detector read off the
  neighbour graph and to several scales (`src/detectors.py`), with version 1
  reproduced bit for bit; LOF; per-point trustworthiness and continuity
  (`src/quality.py`); section I of `validate.py` (56 checks). Ranks kept over
  scores (9.9).
