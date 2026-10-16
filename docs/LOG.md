# Research log

Dated record of decisions about method and data, and the evidence for each.
Newest first. When a later entry supersedes an earlier one, the earlier entry
is marked, not deleted: the method chapter needs the path as well as the end.

---

## 2026-10-16 · Measure v1 reviewed before delivery

**Why.** The v1 work below was written in a session that ended before it was
delivered. It was reviewed afresh and re-run before reaching the repository.

**Reproduced.** On a fresh clone of the GitHub repository with empty caches:
44 of 44 checks pass, and `explore_measure.py` and `scale_check.py` rewrite
the committed result files unchanged.

**Found.**
- *Two of the five development datasets are lattice data.* Every feature of
  shuttle and wbc is a whole number. Shuttle's 20,000 points take only 8,351
  distinct kNN scores, and its 2D pictures under PCA, UMAP, t-SNE and PaCMAP
  each break into more than a thousand islands of near-identical points (169
  under a random projection); magic gives 5 to 65. Shuttle's UMAP, t-SNE and
  PaCMAP results describe those islands as much as the methods. Network data
  has many discrete fields, so this will return. `src/lattice_check.py`,
  `results/week4_lattice.json`, MEASURE.md 5.9.
- *At 2D the masking rate is close to the plain shortlist exit rate* (0.02
  below it on average), because nearly every crossing is far beyond noise.
  The null matters near the boundary and in nearly faithful projections.
  Stated in MEASURE.md section 6, so the non-redundancy argument is not
  overstated.
- *Wording corrected:* PaCMAP is close to chance on all five datasets, but
  significantly above it on shuttle and magic (AO up to 0.09).

**Added for the supervisor:** question 6, lattice data. The definition and
the code of the measure are unchanged.

---

## 2026-10-16 · Measure v1 defined and tested on the development set

**Why.** Week 4 of the plan: first definition of the measure, tested on the
exploration set. The placeholder had four known faults: no null model, a
shortlist cut that was arbitrary among tied scores, masked and phantom counts
equal by arithmetic, and a displacement dominated by the bulk of inliers.

**Definition** (full argument in `docs/MEASURE.md`). Same detector on both
sides. Per point, displacement in bits, `log2(rank_original / rank_projection)`,
with average ranks and tolerance-based ties. A shortlist crossing counts as a
failure only if it lies beyond a per-point noise null: 100 re-rankings of the
original space, each scoring every point against a random half of the data
with k/2 neighbours. Masked and phantom calls are made separately. The
aggregates are masking rate, phantom rate and a chance-adjusted shortlist
overlap. Default budget: the top 5% of points. Labels are not used by the
measure, only to say what failures cost.

**Decisions, each with its evidence.**
- *Log-rank displacement.* A move from rank 1 to 64 is -6 bits; 1001 to 1064
  is -0.09. The placeholder scored them the same.
- *Ties resolved by the data.* A tied group is wholly in or out of a shortlist.
  Replaces the stable sort.
- *Budget top 5%, not the anomaly count.* Three of five development datasets
  have about 35% anomalies. The anomaly-count budget is still reported.
- *Resampling null, not the scDEED-style permutation null.* The permutation
  version calls almost nothing (1 of 18 exits on ionosphere, where PCA's 2D
  shortlist shares nothing with the original's), because on structureless data
  every projection destroys the ranking. The resampling null is calibrated
  (2-3% false calls when the null holds, at alpha 5%), stable to its own seed
  (masked sets agree 97-100%), and agrees with a k-variation null (93-96%).
- *Permutation kept as the chance level* of the aggregate (adjusted overlap, 0
  at chance), in the form of the Campos et al. (2016) adjustment.

**Checks.** `validate.py` now has 44, all passing, including the new section
H: rotation leaves everything unmoved, antisymmetry, monotone invariance,
chance level, calibration of the null, and the planted outliers (17-20 of 20
hidden ones called masked, all 20 visible ones retained).

**Results** (2D, top 5%; `results/week4_*.json`, `figures/week4_*.png`):
- PCA masks 0% (wbc), 39% (magic), 57% (shuttle), 58% (fault), 100%
  (ionosphere). Masking falls with target dimension and reaches zero by full
  dimension on all five.
- UMAP masks 61-100%, t-SNE 70-93%, PaCMAP 85-100%: more than PCA or a random
  projection on four datasets of five. UMAP and PaCMAP are at chance level on
  several.
- Aggregate correlations miss it. On ionosphere, PCA keeps Spearman 0.56 and
  the Schubert et al. weighted correlation 0.55 while its shortlist shares
  nothing with the original's.
- The null matters at the boundary. With PCA to 6D on shuttle, 46 points
  still leave the shortlist but all are within noise, so masking is 0. The
  placeholder would have counted 46 failures.

**Corrections to earlier beliefs.**
- *Shuttle is not a control for PCA.* PCA to 2D hides 57% of its top 5%. The
  September "nothing is lost" rested on AUC retained and Spearman, both
  bulk-dominated. Its median LID of about 2 is local; globally it spans six
  linear dimensions (PCA needs six components for all the variance), and
  masking only vanishes at d = 6.
- *wbc is a clean control only for PCA.* t-SNE and PaCMAP mask 81% and 97%.

**Found along the way.** The neighbour-based methods gather isolated outliers
into tight clumps. Read at k = 20 the clumps look dense, so their points count
as masked. Read at k = 50, wbc's clumped outliers are back near the top. On
shuttle, ionosphere and magic the UMAP, t-SNE and PaCMAP verdicts hold at every
scale from k = 5 to 100. On wbc and fault they depend on the scale. Here the
projection itself creates clustered anomalies, the case the September notes
flagged as the domain's sharpest question. `src/scale_check.py`,
`results/week4_scale.json`.

**Open, for the supervisor.**
1. Resampling null per point, permutation only for the chance level. Acceptable?
2. Budget: top 5% instead of the anomaly count?
3. How to read the projection: same k (the proposal's rule, v1), several
   scales reported, or a multi-scale score? Decides how the UMAP result is
   stated.
4. Keep shuttle as the case where local and global dimension differ?
5. Ranks (v1) or normalised scores (Kriegel et al. 2011) for v2?

---

## 2026-10-15 · Dataset screen and proposed split

**Status: proposed, pending supervisor confirmation.** Record in
`results/splits.json`; full statistics in `results/screen_table.json`.

**Why.** The supervisor's instruction was to drop synthetic data and select
real datasets with promising LID and outlierness profiles as exploration
data. That needed a stated, reproducible selection rule, and the benchmark
needed enough independent datasets for a cross-dataset test (Friedman's test
has little power below about ten).

**What was screened.** All 47 ADBench Classical datasets, with `src/screen.py`.
An earlier ad hoc screen had probed guessed file names and silently missed two
(Wilt, WPBC); all 47 names are now confirmed. Statistics use at most 20,000
points per dataset (stratified, seed 0, after duplicate removal).

**Criteria.** D >= 10, kNN AUC >= 0.65, anomaly rate <= 25%, n >= 200.
Robustness: moving the rate threshold anywhere from 15% to 30% changes
nothing; moving AUC from 0.65 to 0.70 changes nothing. D and n move the
margins only (D >= 8 adds shuttle, stamps, wbc; dropping the n criterion adds
hepatitis, lymphography, wine; n >= 500 removes wdbc).

**Benchmark, 15 datasets.** backdoor, campaign, cardio, celeba, cover, fraud,
letter, mnist, musk, pageblocks, pendigits, satimage2, vowels, waveform,
wdbc. D spans 10 to 196; median LID spans 3.6 (pageblocks) to 14.6
(waveform).

**Development, 5 datasets, all excluded by the screen anyway** (so the
benchmark loses nothing): shuttle (low-LID control, tie test bed), wbc (clean
negative control), ionosphere (heaviest LID tail; PCA's worst case), fault
(27 features but median LID 4.8), magic (about 19,000 points; tests cost).
Measure-design decisions are made on these only.

**Case study.** backdoor, the only high-dimensional network-intrusion
dataset that passes, also analysed at full size.

**Source families.** Versions of one underlying dataset may not sit on both
sides of the split, and count once in the benchmark. Confirmed from the data:
satimage2 and wbc are exact row subsets of satellite and breastw; landsat is
satellite rescaled (identical correlation matrix). Documented source only:
cardio/cardiotocography, thyroid/annthyroid, wdbc/wpbc. This is why
satellite, an earlier development suggestion, is out: its rows contain
satimage2's. `propose_split()` refuses such choices; tested by trying to add
satellite, musk and optdigits, all three rejected.

**Found along the way.**
- *Duplicates.* donors is 94% exact duplicate rows (583,905 of 619,326), and
  removing them moves its anomaly rate from 5.9% to 21.6%: most duplicates
  were inliers. Also celeba 44%, breastw 34%, mammography 30%, census 25%.
- *Subsampling changes kNN AUC at fixed k.* backdoor scores 0.905 on 20,000
  points but 0.839 at full size (87,020): k = 20 is a smaller neighbourhood
  in a denser sample. A benchmark size cap must be uniform and recorded.
- *Anomalies less outlying than inliers.* Six datasets score below chance:
  skin 0.29, vertebral 0.34, optdigits 0.39, yeast 0.39, donors 0.43,
  speech 0.48. Their labels mark classes, not geometric outliers.
- *speech* has the highest LID in the collection (median 29) and no outlier
  structure at all (AUC 0.48). High LID alone does not make outliers.
- *census* needs 5.6 GB of memory to remove duplicates from 1.2 GB of data.

**Open, for the supervisor.**
1. Confirm the criteria and the split.
2. No development dataset has median LID above about 7; every high-LID
   dataset (waveform 14.6, satimage2 11.4, mnist 10.0) is in the benchmark.
   Option: move satimage2 to development, leaving a benchmark of 14.
3. Three development datasets have about 35% anomalies. For developing a
   top-rank measure, downsample the anomaly class (as Campos et al. 2016 do)?
   *2026-10-16:* the measure now uses a label-free budget (top 5%), which
   makes this less pressing for development.
4. A benchmark size cap (20,000 proposed) and whether k should scale with n.

---

## 2026-10-15 · Verification suite; precision asymmetry found and fixed

**Why.** Every result is a comparison between outlier scores in the original
space and in a projection. Before building the real measure on that
comparison, it needed checks whose right answer is known in advance.
`src/validate.py` has 31.

**Found.** The original-space neighbour cache stored distances as float32,
while projections were scored in float64. Section B showed original-space
k-th distances off by about 6e-8 relative (float32 rounding exactly), and the
projected side exact. Section C showed the consequence: data "projected" onto
itself, which must change nothing, moved 29 of 223 points on wbc, 10 on
satellite, 2 on cardio and 2 on satimage2. Rounding created ties on one side
that the other side broke. No shortlist changed.

**Changed.**
- One function, `knn_graph()`, computes neighbours for every space, at float64.
- Self excluded by index (`kneighbors()` with no argument), not by dropping
  column 0. The old way assumes a point is its own first neighbour, which
  fails when points coincide. It never misfired on the PCA projections (no
  coincident points), but UMAP and t-SNE are more prone to stacking points.
- Cache files carry a version tag (`_v2`), so a float32 file is never reused.
- Tie diagnostics compare with a relative tolerance of 1e-9, never `==`.
- Datasets with n <= 100 store n-1 neighbours instead of failing.

**Effect on results.** None at reported precision. Sixteen values moved, all
in the fourth decimal place or beyond. Distinct-score counts rose: float32 had
manufactured 282 false ties on backdoor (84,837 -> 85,119 distinct values).
Shuttle still has 4 points tied at the shortlist boundary; that is real.

**Also learned.**
- *Random projection beats PCA on detection for some datasets.* A random 2D
  projection keeps 92% of kNN AUC on ionosphere against PCA's 62%, and 102%
  on backdoor against 88%. This is not a bug. PCA discards low-variance
  directions by construction, so outliers whose outlyingness lives there are
  masked every time; a random projection keeps a share of every direction.
  PCA does preserve the *ranking* better on most datasets (rank correlation
  0.448 vs 0.195 on musk), ionosphere being the exception (0.564 vs 0.762).
- *Question 3's mechanism, in miniature.* In the planted-outlier test,
  outliers placed in PCA's discarded directions have median LID about 25,
  against about 5 for outliers in the kept plane and 8 for inliers. The
  points a projection cannot represent are the high-LID ones.
- The planted fixture is a unit test with a known answer. It is not the
  synthetic-data component dropped from the proposal, and no result uses it.

---

## 2026-09 · Decisions made while building the pipeline

- **Synthetic data dropped** (supervisor). Select real datasets by their LID
  and outlierness profiles instead. Question 3 survives because LID varies
  point by point within a dataset, so the regression runs within datasets.
  Lost: the ability to separate outlier type from LID cleanly.
- **Duplicate rows removed by default.** Zero distances break the MLE LID
  estimator outright (log of zero) and make LOF degenerate. Campos et al.
  (2016) ship with-and-without variants for the same reason. It changes
  datasets materially: http lost 61% of rows and its anomaly rate fell from
  0.39% to 0.03%.
- **Cache keyed on a content hash** of the loaded arrays, not the dataset
  name, so changing any loading option can never return stale neighbours.
- **LID estimator uses the k-1 denominator.** The first version used k: it
  correlated 1.0000 with scikit-dimension while every value was 5% too large.
  Validate by absolute value, never by correlation alone.
- **PCA with `svd_solver="full"`**, so the one deterministic method is exactly
  reproducible.
- **Stable sort for the top-k shortlist.** numpy's default sort is unstable
  and gave 2,043 vs 2,044 for the same shuttle shortlist on two machines. A
  stable sort makes the answer reproducible, not principled; the tie policy
  is still open (see Schubert, Wojdanowski, Zimek & Kriegel, SDM 2012).
  *Superseded 2026-10-16:* ties are now resolved by average ranks, so a tied
  group is wholly in or out of a shortlist.
- **Cross-platform reproducibility.** Loading, neighbour graphs, kNN scores,
  LID and PCA reproduce bit for bit between Windows and Linux. UMAP and t-SNE
  do not (continuity 0.747 vs 0.767 on the same toy data, same seed). Method
  differences smaller than about 0.02 cannot be called real.
- **trimap not installed.** Version 1.2.0 requires torch and annoy; install
  separately when the full benchmark needs it.
