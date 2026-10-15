# Research log

Dated record of decisions about method and data, and the evidence for each.
Newest first. When a later entry supersedes an earlier one, the earlier entry
is marked, not deleted: the method chapter needs the path as well as the end.

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
- **Cross-platform reproducibility.** Loading, neighbour graphs, kNN scores,
  LID and PCA reproduce bit for bit between Windows and Linux. UMAP and t-SNE
  do not (continuity 0.747 vs 0.767 on the same toy data, same seed). Method
  differences smaller than about 0.02 cannot be called real.
- **trimap not installed.** Version 1.2.0 requires torch and annoy; install
  separately when the full benchmark needs it.
