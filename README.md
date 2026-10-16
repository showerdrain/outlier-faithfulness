# Outlier detection on low-dimensional projections

MSc thesis, Data Science (ICT Systems), SDU Odense.
Supervisor: Arthur Zimek. Co-supervisor: Muhammad Rajabinasab.

## Layout

    src/            code
    docs/LOG.md     dated record of decisions and the evidence for each
    docs/MEASURE.md the faithfulness measure: definition, reasons, test results
    docs/BENCHMARK.md week 6: the benchmark in two dimensions, protocol and results
    docs/QUESTIONS.md open questions for the supervisor, each with a provisional decision
    docs/week4.tex  the week-4 definition and main results in LaTeX, for Overleaf
    docs/week5_6.tex  week 5 and the week-6 benchmark in LaTeX, for Overleaf
    data/raw/       downloaded datasets, untouched
    data/processed/ loader output in a uniform format
    cache/          computed artifacts, chiefly kNN graphs
    results/        numeric output
    figures/        plots

Everything under data/, cache/, results/ and figures/ is gitignored, with
exceptions that record which data the thesis uses and what the measure was
tested against: results/dataset_table.json, results/screen_table.json,
results/splits.json, the week-4 results (results/week4_measure.json,
results/week4_nulls.json, results/week4_scale.json,
results/week4_lattice.json), the week-5 results
(results/week5_revision.json, results/week5_redundancy_v1.json,
results/week5_score_check.json) and the week-6
results (results/week6_benchmark.json, results/week6_stats.json). UMAP,
t-SNE and TriMap do not reproduce bit for bit across machines, so the numbers
the text cites are kept. The repo reconstructs everything else from a clean
checkout.

## Setup (Windows, PowerShell)

    python -m venv .venv          # Python 3.11, not newer: numba lags behind
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.lock.txt
    python src\smoke_test.py

TriMap 1.1.4 needs annoy, which has no Windows wheel: pip compiles it, which
needs Microsoft C++ Build Tools ("Desktop development with C++"). TriMap also
imports pkg_resources, so setuptools must stay below 82.

## Scripts

| script | what it does |
|---|---|
| `datasets.py` | loads any of the 47 ADBench datasets in one uniform form; writes the provenance table |
| `neighbors.py` | nearest-neighbour graph per dataset (cached), kNN outlier score, per-point LID |
| `projections.py` | projections to low dimension (cached): PCA, random projection, UMAP, t-SNE, PaCMAP, TriMap |
| `detectors.py` | outlier detectors read off a neighbour graph (kNN, LOF), at one scale or several (the multi-scale candidate of week 5) |
| `measure.py` | the outlier-faithfulness measure, version 1; definition in `docs/MEASURE.md` |
| `quality.py` | per-point trustworthiness and continuity, for the redundancy question |
| `validate.py` | **run after any change to the files above.** checks with known answers |
| `explore_measure.py` | week 4: tests the measure on the development datasets, and refuses any other |
| `figures_week4.py` | figures for the week-4 tests (`explore_measure.py --figures` redraws them) |
| `scale_check.py` | week 4: how the verdicts depend on the neighbourhood scale used to read the projection |
| `lattice_check.py` | week 4: integer-valued features, tied scores and shattered pictures in the development datasets |
| `revise_measure.py` | week 5: sanity checks, version 1 against the multi-scale candidate, on the development datasets only |
| `score_check.py` | week 5: whether normalised scores could replace ranks |
| `figures_week5.py` | figures for week 5 (`revise_measure.py --figures` redraws them) |
| `benchmark.py` | week 6: the benchmark in 2D, in stages (embed, run, report, figures); refuses development datasets |
| `figures_week6.py` | figures for week 6 (`benchmark.py figures`) |
| `screen.py` | screens all datasets against stated criteria; records the development/benchmark split |
| `smoke_test.py` | confirms every library installs and runs |

## Conventions

- Paths come from `src/config.py`. No module builds its own.
- Every stochastic call takes an explicit seed, recorded in the output filename.
- **One code path for every space.** The original space and every projection
  get their neighbours from `knn_graph()` in `neighbors.py`, at float64.
  Scoring the two sides differently creates displacement that is pure
  numerical artefact; `validate.py` section C exists to catch exactly that.
- The original-space graph is computed once per (dataset, k, metric) and
  cached under a content hash of the data, so changing a loading option can
  never return neighbours computed from different data.
- Compare floats with a tolerance, never `==`. Exact equality has already
  given different answers on Windows and Linux three times.
- Measure-design decisions are made on the development datasets only
  (`results/splits.json`), never on benchmark datasets. `explore_measure.py`
  and `revise_measure.py` read the list from there and refuse anything else;
  `benchmark.py` refuses the development datasets. The proposal's plan calls
  these five the exploration set.
- Ranks with ties use `measure.desc_ranks()`: tolerance-based, average ranks.
  Never `argsort` for a shortlist; that is how the cut became machine-dependent.
