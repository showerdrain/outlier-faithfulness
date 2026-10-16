# Outlier detection on low-dimensional projections

MSc thesis, Data Science (ICT Systems), SDU Odense.
Supervisor: Arthur Zimek. Co-supervisor: Muhammad Rajabinasab.

## Layout

    src/            code
    docs/LOG.md     dated record of decisions and the evidence for each
    docs/MEASURE.md the faithfulness measure: definition, reasons, test results
    docs/week4.tex  the same definition and main results in LaTeX, for Overleaf
    data/raw/       downloaded datasets, untouched
    data/processed/ loader output in a uniform format
    cache/          computed artifacts, chiefly kNN graphs
    results/        numeric output
    figures/        plots

Everything under data/, cache/, results/ and figures/ is gitignored, with
exceptions that record which data the thesis uses and what the measure was
tested against: results/dataset_table.json, results/screen_table.json,
results/splits.json, and the week-4 results results/week4_measure.json,
results/week4_nulls.json, results/week4_scale.json and
results/week4_lattice.json (UMAP and t-SNE do not reproduce bit for bit
across machines, so the numbers the definition was judged on are kept). The
repo reconstructs everything else from a clean checkout.

## Setup (Windows, PowerShell)

    python -m venv .venv          # Python 3.11, not newer: numba lags behind
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.lock.txt
    python src\smoke_test.py

## Scripts

| script | what it does |
|---|---|
| `datasets.py` | loads any of the 47 ADBench datasets in one uniform form; writes the provenance table |
| `neighbors.py` | nearest-neighbour graph per dataset (cached), kNN outlier score, per-point LID |
| `projections.py` | projections to low dimension (cached), and scoring in the projected space |
| `measure.py` | the outlier-faithfulness measure, version 1; definition in `docs/MEASURE.md` |
| `validate.py` | **run after any change to the four files above.** 44 checks with known answers |
| `explore_measure.py` | week 4: tests the measure on the development datasets, and refuses any other |
| `figures_week4.py` | figures for the week-4 tests (`explore_measure.py --figures` redraws them) |
| `scale_check.py` | week 4: how the verdicts depend on the neighbourhood scale used to read the projection |
| `lattice_check.py` | week 4: integer-valued features, tied scores and shattered pictures in the development datasets |
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
  reads the list from there and refuses anything else. The proposal's plan
  calls these five the exploration set.
- Ranks with ties use `measure.desc_ranks()`: tolerance-based, average ranks.
  Never `argsort` for a shortlist; that is how the cut became machine-dependent.
