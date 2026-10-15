# Outlier detection on low-dimensional projections

MSc thesis, Data Science (ICT Systems), SDU Odense.
Supervisor: Arthur Zimek. Co-supervisor: Muhammad Rajabinasab.

## Layout

    src/            code
    docs/LOG.md     dated record of decisions and the evidence for each
    data/raw/       downloaded datasets, untouched
    data/processed/ loader output in a uniform format
    cache/          computed artifacts, chiefly kNN graphs
    results/        numeric output
    figures/        plots

Everything under data/, cache/, results/ and figures/ is gitignored, with
three exceptions that record which data the thesis uses and why:
results/dataset_table.json, results/screen_table.json, results/splits.json.
The repo reconstructs everything else from a clean checkout.

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
| `measure.py` | placeholder faithfulness measure: per-point displacement, shortlist table and figure |
| `validate.py` | **run after any change to the four files above.** 31 checks with known answers |
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
  (`results/splits.json`), never on benchmark datasets.
