# Outlier detection on low-dimensional projections

MSc thesis, Data Science (ICT Systems), SDU Odense.
Supervisor: Arthur Zimek. Co-supervisor: Muhammad Rajabinasab.

## Layout

    src/            code
    data/raw/       downloaded datasets, untouched
    data/processed/ loader output in a uniform format
    cache/          computed artifacts, chiefly kNN graphs
    results/        numeric output
    figures/        plots

Everything under data/, cache/, results/ and figures/ is gitignored.
The repo should reconstruct all of it from a clean checkout.

## Setup

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/config.py

## Conventions

- Paths come from `src/config.py`. No module builds its own.
- Every stochastic call takes an explicit seed, recorded in the output filename.
- The high-dimensional kNN graph is computed once per (dataset, k, metric)
  and cached. Detectors, LID estimation and the quality metrics all read it.
