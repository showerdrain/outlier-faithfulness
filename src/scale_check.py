"""Scale sensitivity of the projection-side reading (week 4 follow-up).

    python src/scale_check.py

v1 reads 'outlying in the projection' with the same detector at the same k as
the original space. UMAP and PaCMAP pull isolated outliers together into tight
micro-clusters, which a kNN score reads as dense when k is small relative to the
clump. This re-scores the 2D pictures (seed 0) at several k:
  - projection side only (k_LD = 5 .. 100, original side fixed at k = 20)
  - both sides at k = 50, with its own noise null
Development datasets only. Writes results/week4_scale.json (tracked).
"""

from __future__ import annotations

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import measure as M
from config import RESULTS
from explore_measure import LOAD, dev_names, refuse_benchmark
from datasets import load_dataset
from neighbors import get_knn, knn_score
from projections import knn_dist, project

K_LD = (5, 10, 20, 50, 100)
METHODS = ("pca", "random", "umap", "tsne", "pacmap")


def main() -> None:
    names = dev_names()
    refuse_benchmark(names)
    rows = []
    for name in names:
        dist, _, _ = get_knn(name, k=100, verbose=False, **LOAD)
        a20, a50 = knn_score(dist, 20), knn_score(dist, 50)
        L20, _ = M.get_null(name, 20, verbose=False, **LOAD)
        L50, _ = M.get_null(name, 50, **LOAD)
        for method in METHODS:
            Y, _ = project(name, method=method, n_components=2, seed=0, verbose=False, **LOAD)
            r = {"dataset": name, "method": method}
            for k in K_LD:
                s = M.summarise(a20, knn_score(knn_dist(Y, k), k), L20)
                r[f"MR_hd20_ld{k}"] = round(s["masking_rate"], 4)
                r[f"AO_hd20_ld{k}"] = round(s["adjusted_overlap"], 4)
            s = M.summarise(a50, knn_score(knn_dist(Y, 50), 50), L50)
            r["MR_both50"] = round(s["masking_rate"], 4)
            r["AO_both50"] = round(s["adjusted_overlap"], 4)
            rows.append(r)
            print(f"{name:<11}{method:<8} MR at k_LD " +
                  "  ".join(f"{k}:{r[f'MR_hd20_ld{k}']:.2f}" for k in K_LD) +
                  f"   | both 50: {r['MR_both50']:.2f}", flush=True)
    (RESULTS / "week4_scale.json").write_text(json.dumps(rows, indent=1))
    print(f"\nWrote {RESULTS / 'week4_scale.json'}")


if __name__ == "__main__":
    main()
