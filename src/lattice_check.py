"""Lattice data among the development datasets (week 4 follow-up).

    python src/lattice_check.py

Two of the five development datasets have only integer-valued features. That
matters twice over. kNN scores tie, because distances on a grid repeat. And
the 2D pictures break into many small islands, because tight groups of
near-identical points become separate neighbourhoods. This records both, per
dataset, so the statements in docs/MEASURE.md (section 5.9) rest on a file:

  integer_features   features whose every value is a whole number (raw data)
  distinct_per_feature   median number of distinct values per feature (raw data)
  distinct_scores    number of distinct kNN scores (k = 20) as a share of n,
                     ties judged with measure.TIE_RTOL
  islands_<method>   strongly connected components of the directed 10-NN
                     graph of the 2D picture (seed 0), and the share of points
                     in the largest one. A component is a group of points
                     whose nearest neighbours all lie inside the group.

Development datasets only. Writes results/week4_lattice.json (tracked).
"""

from __future__ import annotations

import json
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

import measure as M
from config import DATA_RAW, RESULTS
from datasets import REGISTRY, load_dataset
from explore_measure import K, LOAD, dev_names, refuse_benchmark
from neighbors import get_knn, knn_score
from projections import project

METHODS = ("pca", "random", "umap", "tsne", "pacmap")
K_ISLAND = 10


def islands(Y: np.ndarray) -> tuple[int, float]:
    G = kneighbors_graph(Y, K_ISLAND)
    n_comp, label = connected_components(G, directed=True, connection="strong")
    return int(n_comp), float(np.bincount(label).max() / len(Y))


def main() -> None:
    names = dev_names()
    refuse_benchmark(names)
    rows = []
    for name in names:
        raw = np.load(DATA_RAW / f"{REGISTRY[name].file}.npz", allow_pickle=True)
        X_raw = np.asarray(raw["X"], dtype=np.float64)
        integer = np.all(np.isclose(X_raw, np.round(X_raw)), axis=0)
        n_values = [len(np.unique(X_raw[:, j])) for j in range(X_raw.shape[1])]

        X, _, meta = load_dataset(name, **LOAD)
        dist, _, _ = get_knn(name, k=K, verbose=False, **LOAD)
        a = knn_score(dist, K)
        r = {"dataset": name, "n": int(meta["n"]), "D": int(meta["D"]),
             "integer_features": int(integer.sum()),
             "distinct_per_feature_median": float(np.median(n_values)),
             "distinct_scores_share": round(M.n_distinct(a) / len(a), 4)}
        for method in METHODS:
            Y, _ = project(name, method=method, n_components=2, seed=0,
                           verbose=False, **LOAD)
            n_comp, largest = islands(Y)
            r[f"islands_{method}"] = n_comp
            r[f"largest_island_share_{method}"] = round(largest, 4)
        rows.append(r)
        print(f"{name:<11} integer features {r['integer_features']}/{r['D']}, "
              f"distinct scores {100 * r['distinct_scores_share']:.0f}%, islands " +
              ", ".join(f"{m} {r[f'islands_{m}']}" for m in METHODS), flush=True)
    (RESULTS / "week4_lattice.json").write_text(json.dumps(rows, indent=1))
    print(f"\nWrote {RESULTS / 'week4_lattice.json'}")


if __name__ == "__main__":
    main()
