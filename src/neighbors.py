"""Nearest-neighbour graph, computed once per dataset and cached.

Everything downstream wants the same neighbour graph: the distance-based
detectors, the local intrinsic dimensionality estimates, and later the
projection quality metrics. Computing it three times is the difference
between a comfortable November and a miserable one, so it is computed once
and read from disk after that.

The cache key includes meta['data_hash'] from the loader, so changing the
subsample seed, the standardisation flag or the duplicate policy produces a
new entry rather than a silent stale hit.

Graphs are stored at K_CACHE neighbours and sliced down on request, so a
detector at k=20 and another at k=50 both read the same file.

ONE CODE PATH FOR EVERY SPACE. The original space and every projection are
measured by the same function, knn_graph(), at the same float64 precision.
Until October 2026 the cache stored float32 distances while projections were
scored in float64. The identity check in validate.py showed the cost: data
"projected" onto itself moved 29 of 223 points on wbc, because rounding
created ties on one side that the other side broke. Cache files now carry a
version tag, so an old float32 file can never be mistaken for a current one.

    python src/neighbors.py            # build for the default shortlist
    python src/neighbors.py cardio wbc # build for named datasets
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from config import CACHE, DEFAULT_K
from datasets import load_dataset

K_CACHE = 100          # neighbours stored; requests for k <= this are sliced
METRIC = "euclidean"
CACHE_VERSION = "v2"   # v2: float64 distances, self excluded by index


def _cache_path(meta: dict) -> "object":
    return CACHE / (f"knn_{meta['name']}_{meta['data_hash']}_K{K_CACHE}_"
                    f"{METRIC}_{CACHE_VERSION}.npz")


def knn_graph(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k nearest neighbours of every point, the point itself excluded.

    Used for the original space AND for every projection, so the two sides
    of a comparison are computed by literally the same code.

    Self is excluded by calling kneighbors() with no argument, which drops
    each point by its index. The older approach, querying the data against
    itself and dropping column 0, assumes the point is always its own first
    neighbour; that fails silently when two points coincide, which happens
    in projections even after duplicates are removed from the original data.
    """
    nn = NearestNeighbors(n_neighbors=k, metric=METRIC).fit(X)
    dist, idx = nn.kneighbors()
    return dist.astype(np.float64), idx.astype(np.int32)


def knn_to_reference(X: np.ndarray, ref: np.ndarray, k: int,
                     return_idx: bool = False):
    """Distances from EVERY point to its k nearest neighbours among X[ref].

    Used by the noise null in measure.py, which scores all points against a
    random half of the data. Same metric and float64 precision as
    knn_graph(). Points inside the reference set are queried with
    kneighbors() and no argument, which excludes each point by its index;
    points outside it are queried against the reference set directly.

    With return_idx, also the neighbours' indices, in the numbering of the
    full X, so detectors that need a neighbour's own neighbourhood (LOF)
    can read it from that neighbour's row.
    """
    ref = np.asarray(ref, dtype=bool)
    nn = NearestNeighbors(n_neighbors=k, metric=METRIC).fit(X[ref])
    dist = np.empty((len(X), k), dtype=np.float64)
    idx = np.empty((len(X), k), dtype=np.int64)
    dist[ref], idx[ref] = nn.kneighbors()
    if (~ref).any():
        dist[~ref], idx[~ref] = nn.kneighbors(X[~ref])
    if not return_idx:
        return dist
    return dist, np.flatnonzero(ref)[idx].astype(np.int32)


def get_knn(
    name: str,
    k: int = DEFAULT_K,
    rebuild: bool = False,
    verbose: bool = True,
    **loader_kw,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (distances, indices, meta) for the k nearest neighbours.

    Self is excluded, so column 0 is the first true neighbour and column
    k-1 is the k-th. Distances are float64, matching the projected side;
    indices are int32. Very small datasets store n-1 neighbours instead of
    K_CACHE, since a point cannot have more neighbours than that.

    Any keyword accepted by load_dataset passes straight through, and
    affects the cache key via the data hash.
    """
    X, y, meta = load_dataset(name, **loader_kw)
    k_store = min(K_CACHE, meta["n"] - 1)
    if k > k_store:
        raise ValueError(f"k={k} exceeds the {k_store} neighbours stored for "
                         f"{name} (n={meta['n']}, K_CACHE={K_CACHE})")
    path = _cache_path(meta)

    if path.exists() and not rebuild:
        z = np.load(path)
        dist, idx = z["dist"], z["idx"]
    else:
        if verbose:
            print(f"  computing {k_store}-NN graph for {name} "
                  f"(n={meta['n']}, D={meta['D']}) ...", flush=True)
        t0 = time.time()
        dist, idx = knn_graph(X, k_store)
        elapsed = time.time() - t0

        np.savez_compressed(path, dist=dist, idx=idx)
        path.with_suffix(".json").write_text(json.dumps(
            {**meta, "K_CACHE": k_store, "metric": METRIC,
             "cache_version": CACHE_VERSION, "dtype": "float64",
             "build_seconds": round(elapsed, 2),
             "n_zero_first_distances": int((dist[:, 0] == 0).sum())},
            indent=2))
        if verbose:
            print(f"    done in {elapsed:.1f}s, "
                  f"{path.stat().st_size / 1e6:.0f} MB", flush=True)

    return dist[:, :k], idx[:, :k], meta


def knn_score(dist: np.ndarray, k: int | None = None) -> np.ndarray:
    """Distance to the k-th nearest neighbour. Larger means more outlying.

    Verified to match pyod.models.knn.KNN(method='largest') exactly.
    """
    k = dist.shape[1] if k is None else k
    return dist[:, k - 1].astype(np.float64)


def lid_mle(dist: np.ndarray, k: int | None = None) -> np.ndarray:
    """Per-point local intrinsic dimensionality, maximum likelihood estimator.

    The Hill / Levina-Bickel form used by Amsaleg et al. (2015):

        LID_i = - [ (1/(k-1)) * sum_{j=1..k-1} log( r_j / r_k ) ] ^ -1

    Note the (k-1) denominator rather than k. That bias correction is what
    scikit-dimension uses, and getting it wrong rescales every estimate by
    k/(k-1) while leaving the ranking untouched, so it is easy to miss.

    Points whose k-th neighbour sits at distance zero return nan rather than
    silently producing inf. After duplicate removal there should be none;
    if any appear, the neighbourhood has collapsed and the point needs
    looking at rather than averaging over.
    """
    k = dist.shape[1] if k is None else k
    r = dist[:, :k].astype(np.float64)
    r_k = r[:, [k - 1]]

    bad = (r_k[:, 0] <= 0) | (r[:, 0] <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log(r[:, :k - 1] / r_k).sum(axis=1) / (k - 1)
        lid = -1.0 / s
    lid[bad] = np.nan
    return lid


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "wbc", "ionosphere", "cardio", "musk",
        "satimage2", "satellite", "shuttle", "backdoor",
    ]
    print()
    hdr = f"{'dataset':<12}{'n':>7}{'D':>5}{'kNN AUC':>9}{'LID med':>9}{'LID p90':>9}{'nan':>5}"
    print(hdr); print("-" * len(hdr))
    from sklearn.metrics import roc_auc_score
    for name in names:
        dist, idx, meta = get_knn(name, k=DEFAULT_K)
        _, y, _ = load_dataset(name)
        s = knn_score(dist)
        lid = lid_mle(dist)
        auc = roc_auc_score(y, s)
        print(f"{name:<12}{meta['n']:>7}{meta['D']:>5}{auc:>9.3f}"
              f"{np.nanmedian(lid):>9.2f}{np.nanpercentile(lid, 90):>9.2f}"
              f"{int(np.isnan(lid).sum()):>5}")
