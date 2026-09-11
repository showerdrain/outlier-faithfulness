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


def _cache_path(meta: dict) -> "object":
    return CACHE / f"knn_{meta['name']}_{meta['data_hash']}_K{K_CACHE}_{METRIC}.npz"


def get_knn(
    name: str,
    k: int = DEFAULT_K,
    rebuild: bool = False,
    verbose: bool = True,
    **loader_kw,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (distances, indices, meta) for the k nearest neighbours.

    Self is excluded, so column 0 is the first true neighbour and column
    k-1 is the k-th. Distances are float32 and indices int32: ample
    precision for distances, and half the disk of float64/int64.

    Any keyword accepted by load_dataset passes straight through, and
    affects the cache key via the data hash.
    """
    if k > K_CACHE:
        raise ValueError(f"k={k} exceeds K_CACHE={K_CACHE}; raise K_CACHE and rebuild")

    X, y, meta = load_dataset(name, **loader_kw)
    path = _cache_path(meta)

    if path.exists() and not rebuild:
        z = np.load(path)
        dist, idx = z["dist"], z["idx"]
    else:
        if verbose:
            print(f"  computing {K_CACHE}-NN graph for {name} "
                  f"(n={meta['n']}, D={meta['D']}) ...", flush=True)
        t0 = time.time()
        nn = NearestNeighbors(n_neighbors=K_CACHE + 1, metric=METRIC).fit(X)
        dist, idx = nn.kneighbors(X)
        dist, idx = dist[:, 1:], idx[:, 1:]        # drop self
        dist = dist.astype(np.float32)
        idx = idx.astype(np.int32)
        elapsed = time.time() - t0

        np.savez_compressed(path, dist=dist, idx=idx)
        path.with_suffix(".json").write_text(json.dumps(
            {**meta, "K_CACHE": K_CACHE, "metric": METRIC,
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
