"""Projections to low dimension, cached, plus scoring in the projected space.

The pipeline this completes:

    original space  ->  neighbour graph  ->  outlier score  ->  ranks
    projected space ->  neighbour graph  ->  outlier score  ->  ranks

Both sides use the same detector with the same k. Any difference in the
resulting ranks is the projection's doing, which is the whole point.

Embeddings are cached on (data_hash, method, params, n_components, seed).
Seeds are explicit everywhere and appear in the filename, because week 8 is
an analysis of seed variance and that is impossible if any seed was implicit.

    python src/projections.py           # PCA across the shortlist
    python src/projections.py umap      # a different method
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

from config import CACHE, RESULTS, DEFAULT_K, DEFAULT_SEED
from datasets import load_dataset
from neighbors import get_knn, knn_graph, knn_score


# --- the projection methods -------------------------------------------------
# Library defaults throughout, unless a params dict says otherwise; the
# hyperparameter grid is week 8. Every method takes an explicit seed.

def _pca(X, n_components, seed, **kw):
    # svd_solver="full" rather than the default: the randomised solver is
    # seed-dependent, and PCA is the one method here that should be exactly
    # reproducible. sklearn's svd_flip also fixes the sign ambiguity.
    return PCA(n_components=n_components, svd_solver="full").fit_transform(X)


def _umap(X, n_components, seed, n_neighbors=15, min_dist=0.1, **kw):
    import umap
    return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=seed).fit_transform(X)


def _tsne(X, n_components, seed, perplexity=30, **kw):
    from openTSNE import TSNE
    return np.asarray(TSNE(n_components=n_components, perplexity=perplexity,
                           random_state=seed, verbose=False).fit(X))


def _pacmap(X, n_components, seed, n_neighbors=10, **kw):
    import pacmap
    return pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors,
                         random_state=seed).fit_transform(X)


def _random(X, n_components, seed, **kw):
    # The uninformed baseline: a Gaussian random linear map. Its scaling is
    # irrelevant, since every comparison here is rank-based.
    from sklearn.random_projection import GaussianRandomProjection
    return GaussianRandomProjection(n_components=n_components,
                                    random_state=seed).fit_transform(X)


_seed_numba = None


def _trimap(X, n_components, seed, n_inliers=12, n_outliers=4, n_random=3, **kw):
    # TriMap 1.1.4 (Amid & Warmuth), installed without torch; 1.2.0 needs it.
    # It has no random_state. Triplets are drawn from numba's own random
    # generator inside parallel loops, so a fixed seed is reproducible only
    # with that generator seeded and the loops on one thread. Both are set
    # here and restored afterwards (checked: same seed, identical output).
    global _seed_numba
    import numba
    import trimap
    if _seed_numba is None:
        @numba.njit
        def _seed(s):
            np.random.seed(s)
        _seed_numba = _seed
    state, threads = np.random.get_state(), numba.get_num_threads()
    np.random.seed(seed)
    _seed_numba(seed)
    numba.set_num_threads(1)
    try:
        return trimap.TRIMAP(n_dims=n_components, n_inliers=n_inliers,
                             n_outliers=n_outliers, n_random=n_random,
                             verbose=False).fit_transform(np.asarray(X, dtype=np.float64))
    finally:
        numba.set_num_threads(threads)
        np.random.set_state(state)


METHODS = {"pca": _pca, "umap": _umap, "tsne": _tsne, "pacmap": _pacmap, "trimap": _trimap,
           "random": _random}


def project(
    name: str,
    method: str = "pca",
    n_components: int = 2,
    seed: int = DEFAULT_SEED,
    rebuild: bool = False,
    verbose: bool = True,
    params: dict | None = None,
    **loader_kw,
) -> tuple[np.ndarray, dict]:
    """Return (Y, meta) for one embedding. Computed once, then cached."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    params = params or {}

    X, y, meta = load_dataset(name, **loader_kw)
    tag = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    tag = f"_{tag}" if tag else ""
    path = CACHE / f"emb_{name}_{meta['data_hash']}_{method}{tag}_d{n_components}_s{seed}.npy"

    if path.exists() and not rebuild:
        Y = np.load(path)
    else:
        if verbose:
            print(f"  {method} -> {n_components}D on {name} "
                  f"(n={meta['n']}, D={meta['D']}, seed={seed}) ...", flush=True)
        t0 = time.time()
        Y = np.asarray(METHODS[method](X, n_components, seed, **params), dtype=np.float64)
        elapsed = time.time() - t0
        np.save(path, Y)
        path.with_suffix(".json").write_text(json.dumps(
            {**meta, "method": method, "params": params,
             "n_components": n_components, "seed": seed,
             "fit_seconds": round(elapsed, 2)}, indent=2))
        if verbose:
            print(f"    {elapsed:.1f}s", flush=True)

    emb_meta = {**meta, "method": method, "params": params,
                "n_components": n_components, "seed": seed}
    return Y, emb_meta


def knn_dist(Y: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Neighbour distances within a projected space. Self excluded.

    Goes through the same knn_graph() as the original space, so both sides
    of every comparison are computed identically. Not disk-cached: in 2D
    this is a kd-tree query taking well under a second even at n=87k.
    """
    return knn_graph(Y, k)[0]


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    method = sys.argv[1] if len(sys.argv) > 1 else "pca"
    names = ["wbc", "ionosphere", "cardio", "musk",
             "satimage2", "satellite", "shuttle", "backdoor"]
    k = DEFAULT_K
    rows = []

    for name in names:
        dist_hd, _, meta = get_knn(name, k=k, verbose=False)
        _, y, _ = load_dataset(name)
        s_hd = knn_score(dist_hd, k)

        Y, emb = project(name, method=method, n_components=2)
        s_ld = knn_score(knn_dist(Y, k), k)

        auc_hd = roc_auc_score(y, s_hd)
        auc_ld = roc_auc_score(y, s_ld)
        rho = spearmanr(s_hd, s_ld).statistic
        rows.append({"dataset": name, "n": meta["n"], "D": meta["D"],
                     "auc_hd": auc_hd, "auc_ld": auc_ld,
                     "retained": auc_ld / auc_hd, "spearman": rho})

    hdr = (f"{'dataset':<12}{'D':>5}{'AUC orig':>10}{'AUC 2D':>9}"
           f"{'retained':>10}{'rank corr':>11}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<12}{r['D']:>5}{r['auc_hd']:>10.3f}{r['auc_ld']:>9.3f}"
              f"{100 * r['retained']:>9.0f}%{r['spearman']:>11.3f}")

    out = RESULTS / f"step6_{method}_2d.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")
