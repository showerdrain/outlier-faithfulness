"""Week 5: can scores replace ranks? (docs/MEASURE.md, 9.7)

    python src/score_check.py

Version 1 left "ranks or normalised scores" open (MEASURE.md, section 6):
ranks treat a point far ahead of the next one and a point barely ahead alike.
A score-based displacement would compare the size of each point's score in
the two spaces, after putting both on a common scale. The standard way to do
that is Gaussian scaling (Kriegel, Kroger, Schubert and Zimek, SDM 2011):

    P(i) = max(0, erf((s(i) - mean(s)) / (sd(s) sqrt 2)))

with mean and standard deviation over all points of one space. The version-2
score is already a common-scale number, how many times more isolated than a
typical point, so its plain ratio is a second candidate.

The test. Take the points both spaces put in their top 5% (retained points).
For those, the projection agrees with the detector about who is outlying, so a
displacement that is about the point and not about the space should be near
zero on average. If instead it is far from zero for the agreed points, the
scale does not carry over between spaces, and a score-based call would say
"masked" or "phantom" because the space changed, not because the point moved.

A reason to expect that: in d dimensions the k-nearest-neighbour distance
grows like density ^ (-1 / d). The same density contrast is a distance ratio
of 10 ^ (1/9) = 1.29 in nine dimensions and 10 ^ (1/2) = 3.16 in two. Ranks do
not see this; ratios and Gaussian-scaled scores do. PCA to every dimension of
one dataset shows the drift directly, with a rotation (d = D) as the zero.

Development datasets only. Writes results/week5_score_check.json (tracked).
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
from scipy.special import erf

warnings.filterwarnings("ignore")

import detectors as T
import measure as M
from config import RESULTS
from datasets import load_dataset
from explore_measure import LOAD, dev_names, pca_dims, refuse_benchmark
from projections import project

METHODS = ("pca", "random", "umap", "tsne", "pacmap", "trimap")
OUT = RESULTS / "week5_score_check.json"


def gaussian_scaling(s: np.ndarray) -> np.ndarray:
    """Kriegel et al. (2011): probability-like scores in [0, 1]."""
    s = np.asarray(s, dtype=np.float64)
    return np.maximum(0.0, erf((s - s.mean()) / (s.std() * math.sqrt(2))))


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Score-based displacements of the points both spaces put in their top 5%."""
    n = len(a)
    m = M.budget(n)
    rho_a, rho_b = M.desc_ranks(a), M.desc_ranks(b)
    kept = M.shortlist(rho_a, m) & M.shortlist(rho_b, m)
    in_a = M.shortlist(rho_a, m)
    pa, pb = gaussian_scaling(a), gaussian_scaling(b)
    lam = M.displacement(rho_a, rho_b)
    out = {"n": n, "m": m, "retained": int(kept.sum())}
    if kept.any():
        out.update({
            "rank_bits_median": float(np.median(lam[kept])),
            "gauss_diff_median": float(np.median(pb[kept] - pa[kept])),
            "ratio_bits_median": float(np.median(np.log2(b[kept] / a[kept]))),
        })
    out.update({
        "gauss_top_a": float(np.median(pa[in_a])),
        "ratio_top_a": float(np.median(a[in_a])),
        "gauss_top_b": float(np.median(pb[M.shortlist(rho_b, m)])),
        "ratio_top_b": float(np.median(b[M.shortlist(rho_b, m)])),
    })
    return out


def main() -> None:
    names = dev_names()
    refuse_benchmark(names)
    rows = []
    for name in names:
        X, _, meta = load_dataset(name, **LOAD)
        a = T.score_original(name, "knn", T.SCALES_V2, **LOAD)
        for method in METHODS:
            Y, _ = project(name, method=method, n_components=2, seed=0, verbose=False, **LOAD)
            rows.append({"dataset": name, "method": method, "dim": 2,
                         **compare(a, T.score_space(Y, "knn", T.SCALES_V2))})
        for d in pca_dims(meta["D"]):
            if d == 2:
                continue
            Y, _ = project(name, method="pca", n_components=d, seed=0, verbose=False, **LOAD)
            rows.append({"dataset": name, "method": "pca", "dim": d,
                         **compare(a, T.score_space(Y, "knn", T.SCALES_V2))})
        for r in rows:
            if r["dataset"] == name and r["dim"] == 2:
                print(f"{name:<11}{r['method']:<8} retained {r['retained']:>4}  "
                      f"median over retained: rank {r.get('rank_bits_median', float('nan')):+.2f} "
                      f"bits, Gaussian {r.get('gauss_diff_median', float('nan')):+.2f}, "
                      f"ratio {r.get('ratio_bits_median', float('nan')):+.2f} bits", flush=True)
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, float):
                r[k] = None if math.isnan(v) else float(f"{v:.6g}")
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
