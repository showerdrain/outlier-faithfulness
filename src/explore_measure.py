"""Week 4: the v1 measure tested on the development datasets.

    python src/explore_measure.py             # everything; about an hour on 2 cores
    python src/explore_measure.py --quick     # identity, PCA, random projection only
    python src/explore_measure.py --figures   # redraw the figures from saved results

Only the development datasets listed in results/splits.json are used, and
anything else is refused, so no benchmark dataset can shape the measure.
Datasets are capped at 20,000 points (stratified, seed 0), the cap proposed
for the benchmark; of the five, only shuttle is larger.

What is tested, each against an expectation stated before running it
(docs/MEASURE.md, section 5):
  1. methods at 2D   identity, PCA, random projection (5 seeds), UMAP, t-SNE
                     and PaCMAP (3 seeds each), top 5% budget
  2. PCA sweep       target dimension 1 .. D
  3. budget sweep    top 1% .. 20%, and the anomaly count (the old P@n budget)
  4. null models     the v1 noise null against three alternatives
  5. ground truth    what the failures cost, and LID by category (exploratory)

Writes
  results/week4_measure.json             one row per dataset/method/seed/dim/budget
  results/week4_nulls.json               the null-model comparison
  results/week4_points_<dataset>.npz     per-point values, PCA and UMAP at 2D
  figures/week4_{funnel,maps,methods,dimension,budget}.png
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from config import DEFAULT_K, FIGURES, RESULTS
from datasets import load_dataset
from neighbors import get_knn, knn_graph, knn_score, lid_mle
from projections import knn_dist, project
import measure as M

K = DEFAULT_K
CAP = 20_000
LOAD = {"subsample": CAP}          # every load goes through the same options
OUT = RESULTS / "week4_measure.json"
OUT_NULLS = RESULTS / "week4_nulls.json"

METHOD_SEEDS = {"identity": [0], "pca": [0], "random": [0, 1, 2, 3, 4],
                "umap": [0, 1, 2], "tsne": [0, 1, 2], "pacmap": [0, 1, 2]}
QUICK_METHODS = ("identity", "pca", "random")
BUDGETS = {"1%": 0.01, "2%": 0.02, "5%": 0.05, "10%": 0.10, "20%": 0.20}


def dev_names() -> list[str]:
    split = json.loads((RESULTS / "splits.json").read_text())
    return [d["name"] for d in split["development"]]


def refuse_benchmark(names: list[str]) -> None:
    dev = set(dev_names())
    bad = [n for n in names if n not in dev]
    if bad:
        raise SystemExit(f"Refused: {bad} are not development datasets "
                         f"(results/splits.json). The measure is designed on "
                         f"development data only.")


def original_space(name: str) -> dict:
    X, y, meta = load_dataset(name, **LOAD)
    dist, _, _ = get_knn(name, k=40, verbose=False, **LOAD)
    L, _ = M.get_null(name, K, **LOAD)
    return {"X": np.asarray(X), "y": np.asarray(y).astype(bool), "meta": meta,
            "dist": dist, "a": knn_score(dist, K), "lid": lid_mle(dist[:, :K], K),
            "L": L}


def embed(name: str, X: np.ndarray, method: str, d: int, seed: int) -> np.ndarray:
    if method == "identity":
        return X
    Y, _ = project(name, method=method, n_components=d, seed=seed, verbose=False, **LOAD)
    return Y


def row(name, method, seed, d, budget_label, s, extra=None) -> dict:
    r = {"dataset": name, "method": method, "seed": seed, "dim": d,
         "budget": budget_label, **{k: v for k, v in s.items() if k != "points"}}
    if extra:
        r.update(extra)
    return r


def lid_by_category(lid: np.ndarray, cat: np.ndarray) -> dict:
    out = {}
    for code, label in ((M.MASKED, "masked"), (M.RETAINED, "retained"),
                        (M.EXIT_AMBIGUOUS, "exit_ambiguous")):
        sel = cat == code
        out[f"lid_median_{label}"] = float(np.nanmedian(lid[sel])) if sel.any() else None
    return out


def pca_dims(D: int) -> list[int]:
    dims = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 48, 64]
    return sorted({d for d in dims if d < D} | {D})


# --- the null-model comparison ---------------------------------------------

def null_comparison(name: str, hd: dict, Y: np.ndarray, b: np.ndarray) -> list[dict]:
    """How many boundary crossings each candidate null calls significant."""
    a, n = hd["a"], len(hd["a"])
    m = M.budget(n)
    rho_a, rho_b = M.desc_ranks(a), M.desc_ranks(b)
    lam = M.displacement(rho_a, rho_b)
    in_a, in_b = M.shortlist(rho_a, m), M.shortlist(rho_b, m)
    exits, entries = in_a & ~in_b, in_b & ~in_a
    ref = M.classify(rho_a, rho_b, m, M.null_pvalues(hd["L"], lam)) == M.MASKED

    def tally(label, masked, phantom, note):
        masked, phantom = exits & masked, entries & phantom
        union = (masked | ref).sum()
        return {"dataset": name, "null": label, "exits": int(exits.sum()),
                "entries": int(entries.sum()), "masked": int(masked.sum()),
                "phantom": int(phantom.sum()),
                "agreement_with_v1_masked": float((masked & ref).sum() / union) if union else 1.0,
                "note": note}

    out = []
    p = M.null_pvalues(hd["L"], lam)
    out.append(tally("half-reference, B=100 (v1)", p < M.ALPHA, p < M.ALPHA,
                     "every point scored against random halves, k/2 neighbours"))

    L50, _ = M.get_null(name, K, B=50, null_seed=1, verbose=False, **LOAD)
    p50 = M.null_pvalues(L50, lam)
    out.append(tally("half-reference, B=50, other seed", p50 < M.ALPHA, p50 < M.ALPHA,
                     "stability of the v1 null to its own Monte Carlo"))

    ks = [10, 15, 20, 25, 30, 40]
    Lk = np.column_stack([np.log2(M.desc_ranks(knn_score(hd["dist"], kk))) for kk in ks])
    pk = M.null_pvalues(Lk, lam)
    out.append(tally("k-variation, k = 10..40", pk < M.ALPHA, pk < M.ALPHA,
                     "displacement beyond what changing the detector's k does"))

    # scDEED-style (Xia et al. 2024): permute every feature independently, project
    # the permuted data with the same method, pool displacements, cut at 5% / 95%.
    rng = np.random.default_rng(0)
    X = hd["X"]
    Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    from sklearn.decomposition import PCA
    ap = knn_score(knn_graph(Xp, K)[0], K)
    bp = knn_score(knn_dist(PCA(n_components=Y.shape[1], svd_solver="full").fit_transform(Xp), K), K)
    rap, rbp = M.desc_ranks(ap), M.desc_ranks(bp)
    lamp = M.displacement(rap, rbp)
    lo = float(np.quantile(lamp[M.shortlist(rap, m)], M.ALPHA))
    hi = float(np.quantile(lamp[M.shortlist(rbp, m)], 1 - M.ALPHA))
    r = tally("feature permutation (scDEED-style)", lam < lo, lam > hi,
              f"pooled cut-offs {lo:+.2f} / {hi:+.2f} bits")
    r.update(cutoff_masked_bits=lo, cutoff_phantom_bits=hi)
    out.append(r)

    out.append(tally("none: any boundary crossing", np.ones(n, bool), np.ones(n, bool),
                     "the placeholder's rule"))
    return out


# --- the run ----------------------------------------------------------------

def run(quick: bool) -> None:
    names = dev_names()
    refuse_benchmark(names)
    methods = QUICK_METHODS if quick else tuple(METHOD_SEEDS)
    rows, null_rows = [], []
    t_start = time.time()

    for name in names:
        t0 = time.time()
        hd = original_space(name)
        n, D = len(hd["a"]), hd["X"].shape[1]
        y, a, L = hd["y"], hd["a"], hd["L"]
        m5 = M.budget(n)
        print(f"\n{name}: n={n}, D={D}, top 5% = {m5} points, "
              f"{int(y.sum())} labelled anomalies", flush=True)
        points = {"rho_a": M.desc_ranks(a), "lid": hd["lid"], "y": y,
                  "q": M.null_threshold(L)}

        # 1. methods at 2D, top 5%
        for method in methods:
            for seed in METHOD_SEEDS[method]:
                t1 = time.time()
                Y = embed(name, hd["X"], method, 2, seed)
                b = knn_score(knn_dist(Y, K), K)
                s = M.summarise(a, b, L, y=y)
                rows.append(row(name, method, seed, D if method == "identity" else 2,
                                "5%", s, lid_by_category(hd["lid"], s["points"]["cat"])))
                print(f"  {method:<8} seed {seed}  MR {s['masking_rate']:.2f}  "
                      f"PR {s['phantom_rate']:.2f}  AO {s['adjusted_overlap']:+.2f}  "
                      f"({time.time() - t1:.0f}s)", flush=True)
                if method in ("pca", "umap") and seed == 0:
                    for key in ("rho_b", "lam", "p", "cat"):
                        points[f"{method}_{key}"] = s["points"][key]
                    points[f"{method}_Y"] = Y
                if method == "pca":
                    null_rows += null_comparison(name, hd, Y, b)

        # 2. PCA sweep over target dimension, top 5%
        for d in pca_dims(D):
            Y = embed(name, hd["X"], "pca", d, 0)
            s = M.summarise(a, knn_score(knn_dist(Y, K), K), L, y=y)
            rows.append(row(name, "pca", 0, d, "5%", s, {"sweep": "dimension"}))

        # 3. budget sweep at 2D, plus the anomaly count as budget
        for method, seed in (("pca", 0), ("random", 0)) + (() if quick else (("umap", 0),)):
            b = knn_score(knn_dist(embed(name, hd["X"], method, 2, seed), K), K)
            for label, frac in list(BUDGETS.items()) + [("anomalies", None)]:
                m = int(y.sum()) if frac is None else M.budget(n, frac)
                s = M.summarise(a, b, L, m=m, y=y)
                rows.append(row(name, method, seed, 2, label, s, {"sweep": "budget"}))

        np.savez_compressed(RESULTS / f"week4_points_{name}.npz", **points)
        print(f"  {name} done in {time.time() - t0:.0f}s", flush=True)

    for r in rows:                     # rounded for stable diffs; NaN (a median
        for k, v in list(r.items()):   # over no points) becomes null, valid JSON
            if isinstance(v, float):
                r[k] = None if math.isnan(v) else round(v, 6)
    OUT.write_text(json.dumps({"k": K, "cap": CAP, "alpha": M.ALPHA, "null_B": M.NULL_B,
                               "quick": quick, "rows": rows}, indent=1))
    OUT_NULLS.write_text(json.dumps(null_rows, indent=1))
    print(f"\nWrote {OUT}\nWrote {OUT_NULLS}\n({time.time() - t_start:.0f}s in all)")


# --- reporting --------------------------------------------------------------

def report() -> None:
    res = json.loads(OUT.read_text())
    rows = res["rows"]
    main = [r for r in rows if r["budget"] == "5%" and "sweep" not in r]
    print(f"\nTop 5% budget, 2D (identity is the original space itself). "
          f"Means over seeds; MR/PR = masking/phantom rate.")
    print(f"{'dataset':<11}{'method':<9}{'seeds':>6}{'MR':>7}{'PR':>7}{'AO':>8}"
          f"{'wcorr':>8}{'spear':>8}{'bits':>7}{'masked anom':>13}{'phantom anom':>14}")
    for name in dict.fromkeys(r["dataset"] for r in main):
        for method in METHOD_SEEDS:
            rs = [r for r in main if r["dataset"] == name and r["method"] == method]
            if not rs:
                continue
            f = lambda k: np.nanmean([np.nan if r[k] is None else r[k] for r in rs])
            print(f"{name:<11}{method:<9}{len(rs):>6}{f('masking_rate'):>7.2f}"
                  f"{f('phantom_rate'):>7.2f}{f('adjusted_overlap'):>+8.2f}"
                  f"{f('weighted_rank_corr'):>8.2f}{f('spearman'):>8.2f}"
                  f"{f('median_bits_masked'):>7.1f}"
                  f"{f('masked_anomalies'):>8.1f} / {f('anomalies_in_a'):<4.0f}"
                  f"{f('phantom_anomalies'):>9.1f}")
    nulls = json.loads(OUT_NULLS.read_text())
    print("\nNull models, PCA to 2D, top 5%: boundary crossings called significant")
    for r in nulls:
        print(f"  {r['dataset']:<11}{r['null']:<38} masked {r['masked']:>4}/{r['exits']:<4}"
              f" phantom {r['phantom']:>4}/{r['entries']:<4} agreement {r['agreement_with_v1_masked']:.2f}")


if __name__ == "__main__":
    if "--figures" not in sys.argv:
        run(quick="--quick" in sys.argv)
    report()
    from figures_week4 import draw_all
    draw_all()
