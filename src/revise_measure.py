"""Week 5: sanity checks and the revision of the measure (version 2).

    python src/revise_measure.py            # everything; about 40 minutes on 2 cores
    python src/revise_measure.py --figures  # redraw the figures from saved results
    python src/revise_measure.py --report   # the tables of docs/MEASURE.md, section 9
    python src/revise_measure.py --redundancy-v1   # check 5 with version 1's calls

Development datasets only, read from results/splits.json; anything else is
refused. Same cap as week 4 (20,000 points, stratified, seed 0).

What is run, each against an expectation stated in docs/MEASURE.md (section 9):
  1. v1 against v2     every method at 2D, seeds 0-4 (identity and PCA once),
                       top 5%; the PCA dimension sweep and the budget sweep
                       for v2
  2. scale sets        v2 with three other sets of scales, seed 0: does the
                       verdict depend on the choice less than v1's did on k?
  3. a second detector LOF, multi-scale, on the three continuous datasets
                       (on lattice data LOF depends on how tied distances are
                       ordered, so it is not used there)
  4. seed stability    how far per-point calls agree across five seeds
  5. redundancy        per-point trustworthiness and continuity (k = 20)
                       against the masked and phantom calls, seed 0

Writes
  results/week5_revision.json           (tracked)
  results/week5_points_<dataset>.npz    per-point values for the figures
  figures/week5_{versions,maps,seeds,redundancy}.png
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from itertools import combinations

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

import detectors as T
import measure as M
import quality as Q
from config import RESULTS
from datasets import load_dataset
from explore_measure import LOAD, dev_names, pca_dims, refuse_benchmark
from neighbors import get_knn
from projections import project

K = 20
OUT = RESULTS / "week5_revision.json"
SEEDS = [0, 1, 2, 3, 4]
METHOD_SEEDS = {"identity": [0], "pca": [0], "random": SEEDS, "umap": SEEDS,
                "tsne": SEEDS, "pacmap": SEEDS, "trimap": SEEDS}
VERSIONS = {"v1": T.SCALES_V1, "v2": T.SCALES_V2}
ALT_SCALES = {"20-50": (20, 50), "10-20-50-100": (10, 20, 50, 100), "20-100": (20, 100)}
CONTINUOUS = ("ionosphere", "fault", "magic")          # results/week4_lattice.json
BUDGETS = {"1%": 0.01, "2%": 0.02, "5%": 0.05, "10%": 0.10, "20%": 0.20}


def embed(name: str, X: np.ndarray, method: str, d: int, seed: int) -> np.ndarray:
    if method == "identity":
        return X
    return project(name, method=method, n_components=d, seed=seed, verbose=False, **LOAD)[0]


def strip(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "points"}


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def run() -> None:
    names = dev_names()
    refuse_benchmark(names)
    rows, scale_rows, lof_rows, seed_rows, red_rows = [], [], [], [], []
    t_start = time.time()

    for name in names:
        t0 = time.time()
        X, y, meta = load_dataset(name, **LOAD)
        X, y = np.asarray(X), np.asarray(y).astype(bool)
        n, D = X.shape
        hd = {v: T.score_original(name, "knn", sc, **LOAD) for v, sc in VERSIONS.items()}
        nulls = {v: M.get_null(name, K, detector="knn", scales=None if v == "v1" else sc,
                               verbose=True, **LOAD)[0] for v, sc in VERSIONS.items()}
        print(f"\n{name}: n={n}, D={D}", flush=True)
        points = {"y": y}
        calls = {v: {} for v in VERSIONS}

        # 1. v1 against v2, all methods and seeds, 2D, top 5%
        for method, seeds in METHOD_SEEDS.items():
            for seed in seeds:
                Y = embed(name, X, method, 2, seed)
                for v, sc in VERSIONS.items():
                    s = M.summarise(hd[v], T.score_space(Y, "knn", sc), nulls[v], y=y)
                    rows.append({"dataset": name, "method": method, "seed": seed,
                                 "dim": D if method == "identity" else 2, "budget": "5%",
                                 "version": v, **strip(s)})
                    calls[v][(method, seed)] = s["points"]["cat"]
                    if seed == 0 and method in ("pca", "umap", "tsne", "pacmap", "trimap"):
                        points[f"{v}_{method}_cat"] = s["points"]["cat"]
                        points[f"{method}_Y"] = Y
                r1, r2 = rows[-2], rows[-1]
                print(f"  {method:<8} seed {seed}  MR v1 {r1['masking_rate']:.2f} -> "
                      f"v2 {r2['masking_rate']:.2f}   AO {r1['adjusted_overlap']:+.2f} -> "
                      f"{r2['adjusted_overlap']:+.2f}", flush=True)
        points["v1_rho_a"] = M.desc_ranks(hd["v1"])
        points["v2_rho_a"] = M.desc_ranks(hd["v2"])

        # PCA dimension sweep and budget sweep, version 2
        for d in pca_dims(D):
            Y = embed(name, X, "pca", d, 0)
            s = M.summarise(hd["v2"], T.score_space(Y, "knn", T.SCALES_V2), nulls["v2"], y=y)
            rows.append({"dataset": name, "method": "pca", "seed": 0, "dim": d, "budget": "5%",
                         "version": "v2", "sweep": "dimension", **strip(s)})
        for method in ("pca", "umap"):
            b = T.score_space(embed(name, X, method, 2, 0), "knn", T.SCALES_V2)
            for label, frac in list(BUDGETS.items()) + [("anomalies", None)]:
                m = int(y.sum()) if frac is None else M.budget(n, frac)
                s = M.summarise(hd["v2"], b, nulls["v2"], m=m, y=y)
                rows.append({"dataset": name, "method": method, "seed": 0, "dim": 2,
                             "budget": label, "version": "v2", "sweep": "budget", **strip(s)})

        # 2. other scale sets, seed 0
        for label, sc in ALT_SCALES.items():
            a = T.score_original(name, "knn", sc, **LOAD)
            L, _ = M.get_null(name, K, detector="knn", scales=sc, verbose=True, **LOAD)
            for method in ("pca", "random", "umap", "tsne", "pacmap", "trimap"):
                s = M.summarise(a, T.score_space(embed(name, X, method, 2, 0), "knn", sc), L)
                scale_rows.append({"dataset": name, "method": method, "scales": label,
                                   "masking_rate": s["masking_rate"],
                                   "phantom_rate": s["phantom_rate"],
                                   "adjusted_overlap": s["adjusted_overlap"]})

        # 3. LOF, multi-scale, continuous datasets only
        if name in CONTINUOUS:
            a = T.score_original(name, "lof", T.SCALES_V2, **LOAD)
            L, _ = M.get_null(name, K, detector="lof", scales=T.SCALES_V2, verbose=True, **LOAD)
            lof_auc = safe_auc(y, a)
            for method in ("pca", "random", "umap", "tsne", "pacmap", "trimap"):
                for seed in METHOD_SEEDS[method][:3]:
                    Y = embed(name, X, method, 2, seed)
                    s = M.summarise(a, T.score_space(Y, "lof", T.SCALES_V2), L, y=y)
                    lof_rows.append({"dataset": name, "method": method, "seed": seed,
                                     "auc_original": lof_auc, **strip(s)})

        # 4. seed stability of the per-point calls
        for v in VERSIONS:
            rho_a = M.desc_ranks(hd[v])
            in_a = M.shortlist(rho_a, M.budget(n))
            for method in ("random", "umap", "tsne", "pacmap", "trimap"):
                masked = [calls[v][(method, sd)] == M.MASKED for sd in SEEDS]
                phantom = [calls[v][(method, sd)] == M.PHANTOM for sd in SEEDS]
                times = np.sum(masked, axis=0)[in_a]
                seed_rows.append({
                    "dataset": name, "method": method, "version": v,
                    "jaccard_masked": float(np.mean([jaccard(p, q) for p, q in combinations(masked, 2)])),
                    "jaccard_phantom": float(np.mean([jaccard(p, q) for p, q in combinations(phantom, 2)])),
                    "share_masked_all5": float(np.mean(times == 5)),
                    "share_masked_none": float(np.mean(times == 0)),
                    "share_split": float(np.mean((times > 0) & (times < 5))),
                    "masking_rate_min": float(min(m[in_a].mean() for m in masked)),
                    "masking_rate_max": float(max(m[in_a].mean() for m in masked)),
                })

        # 5. redundancy: per-point trustworthiness and continuity, seed 0
        rng = np.random.default_rng(0)
        for method in ("pca", "random", "umap", "tsne", "pacmap", "trimap"):
            Y = embed(name, X, method, 2, 0)
            s = M.summarise(hd["v2"], T.score_space(Y, "knn", T.SCALES_V2), nulls["v2"])
            cat = s["points"]["cat"]
            in_a = M.shortlist(s["points"]["rho_a"], s["m"])
            in_b = M.shortlist(s["points"]["rho_b"], s["m"])
            sample = rng.choice(n, size=min(n, 2000), replace=False)
            rows_tc = np.unique(np.r_[np.flatnonzero(in_a | in_b), sample])
            t, c = Q.local_trust_cont(X, Y, K, rows=rows_tc)
            tt, cc = np.full(n, np.nan), np.full(n, np.nan)
            tt[rows_tc], cc[rows_tc] = t, c
            exits = in_a & (cat != M.RETAINED)
            masked_vs_kept = in_a & ((cat == M.MASKED) | (cat == M.RETAINED))
            entries = in_b & ((cat == M.PHANTOM) | (cat == M.RETAINED))
            red_rows.append({
                "dataset": name, "method": method,
                "masking_rate": s["masking_rate"], "phantom_rate": s["phantom_rate"],
                "trustworthiness": float(np.nanmean(tt[sample])),
                "continuity": float(np.nanmean(cc[sample])),
                "auc_continuity_masked": safe_auc((cat == M.MASKED)[masked_vs_kept],
                                                  -cc[masked_vs_kept]),
                "auc_trust_phantom": safe_auc((cat == M.PHANTOM)[entries], -tt[entries]),
                "continuity_masked_median": float(np.nanmedian(cc[cat == M.MASKED])) if (cat == M.MASKED).any() else None,
                "continuity_retained_median": float(np.nanmedian(cc[cat == M.RETAINED])) if (cat == M.RETAINED).any() else None,
                "continuity_all_median": float(np.nanmedian(cc[sample])),
                "n_exits": int(exits.sum()),
            })
            if method in ("pca", "umap"):
                points[f"{method}_continuity"] = cc
                points[f"{method}_trust"] = tt
            print(f"  redundancy {method:<8} T {red_rows[-1]['trustworthiness']:.3f} "
                  f"C {red_rows[-1]['continuity']:.3f}  AUC(1-c -> masked) "
                  f"{red_rows[-1]['auc_continuity_masked']}", flush=True)

        np.savez_compressed(RESULTS / f"week5_points_{name}.npz", **points)
        print(f"  {name} done in {time.time() - t0:.0f}s", flush=True)

    def clean(rs):
        for r in rs:
            for k, v in list(r.items()):
                if isinstance(v, float):
                    r[k] = None if math.isnan(v) else round(v, 6)
        return rs

    OUT.write_text(json.dumps({
        "k": K, "cap": LOAD["subsample"], "alpha": M.ALPHA, "null_B": M.NULL_B,
        "scales_v1": list(T.SCALES_V1), "scales_v2": list(T.SCALES_V2),
        "rows": clean(rows), "scale_sets": clean(scale_rows), "lof": clean(lof_rows),
        "seed_stability": clean(seed_rows), "redundancy": clean(red_rows)}, indent=1))
    print(f"\nWrote {OUT} ({time.time() - t_start:.0f}s in all)")


OUT_RED_V1 = RESULTS / "week5_redundancy_v1.json"


def redundancy_v1() -> None:
    """Check 5 again with version 1's calls, the version week 5 kept.

    run() made the first look at redundancy with the candidate's calls, before
    the decision. This repeats it for version 1, with the same rows, sample
    and seed, and writes results/week5_redundancy_v1.json.
    """
    names = dev_names()
    refuse_benchmark(names)
    out = []
    for name in names:
        X, _, _ = load_dataset(name, **LOAD)
        X = np.asarray(X)
        n = len(X)
        a = T.score_original(name, "knn", T.SCALES_V1, **LOAD)
        L, _ = M.get_null(name, K, verbose=False, **LOAD)
        rng = np.random.default_rng(0)
        for method in ("pca", "random", "umap", "tsne", "pacmap", "trimap"):
            Y = embed(name, X, method, 2, 0)
            s = M.summarise(a, T.score_space(Y, "knn", T.SCALES_V1), L)
            cat = s["points"]["cat"]
            in_a = M.shortlist(s["points"]["rho_a"], s["m"])
            in_b = M.shortlist(s["points"]["rho_b"], s["m"])
            sample = rng.choice(n, size=min(n, 2000), replace=False)
            rows_tc = np.unique(np.r_[np.flatnonzero(in_a | in_b), sample])
            t, c = Q.local_trust_cont(X, Y, K, rows=rows_tc)
            tt, cc = np.full(n, np.nan), np.full(n, np.nan)
            tt[rows_tc], cc[rows_tc] = t, c
            masked_vs_kept = in_a & ((cat == M.MASKED) | (cat == M.RETAINED))
            entries = in_b & ((cat == M.PHANTOM) | (cat == M.RETAINED))
            out.append({
                "dataset": name, "method": method,
                "masking_rate": s["masking_rate"], "phantom_rate": s["phantom_rate"],
                "trustworthiness": float(np.nanmean(tt[sample])),
                "continuity": float(np.nanmean(cc[sample])),
                "auc_continuity_masked": safe_auc((cat == M.MASKED)[masked_vs_kept],
                                                  -cc[masked_vs_kept]),
                "auc_trust_phantom": safe_auc((cat == M.PHANTOM)[entries], -tt[entries]),
            })
            print(f"  {name:<11}{method:<8}MR {s['masking_rate']:.2f}  AUC(1-c: masked) "
                  f"{out[-1]['auc_continuity_masked']}  AUC(1-t: phantom) "
                  f"{out[-1]['auc_trust_phantom']}", flush=True)
    for r in out:
        for k, v in list(r.items()):
            if isinstance(v, float):
                r[k] = None if math.isnan(v) else round(v, 6)
    OUT_RED_V1.write_text(json.dumps(out, indent=1))
    print(f"Wrote {OUT_RED_V1}")


def report() -> None:
    """The tables of docs/MEASURE.md, section 9, and each expectation's verdict."""
    res = json.loads(OUT.read_text())
    rows = res["rows"]
    names = list(dict.fromkeys(r["dataset"] for r in rows))
    main = [r for r in rows if r["budget"] == "5%" and "sweep" not in r]
    linear, neighbour = ("pca", "random"), ("umap", "tsne", "pacmap", "trimap")

    def pick(rs, **kw):
        return [r for r in rs if all(r.get(k) == v for k, v in kw.items())]

    def mean(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else float("nan")

    def span(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return (min(v), max(v)) if v else (float("nan"), float("nan"))

    print("E1  identity:", ", ".join(
        f"{r['dataset']} {r['version']} MR {r['masking_rate']:.2f} PR {r['phantom_rate']:.2f}"
        for r in pick(main, method="identity")))

    print("\nv1 -> v2 at 2D, top 5%, mean over seeds [range]")
    print(f"{'dataset':<11}{'method':<8}{'masking rate':>28}{'phantom rate':>16}{'AO':>16}")
    mr = {}
    for name in names:
        for method in ("pca", "random") + neighbour:
            r1, r2 = pick(main, dataset=name, method=method, version="v1"), \
                pick(main, dataset=name, method=method, version="v2")
            mr[name, method] = (mean(r1, "masking_rate"), mean(r2, "masking_rate"))
            lo, hi = span(r2, "masking_rate")
            print(f"{name:<11}{method:<8}{mr[name, method][0]:>9.2f} -> {mr[name, method][1]:.2f}"
                  f" [{lo:.2f}-{hi:.2f}]{mean(r1, 'phantom_rate'):>9.2f} -> "
                  f"{mean(r2, 'phantom_rate'):.2f}{mean(r1, 'adjusted_overlap'):>+9.2f} -> "
                  f"{mean(r2, 'adjusted_overlap'):+.2f}")

    d_lin = {(n, m): abs(mr[n, m][1] - mr[n, m][0]) for n in names for m in linear}
    print(f"\nE2  linear methods, |v2 - v1| masking rate: max {max(d_lin.values()):.2f} "
          f"({max(d_lin, key=d_lin.get)}); within 0.10 in {sum(v <= 0.10 for v in d_lin.values())}"
          f" of {len(d_lin)}")
    print("E3  neighbour methods, v1 -> v2 masking rate:")
    for name in names:
        print(f"    {name:<11}" + "  ".join(f"{m} {mr[name, m][0]:.2f}->{mr[name, m][1]:.2f}"
                                           for m in neighbour))
    print("E4  datasets where the neighbour methods' mean masking rate exceeds both linear "
          "methods (v1 / v2):")
    for v, j in (("v1", 0), ("v2", 1)):
        hits = [n for n in names if np.mean([mr[n, m][j] for m in neighbour])
                > max(mr[n, m][j] for m in linear)]
        each = {m: sum(mr[n, m][j] > max(mr[n, q][j] for q in linear) for n in names)
                for m in neighbour}
        print(f"    {v}: {len(hits)} of {len(names)} {hits}; per method, datasets above both "
              f"linear methods: {each}")

    sc = res["scale_sets"]
    diffs = []
    print("\nE5  other scale sets, seed 0: masking rate, v2 then each set")
    for name in names:
        for method in ("pca", "random") + neighbour:
            base = pick(main, dataset=name, method=method, version="v2", seed=0)[0]["masking_rate"]
            alt = {r["scales"]: r["masking_rate"] for r in pick(sc, dataset=name, method=method)}
            diffs += [abs(v - base) for v in alt.values()]
            print(f"    {name:<11}{method:<8}{base:.2f}  " +
                  "  ".join(f"{k} {v:.2f}" for k, v in alt.items()))
    diffs = np.array(diffs)
    print(f"    |difference| from v2: median {np.median(diffs):.3f}, max {diffs.max():.2f}, "
          f"within 0.10 in {np.mean(diffs <= 0.10):.0%} of {len(diffs)}")
    d4 = []
    for r in json.loads((RESULTS / "week4_scale.json").read_text()):
        base = r["MR_hd20_ld20"]
        d4 += [abs(r[k] - base) for k in ("MR_hd20_ld5", "MR_hd20_ld10", "MR_hd20_ld50",
                                          "MR_hd20_ld100", "MR_both50")]
    d4 = np.array(d4)
    print(f"    v1 for comparison (week 4: projection read at k = 5, 10, 50, 100, or both "
          f"sides at 50): median {np.median(d4):.3f}, max {d4.max():.2f}, within 0.10 in "
          f"{np.mean(d4 <= 0.10):.0%} of {len(d4)}")

    lof = res["lof"]
    if lof:
        print("\nE6  LOF (v2 scales) against kNN (v2), seeds 0-2, masking rate")
        from scipy.stats import spearmanr
        for name in dict.fromkeys(r["dataset"] for r in lof):
            a = [mean(pick(lof, dataset=name, method=m), "masking_rate")
                 for m in ("pca", "random") + neighbour]
            b = [mean([r for r in pick(main, dataset=name, method=m, version="v2")
                       if r["seed"] in (0, 1, 2)], "masking_rate")
                 for m in ("pca", "random") + neighbour]
            print(f"    {name:<11}LOF " + " ".join(f"{x:.2f}" for x in a) + "   kNN " +
                  " ".join(f"{x:.2f}" for x in b) +
                  f"   Spearman over methods {spearmanr(a, b).statistic:+.2f}")
        print("    (methods in the order pca random umap tsne pacmap trimap)")

    seeds = res["seed_stability"]
    print("\nE7  seed stability, 5 seeds: masking-rate range and mean Jaccard of masked sets")
    dj = []
    for name in names:
        for method in ("random",) + neighbour:
            s1 = pick(seeds, dataset=name, method=method, version="v1")[0]
            s2 = pick(seeds, dataset=name, method=method, version="v2")[0]
            dj.append(s2["jaccard_masked"] - s1["jaccard_masked"])
            print(f"    {name:<11}{method:<8}range v1 {s1['masking_rate_max'] - s1['masking_rate_min']:.2f}"
                  f" v2 {s2['masking_rate_max'] - s2['masking_rate_min']:.2f}   Jaccard v1 "
                  f"{s1['jaccard_masked']:.2f} v2 {s2['jaccard_masked']:.2f}   v2 all five "
                  f"{s2['share_masked_all5']:.2f}, some {s2['share_split']:.2f}, none "
                  f"{s2['share_masked_none']:.2f}")
    print(f"    Jaccard v2 - v1: mean {np.mean(dj):+.3f}, min {np.min(dj):+.2f}, max {np.max(dj):+.2f}")

    red = res["redundancy"]
    auc_c = [r["auc_continuity_masked"] for r in red if r["auc_continuity_masked"] is not None]
    auc_t = [r["auc_trust_phantom"] for r in red if r["auc_trust_phantom"] is not None]
    print("\nE8  redundancy, seed 0, v2 calls")
    for r in red:
        def f(v):
            return "  -  " if v is None else f"{v:.2f}"
        print(f"    {r['dataset']:<11}{r['method']:<8}T {r['trustworthiness']:.3f} "
              f"C {r['continuity']:.3f}  MR {r['masking_rate']:.2f}  AUC(1-c: masked) "
              f"{f(r['auc_continuity_masked'])}  AUC(1-t: phantom) {f(r['auc_trust_phantom'])}")
    for label, v in (("continuity -> masked", auc_c), ("trustworthiness -> phantom", auc_t)):
        v = np.array(v)
        print(f"    {label}: median AUC {np.median(v):.2f}, above 0.8 in {np.sum(v > 0.8)} "
              f"and above 0.9 in {np.sum(v > 0.9)} of {len(v)}")
    if OUT_RED_V1.exists():
        red1 = json.loads(OUT_RED_V1.read_text())
        print("    the same with version 1's calls (results/week5_redundancy_v1.json):")
        for group, methods in (("linear", linear), ("neighbour-based", neighbour)):
            for key, label in (("auc_continuity_masked", "continuity -> masked"),
                               ("auc_trust_phantom", "trustworthiness -> phantom")):
                for src, rs in (("v1", red1), ("v2", red)):
                    v = np.array([r[key] for r in rs if r["method"] in methods
                                  and r[key] is not None])
                    print(f"      {group:<16}{label:<28}{src}: median {np.median(v):.2f}, "
                          f"range {v.min():.2f}-{v.max():.2f}, below 0.5 in {np.sum(v < 0.5)}, "
                          f"above 0.8 in {np.sum(v > 0.8)} of {len(v)}")

    print("\nPCA dimension sweep, v2: masking / phantom rate")
    for name in names:
        rs = sorted(pick(rows, dataset=name, sweep="dimension"), key=lambda r: r["dim"])
        print(f"    {name:<11}" + "  ".join(f"d={r['dim']} {r['masking_rate']:.2f}/"
                                           f"{r['phantom_rate']:.2f}" for r in rs))
    print("\nBudget sweep, v2, seed 0: masking / phantom rate")
    for name in names:
        for method in ("pca", "umap"):
            rs = pick(rows, dataset=name, method=method, sweep="budget")
            print(f"    {name:<11}{method:<6}" + "  ".join(
                f"{r['budget']} {r['masking_rate']:.2f}/{r['phantom_rate']:.2f}" for r in rs))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
        sys.exit()
    if "--redundancy-v1" in sys.argv:
        redundancy_v1()
        sys.exit()
    if "--figures" not in sys.argv:
        run()
    from figures_week5 import draw_all
    draw_all()
