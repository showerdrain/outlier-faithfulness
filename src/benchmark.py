"""Week 6: the benchmark in two dimensions.

    python src/benchmark.py embed     # the projections only; computed once, cached
    python src/benchmark.py run       # noise nulls and the measure for every run
    python src/benchmark.py report    # tables, Friedman, Nemenyi and Wilcoxon-Holm
    python src/benchmark.py figures   # figures/week6_*.png from the saved results

Benchmark datasets only, read from results/splits.json; the development
datasets are refused, because the measure was designed on them. The stages
are separate on purpose. The projections do not depend on the measure's
definition, so they were computed while week 5 was still revising it, and no
outlier score was computed on benchmark data until week 5 had decided.

Week 5 kept version 1: the multi-scale candidate (version 2) missed two of
the four adoption criteria fixed beforehand, both narrowly (docs/MEASURE.md,
9.11). Version 1 is therefore the primary analysis here. The multi-scale
reading runs beside it as a secondary analysis, declared before the run, so
the decision can be reversed without re-running anything.

The protocol (docs/QUESTIONS.md has the reason for each choice):
  datasets   the 15 benchmark datasets, each capped at 20,000 points
             (stratified, seed 0, after duplicate removal), standardised
  methods    PCA; a Gaussian random projection (the uninformed floor); UMAP,
             t-SNE, PaCMAP and TriMap at library defaults
  seeds      0, 1, 2 for every stochastic method; week 8 adds seeds 3-4 and
             the hyperparameter grid
  dimension  2; week 7 repeats in more dimensions
  measure    version 1 (docs/MEASURE.md): kNN at k = 20 in both spaces, top 5%
             budget, noise null of 100 half-reference replicates, alpha 0.05
  secondary  the anomaly-count budget (the P@n convention); the comparison
             without heavily tied datasets; the multi-scale reading
  anchor     detection AUC in both spaces, the task-level summary the
             proposal asks for alongside the per-point measure
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from config import RESULTS
from projections import project

CAP = 20_000
LOAD = {"subsample": CAP}
DIM = 2
K = 20
METHOD_SEEDS = {"pca": [0], "random": [0, 1, 2], "umap": [0, 1, 2], "tsne": [0, 1, 2],
                "pacmap": [0, 1, 2], "trimap": [0, 1, 2]}
OUT = RESULTS / "week6_benchmark.json"
OUT_STATS = RESULTS / "week6_stats.json"
# Nemenyi critical values q_0.05 by number of methods (Demsar, JMLR 2006, table 5a)
Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
             9: 3.102, 10: 3.164}
NOISE_FLOOR = 0.02      # docs/LOG.md, 2026-09: differences below this are not real
# Fixed before the run: the comparison is repeated without datasets whose
# original-space scores are less than half distinct (heavy ties, lattice data)
TIES_THRESHOLD = 0.5


def benchmark_names() -> list[str]:
    split = json.loads((RESULTS / "splits.json").read_text())
    return [d["name"] for d in split["benchmark"]]


def refuse_development(names: list[str]) -> None:
    split = json.loads((RESULTS / "splits.json").read_text())
    dev = {d["name"] for d in split["development"]}
    bad = [n for n in names if n in dev or n not in benchmark_names()]
    if bad:
        raise SystemExit(f"Refused: {bad} are not benchmark datasets (results/splits.json).")


def embed(names: list[str]) -> None:
    """Compute and cache every projection in the protocol. No measure."""
    t_all = time.time()
    for name in names:
        t0 = time.time()
        for method, seeds in METHOD_SEEDS.items():
            for seed in seeds:
                t1 = time.time()
                project(name, method=method, n_components=DIM, seed=seed,
                        verbose=False, **LOAD)
                print(f"  {name:<11}{method:<8}seed {seed}  {time.time() - t1:5.0f}s",
                      flush=True)
        print(f"{name} embedded in {time.time() - t0:.0f}s", flush=True)
    print(f"all embedded in {time.time() - t_all:.0f}s", flush=True)


def _clean(v):
    """JSON-ready: NaN to null, floats to 6 significant figures (p-values keep their size)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else float(f"{float(v):.6g}")
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def lattice(name: str) -> dict:
    """How grid-like the raw data is (as in lattice_check.py, docs/MEASURE.md 5.9)."""
    from config import DATA_RAW
    from datasets import REGISTRY

    raw = np.load(DATA_RAW / f"{REGISTRY[name].file}.npz", allow_pickle=True)
    X_raw = np.asarray(raw["X"], dtype=np.float64)
    integer = np.all(np.isclose(X_raw, np.round(X_raw)), axis=0)
    values = [len(np.unique(X_raw[:, j])) for j in range(X_raw.shape[1])]
    return {"integer_features": int(integer.sum()),
            "distinct_per_feature_median": float(np.median(values))}


def run(names: list[str]) -> None:
    """The measure on every run of the protocol. Writes results/week6_benchmark.json.

    Top-level fields of each row are the primary analysis, version 1 (kNN at
    k = 20). The nested "multiscale" fields are the multi-scale reading, the
    candidate version 2 of week 5, and "anomaly_budget" is version 1 at the
    P@n budget. Both readings use the same projections and the same budget.
    """
    from sklearn.metrics import roc_auc_score

    import detectors as T
    import measure as M
    from datasets import load_dataset
    from lattice_check import islands
    from neighbors import get_knn, lid_mle

    rows, info = [], {}
    t_all = time.time()
    for name in names:
        t0 = time.time()
        X, y, meta = load_dataset(name, **LOAD)
        X, y = np.asarray(X), np.asarray(y).astype(bool)
        n = len(X)
        a = T.score_original(name, "knn", T.SCALES_V1, **LOAD)          # version 1
        L, _ = M.get_null(name, K, **LOAD)
        scales = T.usable_scales(T.SCALES_V2, n)                          # multi-scale
        a2 = T.score_original(name, "knn", scales, **LOAD)
        L2, _ = M.get_null(name, K, detector="knn", scales=scales, **LOAD)
        dist, idx, _ = get_knn(name, k=K, verbose=False, **LOAD)
        auc_a, auc_a2 = float(roc_auc_score(y, a)), float(roc_auc_score(y, a2))
        m5, m_anom = M.budget(n), int(y.sum())
        info[name] = {"n": n, "n_full": meta["n_full"], "D": meta["D"], "anomalies": m_anom,
                      "budget_5pct": m5, "auc_original": auc_a,
                      "distinct_scores_share": M.n_distinct(a) / n,
                      "multiscale": {"scales": list(scales), "auc_original": auc_a2,
                                     "distinct_scores_share": M.n_distinct(a2) / n},
                      **lattice(name),
                      # clustered anomalies: median share of an anomaly's 20 nearest
                      # neighbours that are anomalies too (0 = isolated, 1 = in a group)
                      "anomaly_neighbour_share_median": float(np.median(y[idx[y]].mean(axis=1)))}
        pts = {"y": y, "rho_a": M.desc_ranks(a).astype(np.float32),
               "rho_a_ms": M.desc_ranks(a2).astype(np.float32),
               "lid": lid_mle(dist, K).astype(np.float32)}
        for method, seeds in METHOD_SEEDS.items():
            for seed in seeds:
                Y, _ = project(name, method=method, n_components=DIM, seed=seed,
                               verbose=False, **LOAD)
                b = T.score_space(Y, "knn", T.SCALES_V1)
                s = M.summarise(a, b, L, m=m5, y=y)
                s_anom = M.summarise(a, b, L, m=m_anom, y=y)
                b2 = T.score_space(Y, "knn", scales)
                s2 = M.summarise(a2, b2, L2, m=m5, y=y)
                n_isl, largest = islands(Y)
                rows.append({
                    "dataset": name, "method": method, "seed": seed,
                    "auc_original": auc_a, "auc_projection": float(roc_auc_score(y, b)),
                    **{k: v for k, v in s.items() if k != "points"},
                    "islands": n_isl, "largest_island_share": largest,
                    "anomaly_budget": {k: s_anom[k] for k in
                                       ("m", "size_a", "masking_rate", "phantom_rate",
                                        "adjusted_overlap", "masked_anomalies",
                                        "anomalies_in_a")},
                    "multiscale": {"auc_original": auc_a2,
                                   "auc_projection": float(roc_auc_score(y, b2)),
                                   **{k: s2[k] for k in
                                      ("size_a", "size_b", "masking_rate", "phantom_rate",
                                       "adjusted_overlap", "n_masked", "n_phantom",
                                       "masked_anomalies", "anomalies_in_a")}},
                })
                for tag, res in (("", s), ("_ms", s2)):
                    pts[f"{method}_s{seed}_cat{tag}"] = res["points"]["cat"]
                    pts[f"{method}_s{seed}_lam{tag}"] = res["points"]["lam"].astype(np.float32)
                    pts[f"{method}_s{seed}_p{tag}"] = res["points"]["p"].astype(np.float32)
                print(f"  {name:<11}{method:<8}seed {seed}  MR {s['masking_rate']:.2f} "
                      f"(multi-scale {s2['masking_rate']:.2f})  PR {s['phantom_rate']:.2f}  "
                      f"AO {s['adjusted_overlap']:+.2f}  AUC {auc_a:.3f} -> "
                      f"{rows[-1]['auc_projection']:.3f}", flush=True)
        np.savez_compressed(RESULTS / f"week6_points_{name}.npz", **pts)
        print(f"{name} done in {time.time() - t0:.0f}s", flush=True)

    OUT.write_text(json.dumps(_clean({
        "measure_version": 1,
        "protocol": {"cap": CAP, "dim": DIM, "k": K, "detector": "knn", "budget": "top 5%",
                     "alpha": M.ALPHA, "null_B": M.NULL_B, "methods": METHOD_SEEDS,
                     "ties_threshold": TIES_THRESHOLD,
                     "secondary": {"multiscale": "the multi-scale reading, candidate "
                                                 f"version 2: kNN at k = {list(T.SCALES_V2)}",
                                   "anomaly_budget": "version 1 at the anomaly-count budget"}},
        "datasets": info, "rows": rows}), indent=1))
    print(f"Wrote {OUT} ({time.time() - t_all:.0f}s in all)")


# --- statistics ---------------------------------------------------------------

def _get(r: dict, key: str):
    """rows[key], or a nested value for "outer.inner"."""
    for part in key.split("."):
        r = r[part]
    return r


def per_dataset_means(rows: list[dict], key: str) -> tuple[list[str], list[str], np.ndarray]:
    """(datasets, methods, N x k matrix) of the mean over seeds of rows[key]."""
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    methods = list(METHOD_SEEDS)
    Mx = np.array([[np.mean([_get(r, key) for r in rows if r["dataset"] == d and r["method"] == m])
                    for m in methods] for d in datasets])
    return datasets, methods, Mx


def compare_methods(Mx: np.ndarray, methods: list[str], lower_is_better: bool) -> dict:
    """Friedman (with the Iman-Davenport F), Nemenyi CD, pairwise Wilcoxon-Holm.

    Demsar (JMLR 2006) for the first two. Benavoli, Corani and Mangili (JMLR
    2016) argue that mean-rank post-hoc tests depend on which other methods are
    in the comparison, and recommend the Wilcoxon signed-rank test with a
    multiplicity correction instead; both are reported.
    """
    from scipy.stats import f as f_dist, friedmanchisquare, rankdata, wilcoxon

    N, k = Mx.shape
    chi2, p_chi2 = friedmanchisquare(*Mx.T)
    ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else float("inf")
    p_ff = float(f_dist.sf(ff, k - 1, (k - 1) * (N - 1)))
    ranks = np.array([rankdata(row if lower_is_better else -row) for row in Mx])
    avg = ranks.mean(axis=0)
    cd = Q_NEMENYI[k] * math.sqrt(k * (k + 1) / (6 * N))
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = Mx[:, i] - Mx[:, j]
            p = float(wilcoxon(Mx[:, i], Mx[:, j]).pvalue) if np.any(diff != 0) else 1.0
            better_a = diff < 0 if lower_is_better else diff > 0
            better_b = diff > 0 if lower_is_better else diff < 0
            pairs.append({"a": methods[i], "b": methods[j], "p": p,
                          "median_diff": float(np.median(diff)),
                          "wins_a": int(better_a.sum()), "wins_b": int(better_b.sum()),
                          "within_noise_floor": int((np.abs(diff) < NOISE_FLOOR).sum()),
                          "nemenyi_significant": bool(abs(avg[i] - avg[j]) > cd)})
    order = np.argsort([q["p"] for q in pairs])
    m = len(pairs)
    running = 0.0
    for rank, idx in enumerate(order):            # Holm step-down, monotone
        running = max(running, min(1.0, (m - rank) * pairs[idx]["p"]))
        pairs[idx]["p_holm"] = running
    return {"N": N, "k": k, "friedman_chi2": float(chi2), "friedman_p": float(p_chi2),
            "iman_davenport_F": float(ff), "iman_davenport_p": p_ff,
            "average_rank": dict(zip(methods, map(float, avg))), "nemenyi_cd": cd,
            "pairs": pairs}


ANALYSES = (  # (name, key, lower is better, datasets: "all" or "ties")
    # primary: version 1
    ("masking_rate", "masking_rate", True, "all"),
    ("phantom_rate", "phantom_rate", True, "all"),
    ("adjusted_overlap", "adjusted_overlap", False, "all"),
    ("auc_retained", "auc_retained", False, "all"),
    # secondary, declared before the run
    ("masking_rate_anomaly_budget", "anomaly_budget.masking_rate", True, "all"),
    ("masking_rate_without_heavy_ties", "masking_rate", True, "ties"),
    ("masking_rate_multiscale", "multiscale.masking_rate", True, "all"),
    ("phantom_rate_multiscale", "multiscale.phantom_rate", True, "all"),
    ("auc_retained_multiscale", "multiscale.auc_retained", False, "all"),
)


LINEAR, NEIGHBOUR = ("pca", "random"), ("umap", "tsne", "pacmap", "trimap")


def expectations(res: dict, rows: list[dict], stats: dict) -> dict:
    """The numbers behind each expectation of docs/BENCHMARK.md, section 2."""
    from scipy.stats import spearmanr

    s = stats["masking_rate"]
    mx = dict(zip(s["datasets"], s["per_dataset_mean"]))
    col = {m: i for i, m in enumerate(s["methods"])}

    def above_linear(table):
        return [d for d, v in table.items()
                if np.mean([v[col[m]] for m in NEIGHBOUR]) > max(v[col[m]] for m in LINEAR)]

    ms = dict(zip(stats["masking_rate_multiscale"]["datasets"],
                  stats["masking_rate_multiscale"]["per_dataset_mean"]))
    gap = np.array([abs(r["masking_rate"] - r["phantom_rate"]) for r in rows])

    def rank_agreement(label):
        if label not in stats:
            return None
        a, b = s["average_rank"], stats[label]["average_rank"]
        return float(spearmanr([a[m] for m in a], [b[m] for m in a]).statistic)

    info = res["datasets"]
    heavy = set(stats["heavy_ties"])

    def islands(group):
        v = [r["islands"] for r in rows if r["method"] in NEIGHBOUR and r["seed"] == 0
             and (r["dataset"] in heavy) == group]
        return float(np.median(v)) if v else None

    linear_shift = [abs(ms[d][col[m]] - mx[d][col[m]]) for d in mx for m in LINEAR]
    out = {
        "B1_datasets_neighbour_above_both_linear": above_linear(mx),
        "B1_friedman_p": s["friedman_p"],
        "B2_datasets_random_below_pca": [d for d, v in mx.items()
                                         if v[col["random"]] < v[col["pca"]]],
        "B3_share_runs_phantom_within_0.05": float(np.mean(gap <= 0.05)),
        "B3_largest_gap": float(gap.max()),
        "B4_friedman_chi2_masking": s["friedman_chi2"],
        "B4_friedman_chi2_auc_retained": stats["auc_retained"]["friedman_chi2"],
        "B5_rank_agreement_anomaly_budget": rank_agreement("masking_rate_anomaly_budget"),
        "B5_rank_agreement_without_heavy_ties": rank_agreement("masking_rate_without_heavy_ties"),
        "B6_rank_agreement_multiscale": rank_agreement("masking_rate_multiscale"),
        "B6_datasets_neighbour_above_both_linear_multiscale": above_linear(ms),
        "B6_linear_shift_median": float(np.median(linear_shift)),
        "B6_linear_shift_max": float(np.max(linear_shift)),
        "B6_mean_change_by_method": {m: float(np.mean([ms[d][col[m]] - mx[d][col[m]]
                                                        for d in mx])) for m in col},
        "B7_islands_median_neighbour_heavy_ties": islands(True),
        "B7_islands_median_neighbour_other": islands(False),
        "datasets_heavy_ties": sorted(heavy),
        "integer_only_datasets": sorted(d for d, v in info.items()
                                        if v["integer_features"] == v["D"]),
    }
    return out


def report() -> None:
    res = json.loads(OUT.read_text())
    rows = res["rows"]
    for r in rows:
        r["auc_retained"] = r["auc_projection"] / r["auc_original"]
        r["multiscale"]["auc_retained"] = (r["multiscale"]["auc_projection"]
                                           / r["multiscale"]["auc_original"])
    heavy = sorted(d for d, v in res["datasets"].items()
                   if v["distinct_scores_share"] < TIES_THRESHOLD)
    stats = {"heavy_ties": heavy, "ties_threshold": TIES_THRESHOLD}
    for label, key, lower, subset in ANALYSES:
        rs = rows if subset == "all" else [r for r in rows if r["dataset"] not in heavy]
        if subset == "ties" and not heavy:
            continue
        datasets, methods, Mx = per_dataset_means(rs, key)
        stats[label] = {"key": key, "lower_is_better": lower, "datasets": datasets,
                        "methods": methods, "per_dataset_mean": Mx.tolist(),
                        "mean": dict(zip(methods, map(float, Mx.mean(axis=0)))),
                        "median": dict(zip(methods, map(float, np.median(Mx, axis=0)))),
                        **compare_methods(Mx, methods, lower)}
    stats["expectations"] = expectations(res, rows, stats)
    OUT_STATS.write_text(json.dumps(_clean(stats), indent=1))

    info = res["datasets"]
    print(f"\n{'dataset':<11}{'n':>7}{'D':>5}{'anom%':>7}{'int.feat':>9}{'distinct':>9}"
          f"{'AUC':>7}{'anom.nbrs':>10}{'islands':>9}")
    for d in sorted(info, key=lambda d: (info[d]["D"], d)):
        v = info[d]
        isl = np.median([r["islands"] for r in rows if r["dataset"] == d
                         and r["method"] in ("umap", "tsne", "pacmap", "trimap")])
        print(f"{d:<11}{v['n']:>7}{v['D']:>5}{100 * v['anomalies'] / v['n']:>7.1f}"
              f"{v['integer_features']:>5}/{v['D']:<3}{v['distinct_scores_share']:>9.2f}"
              f"{v['auc_original']:>7.3f}{v['anomaly_neighbour_share_median']:>10.2f}{isl:>9.0f}")
    print("  (islands: median over the neighbour-based pictures of the 10-NN graph's "
          "strongly connected components)")

    print("\nPer method, mean over datasets of the per-dataset mean over seeds:")
    for label in ("masking_rate", "phantom_rate", "adjusted_overlap", "auc_retained",
                  "masking_rate_anomaly_budget", "masking_rate_multiscale"):
        if label in stats:
            print(f"   {label:<30}" + "  ".join(f"{m} {stats[label]['mean'][m]:.2f}"
                                               for m in stats[label]["methods"]))
    print("\nWhat the failures cost (version 1, top 5%, summed over datasets, mean over seeds):")
    for m in METHOD_SEEDS:
        lost = sum(np.mean([r["masked_anomalies"] for r in rows if r["dataset"] == d
                            and r["method"] == m]) for d in info)
        found = sum(np.mean([r["anomalies_in_a"] for r in rows if r["dataset"] == d
                             and r["method"] == m]) for d in info)
        phantom_anom = sum(np.mean([r["phantom_anomalies"] for r in rows if r["dataset"] == d
                                    and r["method"] == m]) for d in info)
        print(f"   {m:<7} masked anomalies {lost:7.1f} of {found:6.0f} in the original top 5% "
              f"({lost / found:.0%}); anomalies promoted as phantoms {phantom_anom:6.1f}")

    s = stats["masking_rate"]
    print(f"\nMasking rate, 2D, top 5%, mean over seeds ({s['N']} datasets)")
    print(f"{'dataset':<12}" + "".join(f"{m:>9}" for m in s["methods"]))
    for d, row in zip(s["datasets"], s["per_dataset_mean"]):
        print(f"{d:<12}" + "".join(f"{v:>9.2f}" for v in row))
    print(f"{'mean':<12}" + "".join(f"{s['mean'][m]:>9.2f}" for m in s["methods"]))
    print(f"{'avg rank':<12}" + "".join(f"{s['average_rank'][m]:>9.2f}" for m in s["methods"]))
    print(f"\nHeavy ties (distinct scores below {TIES_THRESHOLD:.0%} of n): {heavy or 'none'}")
    for label, *_ in ANALYSES:
        if label not in stats:
            continue
        t = stats[label]
        print(f"\n{label} ({t['N']} datasets): Friedman chi2 {t['friedman_chi2']:.1f}, "
              f"p {t['friedman_p']:.1e}; Iman-Davenport F {t['iman_davenport_F']:.1f}, "
              f"p {t['iman_davenport_p']:.1e}; Nemenyi CD {t['nemenyi_cd']:.2f}")
        print("   average rank: " + ", ".join(f"{m} {r:.2f}" for m, r in
                                             sorted(t["average_rank"].items(), key=lambda x: x[1])))
        for q in sorted(t["pairs"], key=lambda q: q["p_holm"]):
            if q["p_holm"] < 0.05:
                win, lose, w = ((q["a"], q["b"], q["wins_a"]) if q["wins_a"] >= q["wins_b"]
                                else (q["b"], q["a"], q["wins_b"]))
                print(f"   {win:>7} better than {lose:<7} on {w}/{t['N']} datasets, "
                      f"Holm p {q['p_holm']:.3f} (median difference {q['median_diff']:+.3f}, "
                      f"{q['a']} - {q['b']}; {q['within_noise_floor']} within the noise floor)")
    print("\nExpectations (docs/BENCHMARK.md, section 2):")
    for k, v in stats["expectations"].items():
        print(f"   {k}: {v}")
    print(f"\nWrote {OUT_STATS}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "embed"
    names = sys.argv[2:] or benchmark_names()
    refuse_development(names)
    if stage == "embed":
        embed(names)
    elif stage == "run":
        run(names)
    elif stage == "report":
        report()
    elif stage == "figures":
        from figures_week6 import draw_all
        draw_all()
    else:
        raise SystemExit(f"unknown stage {stage!r}")
