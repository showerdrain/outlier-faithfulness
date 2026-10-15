"""Outlier faithfulness, placeholder version.

This is the first draft of the measure the thesis is about. It WILL be
replaced; the proposal says the definition is unsettled, and week 4 is
scheduled for a proper one. The point of this version is to prove the
pipeline runs end to end and to make the failure modes visible.

Definition, per point i, with ranks taken over outlier score ascending so
that rank n is the most outlying point:

    OF_i = ( rank_LD(i) - rank_HD(i) ) / n

    OF_i < 0   the projection pushed the point toward the crowd
    OF_i > 0   the projection pushed the point toward the periphery

The two directions are reported separately rather than combined. They are
different failures with different costs: a masked outlier is an anomaly the
analyst never sees, a phantom outlier is analyst time spent on nothing.

The aggregate is computed over the top ranks only, because those are the
points anyone actually inspects. The budget is the true number of anomalies,
the standard P@n convention in the outlier detection literature.

TIES ARE AN OPEN PROBLEM. A hard top-k cut is only well defined if no points
share a score across the boundary. On shuttle that assumption fails badly:
49,097 points take only 7,580 distinct kNN-distance values, and four points
sit on the boundary score exactly. Which of them falls inside the cut is then
decided by the sort, not by the data.

Sorting here is forced stable so the answer is at least reproducible, and
boundary_ties is reported so fragile cases are visible rather than silent.
That is a patch, not a solution. The real version needs an explicit policy:
break ties by original-space rank, cut on score value rather than position,
or report the tied band as ambiguous. See Schubert, Wojdanowski, Zimek &
Kriegel (SDM 2012) on evaluating outlier rankings, which treats this directly.

Tie DIAGNOSTICS (boundary_ties, distinct_scores) compare with a relative
tolerance, TIE_RTOL, never exact equality. The same distances computed by two
routes agree to about 1e-13 but not bit for bit, so exact-equality counts
differed between Windows and Linux (backdoor: 84,839 vs 84,837 distinct).
Differences below 1e-9 relative carry no meaning for an outlier ranking.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata

TIE_RTOL = 1e-9


def n_distinct(s: np.ndarray, rtol: float = TIE_RTOL) -> int:
    """Number of distinct values, treating values within rtol as equal."""
    v = np.sort(np.asarray(s, dtype=np.float64))
    if v.size == 0:
        return 0
    gaps = np.diff(v) > rtol * np.maximum(np.abs(v[1:]), np.finfo(float).tiny)
    return 1 + int(gaps.sum())


def top_k(s: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Indices of the k most outlying points, plus the tie count at the cut.

    Stable sort, so ties resolve by index and the result does not depend on
    the sorting implementation. numpy's default quicksort is unstable and
    gave different answers on different machines for exactly this reason.
    """
    order = np.argsort(-s, kind="stable")[:k]
    boundary = s[order[-1]]
    n_tied = int(np.isclose(s, boundary, rtol=TIE_RTOL, atol=0.0).sum())
    return order, (n_tied if n_tied > 1 else 0)


def signed_displacement(s_hd: np.ndarray, s_ld: np.ndarray) -> np.ndarray:
    """Per-point signed rank displacement, in [-1, 1]. Negative = masked."""
    n = len(s_hd)
    return (rankdata(s_ld) - rankdata(s_hd)) / n


def topk_failures(s_hd: np.ndarray, s_ld: np.ndarray, k: int) -> dict:
    """Which points enter or leave the analyst's shortlist.

    masked  : in the original space's top k, absent from the projection's
    phantom : in the projection's top k, absent from the original space's

    With equal budgets the two counts are necessarily equal, so the rate is
    reported once. What differs, and what matters, is WHICH points.
    """
    order_hd, ties_hd = top_k(s_hd, k)
    order_ld, ties_ld = top_k(s_ld, k)
    top_hd, top_ld = set(order_hd.tolist()), set(order_ld.tolist())
    masked = sorted(top_hd - top_ld)
    phantom = sorted(top_ld - top_hd)
    return {
        "k": k,
        "overlap": len(top_hd & top_ld),
        "rate": len(masked) / k,
        "boundary_ties_hd": ties_hd,
        "boundary_ties_ld": ties_ld,
        "masked_idx": masked,
        "phantom_idx": phantom,
    }


def summarise(s_hd, s_ld, y, k=None) -> dict:
    """One dataset, one projection: the numbers worth reporting."""
    y = np.asarray(y).astype(bool)
    k = int(y.sum()) if k is None else k
    of = signed_displacement(s_hd, s_ld)
    tk = topk_failures(s_hd, s_ld, k)

    # Of the true anomalies the original space ranked highly, how many did
    # the projection lose? This is the number with a cost attached.
    order_hd, _ = top_k(s_hd, k)
    order_ld, _ = top_k(s_ld, k)
    true_in_top_hd = order_hd[y[order_hd]]
    top_ld_set = set(order_ld.tolist())
    lost = [int(i) for i in true_in_top_hd if int(i) not in top_ld_set]

    return {
        "k": k,
        "shortlist_overlap": tk["overlap"] / k,
        "of_median_anomalies": float(np.median(of[y])),
        "of_median_inliers": float(np.median(of[~y])),
        "of_p05_anomalies": float(np.percentile(of[y], 5)),
        "true_anomalies_found_hd": int(len(true_in_top_hd)),
        "true_anomalies_lost": len(lost),
        "boundary_ties_hd": tk["boundary_ties_hd"],
        "boundary_ties_ld": tk["boundary_ties_ld"],
        "distinct_scores_hd": n_distinct(s_hd),
        "lost_idx": lost,
        "masked_idx": tk["masked_idx"],
        "phantom_idx": tk["phantom_idx"],
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from config import RESULTS, FIGURES, DEFAULT_K
    from datasets import load_dataset
    from neighbors import get_knn, knn_score
    from projections import project, knn_dist

    names = ["wbc", "ionosphere", "cardio", "musk",
             "satimage2", "satellite", "shuttle", "backdoor"]
    method, k = "pca", DEFAULT_K
    rows, panels = [], {}

    for name in names:
        dist_hd, _, meta = get_knn(name, k=k, verbose=False)
        _, y, _ = load_dataset(name)
        s_hd = knn_score(dist_hd, k)
        Y, _ = project(name, method=method, n_components=2, verbose=False)
        s_ld = knn_score(knn_dist(Y, k), k)

        r = summarise(s_hd, s_ld, y)
        r.update(dataset=name, n=meta["n"], D=meta["D"])
        rows.append(r)
        panels[name] = (Y, y, r)

    hdr = (f"{'dataset':<12}{'k':>6}{'overlap':>9}{'found':>7}{'lost':>6}"
           f"{'OF med':>9}{'OF p05':>9}{'distinct':>10}{'ties':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        ties = max(r["boundary_ties_hd"], r["boundary_ties_ld"])
        flag = f"{ties}" if ties else "-"
        print(f"{r['dataset']:<12}{r['k']:>6}{100*r['shortlist_overlap']:>8.0f}%"
              f"{r['true_anomalies_found_hd']:>7}{r['true_anomalies_lost']:>6}"
              f"{r['of_median_anomalies']:>9.3f}{r['of_p05_anomalies']:>9.3f}"
              f"{r['distinct_scores_hd']:>10}{flag:>6}")
    print("\n  distinct = distinct original-space score values out of n")
    print("  ties     = points sharing the boundary score; the cut is arbitrary among them")

    show = ["shuttle", "backdoor", "musk", "ionosphere"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    for ax, name in zip(axes, show):
        Y, y, r = panels[name]
        masked = np.array(r["masked_idx"], dtype=int)
        phantom = np.array(r["phantom_idx"], dtype=int)
        ax.scatter(Y[:, 0], Y[:, 1], s=3, c="#d8d8d8", lw=0, rasterized=True)
        if len(phantom):
            ax.scatter(Y[phantom, 0], Y[phantom, 1], s=14, c="#3b7dd8",
                       lw=0, label=f"phantom ({len(phantom)})")
        if len(masked):
            ax.scatter(Y[masked, 0], Y[masked, 1], s=14, c="#c0392b",
                       lw=0, label=f"masked ({len(masked)})")
        ax.set_title(f"{name}  (D={r['D']}, overlap {100*r['shortlist_overlap']:.0f}%)",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=False, markerscale=1.6)
    fig.suptitle("PCA to 2D: which points enter and leave the analyst's shortlist",
                 fontsize=13)
    fig.tight_layout()
    out_fig = FIGURES / "step7_pca_shortlist.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")

    slim = [{kk: vv for kk, vv in r.items() if not kk.endswith("_idx")} for r in rows]
    out = RESULTS / "step7_placeholder_measure.json"
    out.write_text(json.dumps(slim, indent=2))
    print(f"\nWrote {out}\nWrote {out_fig}")
