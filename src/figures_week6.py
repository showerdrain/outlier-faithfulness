"""Figures for week 6. Reads results/week6_*; draws figures/week6_*.png.

    python src/benchmark.py figures

Version 1 throughout, except week6_readings.png, which sets the multi-scale
candidate beside it. Same colours and style as weeks 4 and 5: masked =
orange, phantom = blue, retained = aqua, greys for ambiguous and background.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import measure as M
from config import FIGURES, RESULTS
from figures_week4 import (AXIS, C_MASKED, C_PHANTOM, INK, INK2, MUTED, STYLE, SURFACE,
                           _legend_handles, _scatter_by_category)
from figures_week5 import METHOD_LABEL

ORDER = ("pca", "random", "umap", "tsne", "pacmap", "trimap")
MARKERS = dict(zip(ORDER, ("o", "s", "^", "v", "D", "P")))
NEIGHBOUR = ("umap", "tsne", "pacmap", "trimap")


def _load():
    res = json.loads((RESULTS / "week6_benchmark.json").read_text())
    stats = json.loads((RESULTS / "week6_stats.json").read_text())
    return res, stats


def _means(res, key, nested=None):
    """{(dataset, method): mean over seeds of key}."""
    out = {}
    for r in res["rows"]:
        v = r[nested][key] if nested else r[key]
        if v is not None:
            out.setdefault((r["dataset"], r["method"]), []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


def _datasets_by_dim(res):
    info = res["datasets"]
    return sorted(info, key=lambda d: (info[d]["D"], d))


def heatmap(res, stats):
    """Masking and phantom rate per dataset and method."""
    names = _datasets_by_dim(res)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 0.36 * len(names) + 2.0), sharey=True)
    for ax, key, colour, title in ((axes[0], "masking_rate", C_MASKED, "masking rate"),
                                   (axes[1], "phantom_rate", C_PHANTOM, "phantom rate")):
        mean = _means(res, key)
        A = np.array([[mean.get((d, m), np.nan) for m in ORDER] for d in names])
        A = np.vstack([A, np.nanmean(A, axis=0)])
        cmap = LinearSegmentedColormap.from_list("c", [SURFACE, colour])
        ax.imshow(A, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color=SURFACE if A[i, j] > 0.6 else INK,
                            fontweight="bold" if i == len(names) else "normal")
        ax.axhline(len(names) - 0.5, color=SURFACE, lw=3)
        ax.set_xticks(range(len(ORDER)), [METHOD_LABEL[m] for m in ORDER])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", pad=22)
    info = res["datasets"]
    axes[0].set_yticks(range(len(names) + 1),
                       [f"{d} (D = {info[d]['D']})" for d in names] + ["mean"])
    fig.suptitle("The benchmark at 2D, version 1: top 5%, mean over seeds, datasets by dimension",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "week6_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cliques(order, pairs):
    """Maximal runs of methods, in rank order, with no significant pair inside."""
    sig = {(q["a"], q["b"]) for q in pairs if q["p_holm"] < 0.05}
    sig |= {(b, a) for a, b in sig}
    out = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and all((order[x], order[j + 1]) not in sig
                                         for x in range(i, j + 1)):
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return out


def cd_diagram(ax, t, title):
    """Average ranks, Nemenyi's critical difference, Wilcoxon-Holm groups."""
    methods = sorted(t["average_rank"], key=lambda m: t["average_rank"][m])
    r = [t["average_rank"][m] for m in methods]
    k = len(methods)
    ax.set_xlim(0.6, k + 0.4)
    ax.set_ylim(-3.2, 1.25)
    ax.axis("off")
    ax.plot([1, k], [0, 0], color=INK2, lw=1)
    for x in range(1, k + 1):
        ax.plot([x, x], [0, 0.12], color=INK2, lw=1)
        ax.text(x, 0.22, str(x), ha="center", va="bottom", fontsize=8, color=INK2)
    cd = t["nemenyi_cd"]
    ax.plot([1, 1 + cd], [0.85, 0.85], color=INK, lw=1.4)
    for x in (1, 1 + cd):
        ax.plot([x, x], [0.77, 0.93], color=INK, lw=1.4)
    ax.text(1 + cd / 2, 0.98, f"Nemenyi CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    half = (k + 1) // 2
    for i, (m, x) in enumerate(zip(methods, r)):
        y = -0.55 - 0.45 * (i if i < half else k - 1 - i)
        side = 0.75 if i < half else k + 0.25
        ax.plot([x, x, side], [0, y, y], color=INK2, lw=0.9)
        ax.text(side + (-0.05 if i < half else 0.05), y, f"{METHOD_LABEL[m]}  {x:.2f}"
                if i < half else f"{x:.2f}  {METHOD_LABEL[m]}",
                ha="right" if i < half else "left", va="center", fontsize=8.5)
    for n, (a, b) in enumerate(_cliques(methods, t["pairs"])):
        y = -0.18 - 0.12 * n
        ax.plot([r[a] - 0.04, r[b] + 0.04], [y, y], color=INK, lw=3, solid_capstyle="round")
    ax.set_title(title, loc="left", fontsize=9.5)


def cd_diagrams(res, stats):
    keys = (("masking_rate", "masking rate (rank 1 = hides least)"),
            ("phantom_rate", "phantom rate (rank 1 = fewest phantoms)"),
            ("auc_retained", "AUC retained (rank 1 = keeps most of the detector's AUC)"))
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.4, 2.5 * len(keys)))
    for ax, (key, title) in zip(axes, keys):
        t = stats[key]
        cd_diagram(ax, t, f"{title}. Friedman p = {t['friedman_p']:.1e}")
    fig.text(0.01, 0.005, "Bars join methods with no significant difference in a pairwise "
             "Wilcoxon signed-rank test, Holm-corrected at 0.05.", fontsize=8, color=INK2)
    fig.suptitle(f"Average ranks over the {stats['masking_rate']['N']} benchmark datasets "
                 "(version 1, 2D, top 5%)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIGURES / "week6_ranks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def auc_vs_masking(res, stats):
    """The task-level summary against the per-point one."""
    mr = _means(res, "masking_rate")
    auc_a = {d: v["auc_original"] for d, v in res["datasets"].items()}
    auc_b = _means(res, "auc_projection")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for m in ORDER:
        pts = [(mr[(d, m)], auc_b[(d, m)] / auc_a[d]) for d in auc_a if (d, m) in mr]
        x, y = zip(*pts)
        ax.scatter(x, y, s=34, marker=MARKERS[m], label=METHOD_LABEL[m],
                   color=C_MASKED if m in NEIGHBOUR else INK2, edgecolor=SURFACE, lw=0.8,
                   alpha=0.9, zorder=3)
    ax.axhline(1, color=AXIS, lw=0.9, zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("masking rate (share of the original top 5% the picture hides)")
    ax.set_ylabel("AUC in the picture / AUC in the original space")
    ax.grid()
    ax.legend(loc="lower left", ncol=2)
    ax.set_title("One point per dataset and method, mean over seeds (version 1, 2D)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "week6_auc_vs_masking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_maps(res) -> list[str]:
    """The dataset where UMAP hides most more than PCA, and the one where it hides least."""
    mr = _means(res, "masking_rate")
    gap = {d: mr[(d, "umap")] - mr[(d, "pca")] for d in res["datasets"]
           if (d, "umap") in mr and (d, "pca") in mr}
    order = sorted(gap, key=gap.get)
    return [order[-1], order[0]] if len(order) > 1 else order


def maps(res, stats):
    from benchmark import DIM, LOAD
    from projections import project

    names = pick_maps(res)
    methods = ("pca", "random", "umap", "tsne")
    fig, axes = plt.subplots(len(names), len(methods),
                             figsize=(3.1 * len(methods), 3.2 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        p = np.load(RESULTS / f"week6_points_{name}.npz")
        for col, method in enumerate(methods):
            ax = axes[row, col]
            Y, _ = project(name, method=method, n_components=DIM, seed=0, verbose=False, **LOAD)
            cat = p[f"{method}_s0_cat"]
            _scatter_by_category(ax, Y[:, 0], Y[:, 1], cat)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            nm, nph = int((cat == M.MASKED).sum()), int((cat == M.PHANTOM).sum())
            ax.set_title(f"{name} · {METHOD_LABEL[method]}\n{nm} masked, {nph} phantom",
                         loc="left", fontsize=9.5)
    fig.legend(handles=_legend_handles(["masked", "phantom", "retained", "ambiguous", "background"]),
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Two benchmark datasets in 2D (version 1, top 5%, seed 0): where UMAP hides "
                 "most beyond PCA, and where it hides least", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIGURES / "week6_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def readings(res, stats):
    """Version 1 against the multi-scale candidate, per dataset and method."""
    v1 = _means(res, "masking_rate")
    ms = _means(res, "masking_rate", nested="multiscale")
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9, zorder=1)
    for m in ORDER:
        pts = [(v1[k], ms[k]) for k in v1 if k[1] == m and k in ms]
        x, y = zip(*pts)
        ax.scatter(x, y, s=34, marker=MARKERS[m], label=METHOD_LABEL[m],
                   color=C_MASKED if m in NEIGHBOUR else INK2, edgecolor=SURFACE, lw=0.8,
                   alpha=0.9, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ticks = [0, 0.25, 0.5, 0.75, 1]
    labels = ["0", "25%", "50%", "75%", "100%"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("masking rate, version 1 (k = 20)")
    ax.set_ylabel("masking rate, multi-scale candidate (k = 20, 50, 100)")
    ax.grid()
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Below the diagonal: the candidate sees less hidden\n"
                 "(one point per dataset and method, mean over seeds, 2D)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "week6_readings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_all() -> None:
    plt.rcParams.update(STYLE)
    res, stats = _load()
    for f in (heatmap, cd_diagrams, auc_vs_masking, readings, maps):
        f(res, stats)
    print("Wrote " + ", ".join(f"figures/week6_{s}.png"
                               for s in ("rates", "ranks", "auc_vs_masking", "readings", "maps")))


if __name__ == "__main__":
    draw_all()
