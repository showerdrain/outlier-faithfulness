"""Figures for the week-4 exploration. Reads results/week4_*; draws figures/week4_*.png.

    python src/figures_week4.py

Colours are fixed by role, never by rank: masked = orange, phantom = blue,
retained = aqua, and neutral greys for ambiguous and background points. These
are the first three slots of a palette validated for colour-vision
deficiency when all pairs appear together, as they do in a scatterplot.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import measure as M
from config import FIGURES, RESULTS

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_MASKED, C_PHANTOM, C_RETAINED = "#eb6834", "#2a78d6", "#1baf7a"
C_AMBIG, C_BACK = "#a9a8a1", "#dcdbd4"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "font.size": 9, "legend.fontsize": 8.5, "legend.frameon": False,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
}
METHOD_LABEL = {"identity": "identity", "pca": "PCA", "random": "random proj.",
                "umap": "UMAP", "tsne": "t-SNE", "pacmap": "PaCMAP"}


def _load():
    res = json.loads((RESULTS / "week4_measure.json").read_text())
    names = list(dict.fromkeys(r["dataset"] for r in res["rows"]))
    pts = {n: dict(np.load(RESULTS / f"week4_points_{n}.npz")) for n in names}
    return res, names, pts


def _legend_handles(kinds):
    spec = {"masked": (C_MASKED, "masked"), "phantom": (C_PHANTOM, "phantom"),
            "retained": (C_RETAINED, "retained"),
            "ambiguous": (C_AMBIG, "crossed the boundary within noise"),
            "background": (C_BACK, "neither shortlist")}
    return [Line2D([], [], ls="", marker="o", ms=6, mfc=spec[k][0], mec="none",
                   label=spec[k][1]) for k in kinds]


def _scatter_by_category(ax, x, y, cat):
    layers = [(M.BACKGROUND, C_BACK, 2), (M.EXIT_AMBIGUOUS, C_AMBIG, 7),
              (M.ENTRY_AMBIGUOUS, C_AMBIG, 7), (M.RETAINED, C_RETAINED, 9),
              (M.PHANTOM, C_PHANTOM, 11), (M.MASKED, C_MASKED, 11)]
    for code, colour, size in layers:
        sel = cat == code
        if sel.any():
            ax.scatter(x[sel], y[sel], s=size, c=colour, lw=0, rasterized=True,
                       zorder=2 + code % 3 if code != M.BACKGROUND else 1)


def funnel(res, names, pts):
    """Displacement against original-space rank, PCA to 2D, with the noise envelope."""
    rows = {r["dataset"]: r for r in res["rows"]
            if r["method"] == "pca" and r["dim"] == 2 and r["budget"] == "5%" and "sweep" not in r}
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.5), sharey=True)
    for ax, name in zip(axes, names):
        p = pts[name]
        rho_a, lam, cat, q = p["rho_a"], p["pca_lam"], p["pca_cat"], p["q"]
        n = len(rho_a)
        edges = np.unique(np.geomspace(1, n + 1, 40))
        mid, env = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (rho_a >= lo) & (rho_a < hi)
            if sel.sum() >= 3:
                mid.append(np.sqrt(lo * hi))
                env.append(np.median(q[sel]))
        _scatter_by_category(ax, rho_a, lam, cat)
        env = np.array(env)
        ax.fill_between(mid, -env, env, color=INK2, alpha=0.22, lw=0, zorder=1.5)
        for sign in (-1, 1):
            ax.plot(mid, sign * env, color=INK2, lw=0.8, zorder=1.6)
        m = rows[name]["m"]
        ax.axvline(m, color=MUTED, lw=0.8, zorder=1)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
        ax.text(m * 1.15, 6.6, "top 5%", color=INK2, fontsize=8, va="top")
        ax.set_xscale("log")
        ax.set_xlim(0.8, n * 1.3)
        ax.set_ylim(-8, 8)
        ax.grid(axis="y")
        r = rows[name]
        ax.set_title(f"{name}\nmasking {r['masking_rate']:.0%} · phantom {r['phantom_rate']:.0%}",
                     loc="left")
        ax.set_xlabel("rank in the original space")
    axes[0].set_ylabel("displacement λ (bits)\n← toward the crowd · toward the edge →")
    fig.legend(handles=_legend_handles(["masked", "phantom", "retained", "ambiguous", "background"]),
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PCA to 2D: how far each point moves in the outlier ranking. "
                 "Grey band: the noise null's 95% one-sided threshold.",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIGURES / "week4_funnel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def maps(res, names, pts):
    """Where the failures sit in the picture an analyst would look at."""
    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.6))
    for col, name in enumerate(names):
        for row, method in enumerate(("pca", "umap")):
            ax = axes[row, col]
            p = pts[name]
            if f"{method}_Y" not in p:
                ax.axis("off")
                continue
            Y, cat = p[f"{method}_Y"], p[f"{method}_cat"]
            _scatter_by_category(ax, Y[:, 0], Y[:, 1], cat)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            nm, nph = int((cat == M.MASKED).sum()), int((cat == M.PHANTOM).sum())
            ax.set_title(f"{name} · {METHOD_LABEL[method]}\n{nm} masked, {nph} phantom",
                         loc="left", fontsize=9.5)
    fig.legend(handles=_legend_handles(["masked", "phantom", "retained", "ambiguous", "background"]),
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The shortlist failures in the 2D pictures (top 5% budget, seed 0)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIGURES / "week4_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def methods(res, names, pts):
    """Masking and phantom rates per method: the two failure modes side by side."""
    main = [r for r in res["rows"] if r["budget"] == "5%" and "sweep" not in r]
    order = [m for m in ("identity", "pca", "random", "umap", "tsne", "pacmap")
             if any(r["method"] == m for r in main)]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 0.45 * len(order) + 1.6),
                             sharey=True)
    for ax, name in zip(axes, names):
        for i, method in enumerate(order):
            rs = [r for r in main if r["dataset"] == name and r["method"] == method]
            for key, colour, dy in (("masking_rate", C_MASKED, 0.13), ("phantom_rate", C_PHANTOM, -0.13)):
                v = np.array([r[key] for r in rs], float)
                if len(v) > 1:
                    ax.plot([v.min(), v.max()], [i + dy, i + dy], color=colour, lw=1.4,
                            solid_capstyle="round", zorder=2)
                ax.scatter([v.mean()], [i + dy], s=34, color=colour, edgecolor=SURFACE,
                           linewidth=1.5, zorder=3)
        ax.set_yticks(range(len(order)), [METHOD_LABEL[m] for m in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.grid(axis="x")
        ax.set_title(name, loc="left")
    handles = [Line2D([], [], ls="", marker="o", ms=6, mfc=C_MASKED, mec="none",
                      label="masking rate: share of the original top 5% the picture hides"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=C_PHANTOM, mec="none",
                      label="phantom rate: share of the picture's top 5% not outlying originally"),
               Line2D([], [], color=MUTED, lw=1.4, label="range over seeds")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Two failure modes, reported separately (2D, top 5% budget; dot = mean over seeds)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(FIGURES / "week4_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def dimension(res, names, pts):
    """PCA target dimension: failures must shrink to zero at d = D."""
    sweep = [r for r in res["rows"] if r.get("sweep") == "dimension"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 2.9), sharey=True)
    for ax, name in zip(axes, names):
        rs = sorted((r for r in sweep if r["dataset"] == name), key=lambda r: r["dim"])
        d = [r["dim"] for r in rs]
        for key, colour, label in (("masking_rate", C_MASKED, "masking rate"),
                                   ("phantom_rate", C_PHANTOM, "phantom rate")):
            ax.plot(d, [r[key] for r in rs], color=colour, lw=2, marker="o", ms=4.5,
                    mec=SURFACE, mew=1.2, label=label, solid_joinstyle="round")
        ax.axvline(2, color=MUTED, lw=0.8)
        ax.set_xscale("log", base=2)
        ticks = [t for t in (1, 2, 4, 8, 16, 32) if t <= max(d)] + ([max(d)] if max(d) not in (1, 2, 4, 8, 16, 32) else [])
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.grid(axis="y")
        ax.set_title(f"{name} (D = {max(d)})", loc="left")
        ax.set_xlabel("PCA target dimension")
    axes[0].legend(loc="upper right")
    fig.suptitle("PCA: failures fall as dimensions are added, and vanish at full dimension (top 5%)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "week4_dimension.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def budget(res, names, pts):
    """Sensitivity to the shortlist size."""
    sweep = [r for r in res["rows"] if r.get("sweep") == "budget" and r["budget"] != "anomalies"]
    meths = [m for m in ("pca", "umap") if any(r["method"] == m for r in sweep)]
    fig, axes = plt.subplots(len(meths), len(names), figsize=(3.1 * len(names), 2.7 * len(meths)),
                             sharey=True, squeeze=False)
    for i, method in enumerate(meths):
        for ax, name in zip(axes[i], names):
            rs = sorted((r for r in sweep if r["dataset"] == name and r["method"] == method),
                        key=lambda r: r["m"])
            x = [r["m"] / r["n"] for r in rs]
            for key, colour, label in (("masking_rate", C_MASKED, "masking rate"),
                                       ("phantom_rate", C_PHANTOM, "phantom rate")):
                ax.plot(x, [r[key] for r in rs], color=colour, lw=2, marker="o", ms=4.5,
                        mec=SURFACE, mew=1.2, label=label)
            ax.set_xscale("log")
            ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2], ["1%", "2%", "5%", "10%", "20%"])
            ax.minorticks_off()
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0, 0.5, 1], ["0", "50%", "100%"])
            ax.grid(axis="y")
            ax.set_title(f"{name} · {METHOD_LABEL[method]}", loc="left")
            if i == len(meths) - 1:
                ax.set_xlabel("shortlist budget (share of points)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Budget sensitivity at 2D (seed 0)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "week4_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_all() -> None:
    plt.rcParams.update(STYLE)
    res, names, pts = _load()
    for f in (funnel, maps, methods, dimension, budget):
        f(res, names, pts)
    print("Wrote " + ", ".join(f"figures/week4_{s}.png"
                               for s in ("funnel", "maps", "methods", "dimension", "budget")))


if __name__ == "__main__":
    draw_all()
