"""Figures for week 5. Reads results/week5_*; draws figures/week5_*.png.

    python src/figures_week5.py

Same colours and style as week 4 (figures_week4.py): masked = orange,
phantom = blue, retained = aqua, greys for ambiguous and background points.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import measure as M
from config import FIGURES, RESULTS
from figures_week4 import (AXIS, C_AMBIG, C_MASKED, C_PHANTOM, C_RETAINED, INK2, MUTED,
                           STYLE, SURFACE, _legend_handles, _scatter_by_category)

METHOD_LABEL = {"identity": "identity", "pca": "PCA", "random": "random proj.",
                "umap": "UMAP", "tsne": "t-SNE", "pacmap": "PaCMAP", "trimap": "TriMap"}
ORDER = ("pca", "random", "umap", "tsne", "pacmap", "trimap")
STOCHASTIC = ("random", "umap", "tsne", "pacmap", "trimap")


def _load():
    res = json.loads((RESULTS / "week5_revision.json").read_text())
    names = list(dict.fromkeys(r["dataset"] for r in res["rows"]))
    pts = {n: dict(np.load(RESULTS / f"week5_points_{n}.npz")) for n in names}
    return res, names, pts


def _mean(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    return float(np.nanmean(v)) if v.size else float("nan")


def _rate_axis(ax):
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    ax.grid(axis="x")


def versions(res, names, pts):
    """Masking and phantom rate under v1 (k = 20) and v2 (k = 20, 50, 100)."""
    main = [r for r in res["rows"] if r["budget"] == "5%" and "sweep" not in r]
    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.6), sharey=True)
    for col, name in enumerate(names):
        for row, (key, colour, label) in enumerate((("masking_rate", C_MASKED, "masking rate"),
                                                    ("phantom_rate", C_PHANTOM, "phantom rate"))):
            ax = axes[row, col]
            for i, method in enumerate(ORDER):
                v = {ver: _mean([r for r in main if r["dataset"] == name and r["method"] == method
                                 and r["version"] == ver], key) for ver in ("v1", "v2")}
                ax.annotate("", xy=(v["v2"], i), xytext=(v["v1"], i),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.2,
                                            shrinkA=4, shrinkB=4, mutation_scale=9),
                            zorder=2)
                ax.scatter([v["v1"]], [i], s=30, facecolor=SURFACE, edgecolor=colour, lw=1.4,
                           zorder=3)
                ax.scatter([v["v2"]], [i], s=30, color=colour, edgecolor=SURFACE, lw=1.2,
                           zorder=4)
            ax.set_yticks(range(len(ORDER)), [METHOD_LABEL[m] for m in ORDER])
            ax.set_ylim(len(ORDER) - 0.5, -0.5)
            _rate_axis(ax)
            ax.set_title(f"{name} · {label}", loc="left")
    handles = [Line2D([], [], ls="", marker="o", ms=6, mfc=SURFACE, mec=INK2, mew=1.4,
                      label="v1: kNN at k = 20"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=INK2, mec="none",
                      label="candidate (v2): kNN at k = 20, 50 and 100, the most outlying reading")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Version 1 against the multi-scale candidate, v2 (2D, top 5%, mean over seeds)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIGURES / "week5_versions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def maps(res, names, pts):
    """The calls in the pictures where the two versions differ most."""
    cases = [(n, m) for n in ("wbc", "fault") if n in pts for m in ("umap", "pacmap")
             if f"{m}_Y" in pts[n]]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases) // 2, 4, figsize=(12.4, 3.2 * (len(cases) // 2)),
                             squeeze=False)
    for j, (name, method) in enumerate(cases):
        row, col0 = divmod(j, 2)
        p = pts[name]
        Y = p[f"{method}_Y"]
        for dv, ver in enumerate(("v1", "v2")):
            ax = axes[row, 2 * col0 + dv]
            cat = p[f"{ver}_{method}_cat"]
            _scatter_by_category(ax, Y[:, 0], Y[:, 1], cat)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            nm, nph = int((cat == M.MASKED).sum()), int((cat == M.PHANTOM).sum())
            ax.set_title(f"{name} · {METHOD_LABEL[method]} · {ver}\n{nm} masked, {nph} phantom",
                         loc="left", fontsize=9.5)
    fig.legend(handles=_legend_handles(["masked", "phantom", "retained", "ambiguous", "background"]),
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The same pictures read by version 1 and by the multi-scale candidate, v2 "
                 "(top 5%, seed 0)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIGURES / "week5_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def seeds(res, names, pts):
    """How often each point of the original top 5% is masked over five seeds."""
    rows = res["seed_stability"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.3), sharey=True)
    for ax, name in zip(axes, names):
        for i, method in enumerate(STOCHASTIC):
            for dy, ver, alpha in ((-0.19, "v1", 0.45), (0.19, "v2", 1.0)):
                r = next((r for r in rows if r["dataset"] == name and r["method"] == method
                          and r["version"] == ver), None)
                if r is None:
                    continue
                left = 0.0
                for key, colour in (("share_masked_all5", C_MASKED), ("share_split", C_AMBIG),
                                    ("share_masked_none", C_RETAINED)):
                    ax.barh(i + dy, r[key], left=left, height=0.34, color=colour, alpha=alpha,
                            lw=0)
                    left += r[key]
                ax.text(1.02, i + dy, ver, va="center", fontsize=7, color=MUTED)
        ax.set_yticks(range(len(STOCHASTIC)), [METHOD_LABEL[m] for m in STOCHASTIC])
        ax.set_ylim(len(STOCHASTIC) - 0.5, -0.5)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
        ax.set_title(name, loc="left")
        ax.set_xlabel("share of the original top 5%")
    handles = [Patch(color=C_MASKED, label="masked under all five seeds"),
               Patch(color=C_AMBIG, label="masked under some seeds"),
               Patch(color=C_RETAINED, label="masked under none")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Seed stability of the per-point calls (2D, top 5%, seeds 0-4; "
                 "upper bar version 1, lower bar the candidate, v2)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIGURES / "week5_seeds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def redundancy(res, names, pts):
    """Does per-point continuity or trustworthiness already find the failures?"""
    rows = res["redundancy"]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(2.75 * (len(names) + 1), 3.3),
                             gridspec_kw={"width_ratios": [1] * len(names) + [1.25]})
    for ax, name in zip(axes, names):
        for i, method in enumerate(ORDER):
            r = next((r for r in rows if r["dataset"] == name and r["method"] == method), None)
            if r is None:
                continue
            for key, colour, dy in (("auc_continuity_masked", C_MASKED, -0.12),
                                    ("auc_trust_phantom", C_PHANTOM, 0.12)):
                if r[key] is not None:
                    ax.scatter([r[key]], [i + dy], s=30, color=colour, edgecolor=SURFACE,
                               lw=1.2, zorder=3)
        ax.axvline(0.5, color=AXIS, lw=0.9, zorder=1)
        ax.axvline(0.8, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_yticks(range(len(ORDER)), [METHOD_LABEL[m] for m in ORDER])
        ax.set_ylim(len(ORDER) - 0.5, -0.5)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(axis="x")
        ax.set_title(name, loc="left")
        ax.set_xlabel("AUC")
    for ax in axes[1:len(names)]:
        ax.set_yticklabels([])
    ax = axes[-1]
    markers = dict(zip(ORDER, ("o", "s", "^", "v", "D", "P")))
    for r in rows:
        ax.scatter(r["continuity"], r["masking_rate"], s=26, marker=markers[r["method"]],
                   color=C_MASKED, edgecolor=SURFACE, lw=0.8, alpha=0.9)
    ax.set_xlabel("continuity (k = 20, mean over points)")
    ax.set_ylabel("masking rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid()
    ax.set_title("aggregate, every dataset", loc="left")
    handles = [Line2D([], [], ls="", marker="o", ms=6, mfc=C_MASKED, mec="none",
                      label="1 - continuity separating masked from retained points"),
               Line2D([], [], ls="", marker="o", ms=6, mfc=C_PHANTOM, mec="none",
                      label="1 - trustworthiness separating phantom from retained points"),
               Line2D([], [], color=MUTED, ls=(0, (3, 3)), lw=0.8, label="AUC 0.8")]
    handles += [Line2D([], [], ls="", marker=markers[m], ms=5.5, mfc=INK2, mec="none",
                       label=METHOD_LABEL[m]) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Per-point trustworthiness and continuity against the candidate's calls, v2 "
                 "(2D, top 5%, seed 0)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(FIGURES / "week5_redundancy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_all() -> None:
    plt.rcParams.update(STYLE)
    res, names, pts = _load()
    for f in (versions, maps, seeds, redundancy):
        f(res, names, pts)
    print("Wrote " + ", ".join(f"figures/week5_{s}.png"
                               for s in ("versions", "maps", "seeds", "redundancy")))


if __name__ == "__main__":
    draw_all()
