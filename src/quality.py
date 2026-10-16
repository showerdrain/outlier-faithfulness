"""Existing projection quality measures, per point, for the redundancy question.

Question 1 asks whether outlier faithfulness is redundant with the measures
that already exist. The standard pair is trustworthiness and continuity
(Venna and Kaski 2001), which ZADU (Jeon et al. 2023) also reports per
point. For point i with k nearest neighbours N_X(i) in the original space
and N_Y(i) in the projection, and r(i, j) the rank of j among i's neighbours:

    t(i) = 1 - 2 / (k (2n - 3k - 1)) * sum_{j in N_Y(i) \\ N_X(i)} (r_X(i, j) - k)
    c(i) = 1 - 2 / (k (2n - 3k - 1)) * sum_{j in N_X(i) \\ N_Y(i)} (r_Y(i, j) - k)

Low t(i): the picture puts strangers next to i (false neighbours). Low c(i):
the picture drops i's real neighbours (missing neighbours). Their means over
all points are the usual trustworthiness and continuity.

ZADU builds the full n x n ranking, which does not fit in memory at n =
20,000. This computes the same numbers row by row, for any subset of rows,
with ZADU's conventions: ranks from a stable sort with the point itself first,
so ties go by index. validate.py section I checks the result against ZADU by
absolute value.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _ranks_rows(Z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Full rank matrix for the given rows: rank[a, j] = position of j in row a."""
    D = cdist(Z[rows], Z)
    D[np.arange(len(rows)), rows] = -np.inf          # the point itself goes first
    order = np.argsort(D, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(Z.shape[0])[None, :], axis=1)
    return rank


def local_trust_cont(X: np.ndarray, Y: np.ndarray, k: int = 20,
                     rows: np.ndarray | None = None,
                     chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-point trustworthiness and continuity for the given rows (default all)."""
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    n = len(X)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    norm = 2.0 / (k * (2 * n - 3 * k - 1))
    t = np.empty(len(rows))
    c = np.empty(len(rows))
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        rx, ry = _ranks_rows(X, r), _ranks_rows(Y, r)
        in_x = (rx >= 1) & (rx <= k)                   # the k nearest, self excluded
        in_y = (ry >= 1) & (ry <= k)
        t[s:s + chunk] = 1 - norm * np.where(in_y & ~in_x, rx - k, 0).sum(axis=1)
        c[s:s + chunk] = 1 - norm * np.where(in_x & ~in_y, ry - k, 0).sum(axis=1)
    return t, c
