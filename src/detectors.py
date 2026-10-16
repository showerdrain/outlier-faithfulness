"""Outlier detectors read off a neighbour graph, at one scale or several.

Every detector here takes the same input: a neighbour graph (dist, idx), each
point's nearest neighbours in increasing distance with the point itself
excluded. The original space, every projection and every replicate of the
noise null are scored through this one code path, so a difference between
two rankings can only come from the data, never from how it was scored.

  knn   distance to the k-th nearest neighbour. Matches
        pyod.models.knn.KNN(method="largest") exactly (validate.py, A).
  lof   local outlier factor (Breunig, Kriegel, Ng and Sander, SIGMOD 2000),
        computed from the same graph. Matches scikit-learn's
        LocalOutlierFactor by absolute value (validate.py, I).

MULTI-SCALE READING (candidate version 2 of the measure; tested in week 5 and
not adopted, so it is reported beside version 1, docs/MEASURE.md section 9).
One neighbourhood size k fixes the scale at which "apart from the rest" is
judged. UMAP and PaCMAP gather isolated outliers into small tight groups; at
k = 20 the members of a group of 40 have their 20th neighbour inside the
group and look dense, at k = 50 the group as a whole is seen to be apart. A
person looking at the picture sees both. So the candidate scores every point
at each k in SCALES, divides each score by its median over all points (how
many times more isolated than a typical point, at that scale), and takes the
maximum over the scales. The same rule is applied to the original space, so
the comparison still runs the same detector on both sides. The precedent is Breunig et al. (2000), who rank
points by their maximum LOF over a range of MinPts values; LOF is already a
ratio to the neighbourhood, so dividing by its median changes little there.

Why ratios and not ranks: an earlier candidate took each point's best rank
over the scales. On a test with a known answer (validate.py, I) it let the
scales split the shortlist, so half of a far-away clump of outliers still
counted as hidden: the top 20 at k = 20 (mild edge points of the bulk) took
as many places as the top 20 at k = 50 (the clump). Ratios keep the size of
the gap, and the clump wins.

Scales larger than (n - 1) // 2 are dropped, so the noise null, which scores
against random halves with k // 2 neighbours, stays well defined.
"""

from __future__ import annotations

import numpy as np

SCALES_V1 = (20,)
SCALES_V2 = (20, 50, 100)
DETECTORS = ("knn", "lof")


def usable_scales(scales: tuple[int, ...], n: int) -> tuple[int, ...]:
    """The scales that fit a dataset of n points (see the module note)."""
    out = tuple(k for k in scales if k <= (n - 1) // 2)
    if not out:
        raise ValueError(f"no scale in {scales} fits n = {n}")
    return out


def lof_from_graph(dist: np.ndarray, idx: np.ndarray, k: int,
                   rtol: float = 1e-9) -> np.ndarray:
    """Local outlier factor at k from a neighbour graph. Larger = more outlying.

    Breunig et al.'s definition, ties included: a point's neighbourhood is
    every point within its k-distance, so it can hold more than k points when
    distances tie. scikit-learn takes exactly k and breaks ties arbitrarily,
    which made LOF move under an exact rotation of wbc, whose integer features
    tie many distances; with ties included it does not. Distances within rtol
    of the k-distance count as tied, as everywhere else in this project. On
    data without ties at the k-th neighbour this equals scikit-learn exactly
    (validate.py, I). The neighbourhood can only reach as far as the graph's
    stored columns.

    Every neighbour o of a point must have its own row in the graph: its
    k-distance and local reachability density come from that row. This holds
    for a full graph and for a noise-null replicate, where every point's
    neighbours lie in the reference half and every reference point has a row.
    The 1e-10 guard against coincident points follows scikit-learn.
    """
    d = np.asarray(dist, dtype=np.float64)
    i = np.asarray(idx)
    k_distance = d[:, k - 1]
    member = d <= k_distance[:, None] * (1 + rtol)
    member[:, :k] = True
    size = member.sum(axis=1)
    reach = np.maximum(d, k_distance[i])
    lrd = 1.0 / ((reach * member).sum(axis=1) / size + 1e-10)
    return (lrd[i] * member).sum(axis=1) / size / lrd


def scale_scores(dist: np.ndarray, idx: np.ndarray | None, detector: str,
                 scales: tuple[int, ...]) -> list[np.ndarray]:
    """One score vector per scale, larger = more outlying."""
    if detector == "knn":
        return [np.asarray(dist[:, k - 1], dtype=np.float64) for k in scales]
    if detector == "lof":
        if idx is None:
            raise ValueError("lof needs the neighbour indices")
        return [lof_from_graph(dist, idx, k) for k in scales]
    raise KeyError(f"unknown detector {detector!r}; known: {DETECTORS}")


def combine(scores: list[np.ndarray]) -> np.ndarray:
    """Maximum over the scales of each score divided by its median.

    One scale returns its score unchanged, so version 1 is reproduced exactly.
    """
    if len(scores) == 1:
        return np.asarray(scores[0], dtype=np.float64)
    ratios = [np.asarray(s, dtype=np.float64) / max(float(np.median(s)), 1e-300)
              for s in scores]
    return np.max(np.column_stack(ratios), axis=1)


def score_graph(dist: np.ndarray, idx: np.ndarray | None, detector: str = "knn",
                scales: tuple[int, ...] = SCALES_V2) -> np.ndarray:
    """Scores from a neighbour graph holding at least max(scales) neighbours."""
    scales = usable_scales(scales, len(dist))
    return combine(scale_scores(dist, idx, detector, scales))


def score_space(X: np.ndarray, detector: str = "knn",
                scales: tuple[int, ...] = SCALES_V2) -> np.ndarray:
    """Scores for points given as coordinates (a projection, or any matrix)."""
    from neighbors import knn_graph

    scales = usable_scales(scales, len(X))
    dist, idx = knn_graph(np.asarray(X), max(scales))
    return combine(scale_scores(dist, idx, detector, scales))


def score_original(name: str, detector: str = "knn",
                   scales: tuple[int, ...] = SCALES_V2, **loader_kw) -> np.ndarray:
    """Scores in the original space, from the cached neighbour graph."""
    from neighbors import get_knn

    _, _, meta = get_knn(name, k=1, verbose=False, **loader_kw)
    scales = usable_scales(scales, meta["n"])
    dist, idx, _ = get_knn(name, k=max(scales), verbose=False, **loader_kw)
    return combine(scale_scores(dist, idx, detector, scales))
