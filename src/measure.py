"""Outlier faithfulness, version 1 (week 4, October 2026; kept in week 5).

WHAT IT MEASURES. Whether a projection preserves the outlyingness of
individual points. The same detector, with the same parameters, scores the
original space and the projected coordinates, and each point's position in
the two outlier rankings is compared. The detector is the distance to the
k-th nearest neighbour (k = 20); nothing below depends on that choice.
"Outlying in a projection" is therefore operationalised as "ranked highly by
the same detector run on the projected coordinates". That is a reproducible
algorithmic proxy, not visual salience; the gap between the two is still
open.

WEEK 5 tested a revision, a multi-scale reading (candidate version 2): every
space read at k = 20, 50 and 100, each reading divided by its median, each
point's largest ratio kept (detectors.py). It missed two of the four
adoption criteria fixed beforehand, both narrowly, so version 1 stands
(docs/MEASURE.md, section 9). The code below is written for any score and
any set of scales, so the candidate runs through it unchanged: scores from
detectors.score_original() and detectors.score_space(), the null from
get_null(..., scales=detectors.SCALES_V2). With one scale, every function is
version 1 exactly (validate.py, section I).

The full argument for each choice, the alternatives tried and the evidence
from the development datasets are in docs/MEASURE.md. In brief:

Notation: a = detector scores in the original space, b = in the projection,
n points, m = the shortlist budget (how many points an analyst inspects).

1. RANKS. rho(i) is the descending rank of point i, 1 = most outlying.
   Scores within a relative tolerance TIE_RTOL count as tied, and tied points
   share the average of the positions they span. Ties are resolved by the
   data, never by the sort, so the answer cannot depend on the machine.

2. DISPLACEMENT, in bits:

       lambda(i) = log2( rho_a(i) / rho_b(i) )

   lambda < 0  the projection pushed the point toward the crowd (masking side)
   lambda > 0  it pushed the point toward the periphery (phantom side)

   |lambda| counts rank doublings: lambda = -3 puts the point eight times
   deeper in the projection's ranking than in the original's. It weights the
   top of the ranking, where an analyst looks: rank 1 -> 64 is -6 bits, rank
   1001 -> 1064 is -0.09. The placeholder, (rank_LD - rank_HD) / n, scored
   both moves the same (63 / n), so the bulk of inliers dominated it.

3. SHORTLISTS. S_a = {i : rho_a(i) <= m}, S_b = {i : rho_b(i) <= m}. A tied
   group shares one rank, so it is wholly in or wholly out, and a shortlist
   can differ from m by at most half a tied group. This replaces the stable
   sort, which made the cut reproducible but not principled.

4. NOISE NULL. Not every move across the shortlist boundary is the
   projection's doing: a point's rank is only as precise as the sample it
   was estimated from. NULL_B replicate rankings of the ORIGINAL space are
   made, each scoring every point against a random half of the data with
   k // 2 neighbours, so the neighbourhood holds the same share of the data.
   The competitor set stays all n points, as in the real comparison. For a
   point i, the null distribution of displacement is the set of pairwise
   differences of its log2-ranks across replicates. Half-sampling gives the
   variance of a full-sample estimate, as the bootstrap does, without
   duplicating points (duplicates put zero distances into the neighbour
   graph). The null depends on the data and the detector only, never on the
   projection, so it is computed once per dataset and cached.

5. CALLS at budget m and level ALPHA, with p(i) = P_null(|D_i| >= |lambda_i|) / 2,
   the one-sided p-value in the direction the point moved:

       masked     in S_a, not in S_b, p < ALPHA   hidden beyond noise
       phantom    in S_b, not in S_a, p < ALPHA   promoted beyond noise
       ambiguous  crossed the boundary, but within noise
       retained   in both shortlists

   Masked and phantom counts are no longer equal by construction, as they
   were under the placeholder's equal budgets.

6. AGGREGATES.
       masking rate  MR = |masked| / |S_a|   share of the detector's shortlist
                                             that the projection hides
       phantom rate  PR = |phantom| / |S_b|  share of the projection's
                                             shortlist that is not outlying
       adjusted overlap  AO = (O - E) / (min(|S_a|, |S_b|) - E),
                     O = |S_a & S_b|, E = |S_a| |S_b| / n, the expected
                     overlap when the projected scores are permuted across
                     points. 1 = same shortlist, 0 = no better than chance.
                     The form follows the chance adjustment of P@n in Campos
                     et al. (2016). p_perm is the hypergeometric tail.
   Reported alongside, as the closest existing measure of agreement between
   two outlier rankings: the weighted correlation of Schubert, Wojdanowski,
   Zimek & Kriegel (SDM 2012), on rank-normalised scores, with their
   class-balancing weights (the union of the two shortlists against the rest).

7. GROUND TRUTH. The measure itself never uses labels: it asks whether the
   projection preserves what the detector sees, so a labelled anomaly that is
   not outlying in the original space cannot be "masked". That is the
   conditioning on original-space outlyingness the proposal asked for. Labels
   are used afterwards, to say what the failures cost: masked anomalies are
   detections lost to the projection, phantom anomalies are anomalies the
   projection surfaces that the detector ranked low.

Tie DIAGNOSTICS compare with a relative tolerance, TIE_RTOL, never exact
equality. The same distances computed by two routes agree to about 1e-13 but
not bit for bit, so exact-equality counts differed between Windows and Linux.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import hypergeom

TIE_RTOL = 1e-9
ALPHA = 0.05            # one-sided level of each masked / phantom call
BUDGET_FRAC = 0.05      # default shortlist: the top 5% of points
NULL_B = 100            # replicate rankings in the noise null
NULL_SEED = 0
NULL_VERSION = "v1"     # bump if null_log_ranks() changes, so caches rebuild

RETAINED, MASKED, EXIT_AMBIGUOUS, PHANTOM, ENTRY_AMBIGUOUS, BACKGROUND = range(6)
CATEGORY_NAMES = ("retained", "masked", "exit_ambiguous",
                  "phantom", "entry_ambiguous", "background")


# --- ranks and ties ---------------------------------------------------------

def _tie_groups(v_desc: np.ndarray, rtol: float) -> np.ndarray:
    """Group ids for values sorted in descending order; within rtol = tied."""
    v = v_desc
    new = np.ones(len(v), dtype=bool)
    if len(v) > 1:
        scale = np.maximum(np.maximum(np.abs(v[:-1]), np.abs(v[1:])),
                           np.finfo(float).tiny)
        new[1:] = (v[:-1] - v[1:]) > rtol * scale
    return np.cumsum(new) - 1


def n_distinct(s: np.ndarray, rtol: float = TIE_RTOL) -> int:
    """Number of distinct values, treating values within rtol as equal."""
    v = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    return 0 if v.size == 0 else int(_tie_groups(v, rtol)[-1] + 1)


def desc_ranks(s: np.ndarray, rtol: float = TIE_RTOL) -> np.ndarray:
    """Descending ranks, 1 = highest score; tied scores share their average rank.

    Two scores are tied when they differ by at most rtol relative to the
    larger. The result does not depend on the order of the input or on the
    sorting algorithm, which is what makes it portable across machines.
    """
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    order = np.argsort(-s, kind="stable")
    gid = _tie_groups(s[order], rtol)
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    ends = np.r_[starts[1:], n] - 1
    avg = (starts + ends) / 2.0 + 1.0
    rho = np.empty(n)
    rho[order] = avg[gid]
    return rho


def displacement(rho_a: np.ndarray, rho_b: np.ndarray) -> np.ndarray:
    """lambda = log2(rho_a / rho_b) in bits. Negative = pushed toward the crowd."""
    return np.log2(np.asarray(rho_a, float) / np.asarray(rho_b, float))


def budget(n: int, frac: float = BUDGET_FRAC) -> int:
    """Default shortlist size: the top frac of n points, at least one."""
    return max(1, int(np.ceil(frac * n)))


def shortlist(rho: np.ndarray, m: float) -> np.ndarray:
    """Points ranked within the budget. Tied points share one status."""
    return np.asarray(rho) <= m


# --- the noise null ---------------------------------------------------------

def null_log_ranks(X: np.ndarray, k: int, B: int = NULL_B,
                   seed: int = NULL_SEED, detector: str = "knn",
                   scales: tuple[int, ...] | None = None) -> np.ndarray:
    """(n, B) log2-ranks of every point in B half-reference replicates.

    Replicate b draws a random half H_b of the points and scores every point
    with the detector at each scale s, using s // 2 neighbours within H_b, the
    point itself excluded, so each neighbourhood holds the same share of the
    data as s does in the full set. Several scales are combined exactly as in
    detectors.combine(). Ranks are then taken over all n points. Neighbours
    come from neighbors.knn_to_reference(), the same metric and precision as
    the main comparison.

    detector="knn" with scales=None is version 1's null (one scale, k), bit
    for bit: the same random halves and the same k // 2-th distance.
    """
    from detectors import combine, scale_scores
    from neighbors import knn_to_reference

    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    scales = (k,) if scales is None else tuple(scales)
    half = tuple(max(1, s // 2) for s in scales)
    rng = np.random.default_rng(seed)
    L = np.empty((n, B))
    for b in range(B):
        ref = np.zeros(n, dtype=bool)
        ref[rng.choice(n, size=n // 2, replace=False)] = True
        dist, idx = knn_to_reference(X, ref, max(half), return_idx=True)
        s = combine(scale_scores(dist, idx, detector, half))
        L[:, b] = np.log2(desc_ranks(s))
    return L


def _pairwise_abs_diffs(L_chunk: np.ndarray) -> np.ndarray:
    B = L_chunk.shape[1]
    iu, ju = np.triu_indices(B, 1)
    return np.abs(L_chunk[:, iu] - L_chunk[:, ju])


def null_threshold(L: np.ndarray, alpha: float = ALPHA,
                   chunk: int = 256) -> np.ndarray:
    """Per-point threshold q(i): |lambda| > q(i) is significant at one-sided alpha.

    The null of a difference is symmetric, so P(D > q) = alpha is the
    (1 - 2 alpha) quantile of |D|. Computed over all B(B-1)/2 replicate pairs.
    """
    out = np.empty(L.shape[0])
    for s in range(0, L.shape[0], chunk):
        out[s:s + chunk] = np.quantile(_pairwise_abs_diffs(L[s:s + chunk]),
                                       1 - 2 * alpha, axis=1)
    return out


def null_pvalues(L: np.ndarray, lam: np.ndarray, chunk: int = 256) -> np.ndarray:
    """One-sided p-value in the direction each point moved: P(|D| >= |lambda|) / 2."""
    lam = np.abs(np.asarray(lam, float))
    out = np.empty(L.shape[0])
    for s in range(0, L.shape[0], chunk):
        D = _pairwise_abs_diffs(L[s:s + chunk])
        out[s:s + chunk] = 0.5 * (D >= lam[s:s + chunk, None] - 1e-12).mean(axis=1)
    return out


def get_null(name: str, k: int, B: int = NULL_B, null_seed: int = NULL_SEED,
             rebuild: bool = False, verbose: bool = True, detector: str = "knn",
             scales: tuple[int, ...] | None = None,
             **loader_kw) -> tuple[np.ndarray, dict]:
    """The noise null for one dataset, computed once and cached.

    Keyed on the data hash, like the neighbour graph, so a different loading
    option can never return a null computed from different data. null_seed
    draws the half-samples; loader keywords (subsample, seed, ...) pass
    straight to load_dataset. With detector="knn" and scales=None (version 1)
    the cache file keeps its version-1 name, so earlier results are reused;
    any other detector or scale set gets its own file. Scales that do not fit
    the dataset are dropped (detectors.usable_scales).
    """
    import time
    from config import CACHE
    from datasets import load_dataset
    from detectors import usable_scales

    X, _, meta = load_dataset(name, **loader_kw)
    if detector == "knn" and (scales is None or tuple(scales) == (k,)):
        scales = None
        path = CACHE / (f"null_{meta['name']}_{meta['data_hash']}_k{k}_B{B}_s{null_seed}_"
                        f"{NULL_VERSION}.npz")
    else:
        scales = usable_scales(tuple(scales or (k,)), meta["n"])
        tag = "-".join(str(s) for s in scales)
        path = CACHE / (f"null_{meta['name']}_{meta['data_hash']}_{detector}_K{tag}_B{B}_"
                        f"s{null_seed}_{NULL_VERSION}.npz")
    if path.exists() and not rebuild:
        return np.load(path)["L"], meta
    if verbose:
        print(f"  noise null for {name}: {B} half-reference replicates, {detector}, "
              f"scales {scales or (k,)} (n={meta['n']}, D={meta['D']}) ...", flush=True)
    t0 = time.time()
    L = null_log_ranks(np.asarray(X), k, B, null_seed, detector, scales)
    np.savez_compressed(path, L=L)
    path.with_suffix(".json").write_text(json.dumps(
        {**meta, "k": k, "detector": detector, "scales": list(scales or (k,)),
         "B": B, "null_seed": null_seed, "null_version": NULL_VERSION,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    if verbose:
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return L, meta


# --- calls and aggregates ---------------------------------------------------

def classify(rho_a: np.ndarray, rho_b: np.ndarray, m: float,
             p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Category code per point (see CATEGORY_NAMES)."""
    in_a, in_b = shortlist(rho_a, m), shortlist(rho_b, m)
    sig = np.asarray(p) < alpha
    cat = np.full(len(rho_a), BACKGROUND, dtype=np.int8)
    cat[in_a & in_b] = RETAINED
    cat[in_a & ~in_b & sig] = MASKED
    cat[in_a & ~in_b & ~sig] = EXIT_AMBIGUOUS
    cat[in_b & ~in_a & sig] = PHANTOM
    cat[in_b & ~in_a & ~sig] = ENTRY_AMBIGUOUS
    return cat


def adjusted_overlap(in_a: np.ndarray, in_b: np.ndarray) -> tuple[float, float]:
    """Chance-adjusted shortlist overlap and its permutation p-value.

    Under the permutation null (projected scores shuffled across points) the
    overlap is hypergeometric with mean |S_a| |S_b| / n. Returns
    (AO, P(overlap >= observed)).
    """
    n = len(in_a)
    sa, sb = int(in_a.sum()), int(in_b.sum())
    o = int((in_a & in_b).sum())
    e = sa * sb / n
    top = min(sa, sb) - e
    ao = (o - e) / top if top > 0 else float("nan")
    p = float(hypergeom.sf(o - 1, n, sa, sb))
    return float(ao), p


def weighted_rank_corr(rho_a: np.ndarray, rho_b: np.ndarray,
                       in_a: np.ndarray, in_b: np.ndarray) -> float:
    """Weighted Pearson correlation of rank-normalised scores.

    Weights after Schubert, Wojdanowski, Zimek & Kriegel (SDM 2012): the
    union of the two shortlists, K points, shares half the total weight and
    the remaining n - K points share the other half, so the bulk of inliers
    cannot dominate. They normalise scores rather than ranks; ranks are used
    here to keep v1 rank-based throughout (a v2 question).
    """
    n = len(rho_a)
    u_a = 1 - (np.asarray(rho_a) - 1) / (n - 1)
    u_b = 1 - (np.asarray(rho_b) - 1) / (n - 1)
    union = in_a | in_b
    K = int(union.sum())
    if K in (0, n):
        w = np.ones(n)
    else:
        w = np.where(union, 1 / (2 * K), 1 / (2 * (n - K)))
    w = w / w.sum()
    ma, mb = (w * u_a).sum(), (w * u_b).sum()
    cov = (w * (u_a - ma) * (u_b - mb)).sum()
    va, vb = (w * (u_a - ma) ** 2).sum(), (w * (u_b - mb) ** 2).sum()
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan")


def summarise(a: np.ndarray, b: np.ndarray, L: np.ndarray, m: float | None = None,
              y: np.ndarray | None = None, alpha: float = ALPHA) -> dict:
    """One dataset, one projection, one budget: every number worth reporting.

    a, b : detector scores in the original space and in the projection
    L    : the noise null, from null_log_ranks() or get_null()
    m    : shortlist budget; default the top BUDGET_FRAC of points
    y    : labels, optional; used only to say what the failures cost
    Per-point arrays (rho, lambda, p, category) come back under "points".
    """
    from scipy.stats import spearmanr

    n = len(a)
    m = budget(n) if m is None else m
    rho_a, rho_b = desc_ranks(a), desc_ranks(b)
    lam = displacement(rho_a, rho_b)
    p = null_pvalues(L, lam)
    cat = classify(rho_a, rho_b, m, p, alpha)
    in_a, in_b = shortlist(rho_a, m), shortlist(rho_b, m)
    count = {name: int((cat == code).sum()) for code, name in enumerate(CATEGORY_NAMES)}
    sa, sb = int(in_a.sum()), int(in_b.sum())
    ao, p_perm = adjusted_overlap(in_a, in_b)

    def med(mask):
        return float(np.median(lam[mask])) if mask.any() else float("nan")

    out = {
        "n": n, "m": float(m), "alpha": alpha,
        "size_a": sa, "size_b": sb,
        "overlap": int((in_a & in_b).sum()),
        "adjusted_overlap": ao, "p_perm": p_perm,
        "masking_rate": count["masked"] / sa if sa else float("nan"),
        "phantom_rate": count["phantom"] / sb if sb else float("nan"),
        "exit_rate": (count["masked"] + count["exit_ambiguous"]) / sa if sa else float("nan"),
        **{f"n_{k}": v for k, v in count.items()},
        "median_bits_masked": med(cat == MASKED),
        "median_bits_phantom": med(cat == PHANTOM),
        "median_bits_shortlist_a": med(in_a),
        "weighted_rank_corr": weighted_rank_corr(rho_a, rho_b, in_a, in_b),
        "spearman": float(spearmanr(a, b).statistic),
        "ties_a": n - n_distinct(a), "ties_b": n - n_distinct(b),
    }
    if y is not None:
        y = np.asarray(y).astype(bool)
        out.update({
            "anomalies": int(y.sum()),
            "anomalies_in_a": int((in_a & y).sum()),
            "anomalies_in_b": int((in_b & y).sum()),
            "masked_anomalies": int(((cat == MASKED) & y).sum()),
            "phantom_anomalies": int(((cat == PHANTOM) & y).sum()),
            "phantom_inliers": int(((cat == PHANTOM) & ~y).sum()),
            "precision_a": float((in_a & y).sum() / sa) if sa else float("nan"),
            "precision_b": float((in_b & y).sum() / sb) if sb else float("nan"),
        })
    out["points"] = {"rho_a": rho_a, "rho_b": rho_b, "lam": lam, "p": p, "cat": cat}
    return out
