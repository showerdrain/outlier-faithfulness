"""Verification suite: checks whose right answer is known in advance.

Every result in this project is a comparison between outlier scores in the
original space and outlier scores in a projection. These checks test the
machinery of that comparison on cases where the answer is known, so that a
surprising result can be trusted to come from the data and not from the code.

    python src/validate.py           # everything; exits non-zero on failure
    python src/validate.py --quick   # skip the random-projection floor

Sections
  A  Reference implementations   kNN score vs pyod, LID vs scikit-dimension,
                                 compared by absolute value. Correlation is
                                 not enough: an early LID bug correlated at
                                 1.0000 while every value was 5% too large.
  B  Exact neighbour distances   the k-th neighbour distance, in the original
                                 space and in the projection, against an
                                 exhaustive pairwise computation that shares
                                 no code path with the pipeline.
  C  Identity projection         'projecting' data onto itself must leave every
                                 rank unchanged. Any movement is a difference
                                 in how the two sides are scored, not a
                                 property of projections.
  D  Known dimension             LID recovers the dimension of Gaussian data
                                 placed on a known linear subspace.
  E  Planted outliers            outliers placed in directions PCA keeps must
                                 survive projection; outliers placed in the
                                 directions it discards must be masked, with
                                 negative displacement. Validates the sign
                                 convention of the measure end to end.
  F  Measure arithmetic          small cases worked by hand: ranks with ties,
                                 displacement, shortlists, the chance-adjusted
                                 overlap.
  G  Random-projection floor     reported, not asserted. See the note there.
  H  Measure behaviour           properties the v1 measure must have whatever
                                 the data: zero under rotation, antisymmetry,
                                 invariance to monotone rescoring, a calibrated
                                 noise null, a chance level of zero, and the
                                 planted outliers masked and nothing else.
  I  Version 2 and new code      LOF and per-point trustworthiness/continuity
                                 against scikit-learn and ZADU by absolute
                                 value; the generalised null equal to v1's for
                                 one scale; the H properties again for the
                                 multi-scale kNN and LOF detectors; and the
                                 clumped-outlier fixture, the case version 2
                                 exists for.

The planted-outlier fixtures in E, H and I are small artificial datasets used
only as tests with a known answer. They play no part in any result, and they
are not the synthetic-data component that was dropped from the proposal.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import GaussianRandomProjection

from config import DEFAULT_K
from datasets import load_dataset
import measure as M
from neighbors import K_CACHE, get_knn, knn_score, knn_to_reference, lid_mle
from projections import knn_dist, project

K = DEFAULT_K
SHORTLIST = ["wbc", "ionosphere", "cardio", "musk",
             "satimage2", "satellite", "shuttle", "backdoor"]
_results: list[tuple[str, str, bool, str]] = []


def check(section: str, name: str, passed: bool, detail: str = "") -> None:
    _results.append((section, name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"   ({detail})" if detail else ""), flush=True)


def exact_kth(X: np.ndarray, k: int) -> np.ndarray:
    """k-th nearest-neighbour distance by exhaustive pairwise computation.

    Deliberately the most literal method available: every distance computed
    directly as sqrt(sum((x - y)^2)), the point itself removed by index. No
    trees, no dot-product shortcut, so it shares no code with the pipeline.
    """
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def planted_fixture(seed: int):
    """2,000 inliers in 10D; PCA keeps dimensions 0 and 1 (variance 36, 25).

    group 1: 20 outliers far out in the plane PCA keeps
    group 2: 20 outliers typical in that plane, far out in the 8 discarded
             directions (12 standard deviations, each in its own direction)
    """
    rng = np.random.default_rng(seed)
    D, n_in, m = 10, 2000, 20
    scale = np.array([6.0, 5.0] + [1.0] * (D - 2))
    X_in = rng.normal(size=(n_in, D)) * scale
    ang = rng.uniform(0, 2 * np.pi, m)
    X_vis = rng.normal(size=(m, D)) * np.r_[0.0, 0.0, np.ones(D - 2)]
    X_vis[:, 0], X_vis[:, 1] = 36 * np.cos(ang), 30 * np.sin(ang)
    X_hid = rng.normal(size=(m, D)) * scale
    u = rng.normal(size=(m, D - 2))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X_hid[:, 2:] += 12 * u
    X = np.vstack([X_in, X_vis, X_hid])
    group = np.r_[np.zeros(n_in), np.ones(m), 2 * np.ones(m)].astype(int)
    return X, group


# --------------------------------------------------------------------------

def section_a() -> None:
    print("\nA. Reference implementations, compared by absolute value")
    from pyod.models.knn import KNN
    import skdim

    X, _, _ = load_dataset("cardio")
    d, _, _ = get_knn("cardio", k=K, verbose=False)
    s = knn_score(d, K)
    ref = KNN(n_neighbors=K, method="largest").fit(X).decision_scores_
    rel = float(np.max(np.abs(s - ref) / ref))
    check("A", "kNN score agrees with pyod", rel < 1e-6,
          f"max relative error {rel:.1e}")

    lid = lid_mle(d, K)
    ref = np.asarray(skdim.id.MLE(K=K).fit(X).dimension_pw_)
    err = float(np.nanmax(np.abs(lid - ref)))
    check("A", "LID agrees with scikit-dimension", err < 1e-3,
          f"max absolute error {err:.1e}")


def section_b() -> None:
    print("\nB. Neighbour distances against exhaustive computation")
    for name in ("wbc", "ionosphere", "cardio"):
        X, _, _ = load_dataset(name)
        truth = exact_kth(np.asarray(X), K)
        d, _, _ = get_knn(name, k=K, verbose=False)
        rel = float(np.max(np.abs(knn_score(d, K) - truth) / truth))
        check("B", f"{name}: original-space k-th distance is exact", rel < 1e-9,
              f"max relative error {rel:.1e}")

        Y, _ = project(name, "pca", n_components=2, verbose=False)
        truth = exact_kth(Y, K)
        rel = float(np.max(np.abs(knn_score(knn_dist(Y, K), K) - truth) / truth))
        check("B", f"{name}: projected-space k-th distance is exact", rel < 1e-9,
              f"max relative error {rel:.1e}")

    for name in SHORTLIST:
        _, idx, meta = get_knn(name, k=min(K_CACHE, 100), verbose=False)
        n = idx.shape[0]
        hits = int((idx == np.arange(n)[:, None]).sum())
        check("B", f"{name}: no point is listed as its own neighbour", hits == 0,
              f"{hits} self-references in {n} rows")


def section_c() -> None:
    print("\nC. Identity projection: nothing may move")
    for name in ("wbc", "ionosphere", "cardio", "musk", "satimage2", "satellite"):
        X, y, _ = load_dataset(name)
        d_hd, _, _ = get_knn(name, k=K, verbose=False)
        s_hd = knn_score(d_hd, K)
        s_ld = knn_score(knn_dist(np.asarray(X), K), K)   # X 'projected' onto itself
        of = M.displacement(M.desc_ranks(s_hd), M.desc_ranks(s_ld))
        kk = int(y.sum())
        a = M.shortlist(M.desc_ranks(s_hd), kk)
        b = M.shortlist(M.desc_ranks(s_ld), kk)
        overlap = (a & b).sum() / a.sum()
        moved = int(np.count_nonzero(of))
        check("C", f"{name}: identity leaves every rank unchanged",
              moved == 0 and overlap == 1.0,
              f"{moved} of {len(of)} points moved, max |displacement| "
              f"{np.abs(of).max():.1e}, shortlist overlap {100 * overlap:.1f}%")


def section_d() -> None:
    print("\nD. LID on data of known dimension (Gaussian in 30D, n = 5,000)")
    rng = np.random.default_rng(0)
    for d_true in (2, 5, 10):
        Z = rng.normal(size=(5000, d_true))
        Q, _ = np.linalg.qr(rng.normal(size=(30, d_true)))
        dist = knn_dist(Z @ Q.T, K)
        med = float(np.median(lid_mle(dist, K)))
        err = abs(med - d_true) / d_true
        check("D", f"recovers dimension {d_true}", err < 0.10,
              f"median LID {med:.2f}, error {100 * err:.1f}%")


def section_e() -> None:
    print("\nE. Planted outliers with a known fate under PCA (five seeds)")
    rows = []
    for seed in range(5):
        X, g = planted_fixture(seed)
        y = (g > 0).astype(int)
        vis, hid = np.flatnonzero(g == 1), np.flatnonzero(g == 2)
        d_hd = knn_dist(X, K)
        s_hd = knn_score(d_hd, K)
        Y = PCA(n_components=2, svd_solver="full").fit_transform(X)
        s_ld = knn_score(knn_dist(Y, K), K)
        of = M.displacement(M.desc_ranks(s_hd), M.desc_ranks(s_ld))
        top_hd = np.flatnonzero(M.shortlist(M.desc_ranks(s_hd), 40))
        top_ld = np.flatnonzero(M.shortlist(M.desc_ranks(s_ld), 40))
        lid = lid_mle(d_hd, K)
        rows.append(dict(
            seed=seed, auc=roc_auc_score(y, s_hd),
            vis_hd=int(np.isin(vis, top_hd).sum()), vis_ld=int(np.isin(vis, top_ld).sum()),
            hid_hd=int(np.isin(hid, top_hd).sum()), hid_ld=int(np.isin(hid, top_ld).sum()),
            of_vis=float(np.median(of[vis])), of_hid=float(np.median(of[hid])),
            lid_in=float(np.median(lid[g == 0])), lid_vis=float(np.median(lid[vis])),
            lid_hid=float(np.median(lid[hid]))))

    r = rows
    check("E", "every planted outlier is found in the original space",
          all(x["auc"] >= 0.99 for x in r),
          f"AUC {min(x['auc'] for x in r):.4f} or better")
    check("E", "outliers in the kept plane survive projection",
          all(x["vis_ld"] == 20 and x["of_vis"] >= -0.5 for x in r),
          f"{min(x['vis_ld'] for x in r)}/20 or more stay in the top 40, "
          f"median displacement {min(x['of_vis'] for x in r):+.2f} bits at worst")
    check("E", "outliers in the discarded directions are masked",
          all(x["hid_hd"] >= 15 and x["hid_ld"] <= 2 for x in r),
          f"{min(x['hid_hd'] for x in r)}-{max(x['hid_hd'] for x in r)}/20 in the original "
          f"top 40, {max(x['hid_ld'] for x in r)}/20 at most after projection")
    check("E", "masked outliers carry negative displacement (sign convention)",
          all(x["of_hid"] <= -2.0 for x in r),
          f"median displacement {max(x['of_hid'] for x in r):+.2f} bits at best")
    print(f"     (reported) median LID: inliers {np.mean([x['lid_in'] for x in r]):.1f}, "
          f"outliers in kept plane {np.mean([x['lid_vis'] for x in r]):.1f}, "
          f"outliers in discarded directions {np.mean([x['lid_hid'] for x in r]):.1f}")


def section_f() -> None:
    print("\nF. Measure arithmetic, worked by hand")
    rho = M.desc_ranks(np.array([5.0, 3.0, 3.0, 3.0, 1.0]))
    check("F", "tied scores share the average of their positions",
          rho.tolist() == [1.0, 3.0, 3.0, 3.0, 5.0], f"got {rho.tolist()}")

    s = np.array([5.0, 3.0, 3.0, 3.0, 1.0])
    perm = np.random.default_rng(0).permutation(5)
    check("F", "ranks do not depend on input order",
          np.array_equal(M.desc_ranks(s[perm]), M.desc_ranks(s)[perm]))

    near = M.desc_ranks(np.array([2.0, 2.0 * (1 + 1e-13), 1.0]))
    apart = M.desc_ranks(np.array([2.0, 2.0 * (1 + 1e-6), 1.0]))
    check("F", "scores within the tolerance tie, scores beyond it do not",
          near.tolist() == [1.5, 1.5, 3.0] and apart.tolist() == [2.0, 1.0, 3.0],
          f"1e-13 apart {near.tolist()}, 1e-6 apart {apart.tolist()}")

    lam = M.displacement(M.desc_ranks(np.array([1.0, 2.0, 3.0])),
                         M.desc_ranks(np.array([3.0, 2.0, 1.0])))
    check("F", "full reversal gives [+log2 3, 0, -log2 3] bits",
          np.allclose(lam, [np.log2(3), 0, -np.log2(3)]), f"got {np.round(lam, 4).tolist()}")

    rho = M.desc_ranks(s)
    in2, in3 = M.shortlist(rho, 2), M.shortlist(rho, 3)
    check("F", "a tied group straddling the budget is wholly in or wholly out",
          in2.tolist() == [True, False, False, False, False]
          and in3.tolist() == [True, True, True, True, False],
          f"m=2 keeps {int(in2.sum())}, m=3 keeps {int(in3.sum())}")

    n = 100
    in_a = np.zeros(n, bool); in_a[:10] = True
    in_b = np.zeros(n, bool); in_b[[0] + list(range(50, 59))] = True
    ao_chance, _ = M.adjusted_overlap(in_a, in_b)
    ao_same, p_same = M.adjusted_overlap(in_a, in_a)
    check("F", "adjusted overlap is 1 for identical shortlists, 0 at chance",
          np.isclose(ao_same, 1.0) and np.isclose(ao_chance, 0.0) and p_same < 1e-10,
          f"identical {ao_same:.3f} (p {p_same:.1e}), overlap 1 of 10 at n=100 {ao_chance:.3f}")


def section_g() -> None:
    print("\nG. Random-projection floor (reported, not asserted)")
    print("   A random 2D projection is the uninformed baseline. PCA is NOT")
    print("   required to beat it: an outlier whose outlyingness lies in")
    print("   low-variance directions is discarded by PCA by construction")
    print("   (section E), while a random projection keeps a share of every")
    print("   direction. PCA losing to random on a dataset is a finding.\n")
    print(f"   {'dataset':<12}{'PCA rank corr':>14}{'random (5 seeds)':>20}"
          f"{'PCA AUC kept':>14}{'random AUC kept':>18}")
    for name in SHORTLIST:
        X, y, _ = load_dataset(name)
        d_hd, _, _ = get_knn(name, k=K, verbose=False)
        s_hd = knn_score(d_hd, K)
        auc_hd = roc_auc_score(y, s_hd)
        Y, _ = project(name, "pca", n_components=2, verbose=False)
        s_p = knn_score(knn_dist(Y, K), K)
        rho_p = spearmanr(s_hd, s_p).statistic
        keep_p = roc_auc_score(y, s_p) / auc_hd
        rho_r, keep_r = [], []
        for seed in range(5):
            Yr = GaussianRandomProjection(n_components=2, random_state=seed).fit_transform(X)
            s_r = knn_score(knn_dist(Yr, K), K)
            rho_r.append(spearmanr(s_hd, s_r).statistic)
            keep_r.append(roc_auc_score(y, s_r) / auc_hd)
        print(f"   {name:<12}{rho_p:>14.3f}{np.mean(rho_r):>13.3f} ± {np.std(rho_r):.3f}"
              f"{100 * keep_p:>13.0f}%{100 * np.mean(keep_r):>11.0f}% ± {100 * np.std(keep_r):.0f}")



def section_h() -> None:
    print("\nH. Measure behaviour (development datasets and the planted fixture only)")

    for name in ("wbc", "ionosphere"):
        X, _, _ = load_dataset(name)
        X = np.asarray(X)
        a = knn_score(knn_dist(X, K), K)
        Y = PCA(n_components=X.shape[1], svd_solver="full").fit_transform(X)
        lam = M.displacement(M.desc_ranks(a), M.desc_ranks(knn_score(knn_dist(Y, K), K)))
        check("H", f"{name}: PCA to full dimension (a rotation) moves nothing",
              np.count_nonzero(lam) == 0, f"max |lambda| {np.abs(lam).max():.1e} bits")

    X, _, _ = load_dataset("fault")
    X = np.asarray(X)
    a = knn_score(knn_dist(X, K), K)
    b = knn_score(knn_dist(PCA(n_components=2, svd_solver="full").fit_transform(X), K), K)
    L = M.null_log_ranks(X, K, 100, 0)
    ab = M.summarise(a, b, L)
    ba = M.summarise(b, a, L)
    check("H", "swapping the two spaces negates lambda and swaps masked/phantom",
          np.allclose(ab["points"]["lam"], -ba["points"]["lam"])
          and np.array_equal(ab["points"]["cat"] == M.MASKED, ba["points"]["cat"] == M.PHANTOM),
          f"{ab['n_masked']} masked / {ab['n_phantom']} phantom, swapped "
          f"{ba['n_masked']} / {ba['n_phantom']}")
    mono = M.summarise(a, b ** 3 + 2.0, L)
    check("H", "a monotone rescoring of either space changes nothing",
          np.array_equal(mono["points"]["cat"], ab["points"]["cat"]))

    rng = np.random.default_rng(0)
    rho_a = M.desc_ranks(a)
    in_a = M.shortlist(rho_a, M.budget(len(a)))
    aos, ps = [], []
    for _ in range(200):
        ao, p = M.adjusted_overlap(in_a, M.shortlist(M.desc_ranks(rng.permutation(b)),
                                                      M.budget(len(a))))
        aos.append(ao)
        ps.append(p)
    check("H", "projected scores shuffled across points: adjusted overlap ~ 0",
          abs(np.mean(aos)) < 0.03 and np.mean(np.array(ps) < 0.05) <= 0.08,
          f"mean {np.mean(aos):+.3f} over 200 shuffles, "
          f"{100 * np.mean(np.array(ps) < 0.05):.1f}% with p_perm < 0.05")

    # Size of the noise null: compare two fresh re-measurements of the original
    # space with each other. That is the null hypothesis itself, so the share of
    # boundary crossings called significant must stay near alpha.
    for name in ("wbc", "ionosphere", "fault"):
        X, _, _ = load_dataset(name)
        X = np.asarray(X)
        n = len(X)
        L = M.null_log_ranks(X, K, 100, 0)
        rng = np.random.default_rng(12345)
        mr, pr = [], []
        for _ in range(20):
            reps = []
            for _ in range(2):
                ref = np.zeros(n, bool)
                ref[rng.choice(n, n // 2, replace=False)] = True
                reps.append(knn_to_reference(X, ref, K // 2)[:, -1])
            r = M.summarise(reps[0], reps[1], L)
            mr.append(r["masking_rate"])
            pr.append(r["phantom_rate"])
        check("H", f"{name}: under the null itself, calls stay near alpha = {M.ALPHA}",
              np.mean(mr) <= 2 * M.ALPHA and np.mean(pr) <= 2 * M.ALPHA,
              f"mean masking rate {np.mean(mr):.3f}, phantom rate {np.mean(pr):.3f}")

    hid_masked, vis_masked, vis_kept = [], [], []
    for seed in range(5):
        X, g = planted_fixture(seed)
        vis, hid = np.flatnonzero(g == 1), np.flatnonzero(g == 2)
        a = knn_score(knn_dist(X, K), K)
        b = knn_score(knn_dist(PCA(n_components=2, svd_solver="full").fit_transform(X), K), K)
        cat = M.summarise(a, b, M.null_log_ranks(X, K, 100, 0), m=40)["points"]["cat"]
        hid_masked.append(int((cat[hid] == M.MASKED).sum()))
        vis_masked.append(int((cat[vis] == M.MASKED).sum()))
        vis_kept.append(int((cat[vis] == M.RETAINED).sum()))
    check("H", "planted fixture: hidden outliers are masked, visible ones retained",
          min(hid_masked) >= 15 and max(vis_masked) == 0 and min(vis_kept) == 20,
          f"hidden masked {min(hid_masked)}-{max(hid_masked)}/20, visible masked "
          f"{max(vis_masked)} at most, visible retained {min(vis_kept)}/20 at least")


def clump_fixture(seed: int):
    """2,000 inliers in 10D and 40 outliers, each alone in its own direction.

    Three 2D pictures share the inliers (their first two coordinates) and
    place the 40 outliers differently:
      clumped    one tight group far from the bulk, what UMAP and PaCMAP do
      scattered  spread on a wide circle, each still alone
      absorbed   inside the bulk, a genuine loss
    A faithful reading calls the first two retained and the third masked.
    """
    rng = np.random.default_rng(seed)
    n_in, m, D = 2000, 40, 10
    X_in = rng.normal(size=(n_in, D))
    u = rng.normal(size=(m, D))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = np.vstack([X_in, 12 * u + rng.normal(scale=0.5, size=(m, D))])
    Y_in = X_in[:, :2]
    ang = rng.uniform(0, 2 * np.pi, m)
    pictures = {
        "clumped": np.vstack([Y_in, np.array([30.0, 0.0]) + rng.normal(scale=0.05, size=(m, 2))]),
        "scattered": np.vstack([Y_in, 30 * np.column_stack([np.cos(ang), np.sin(ang)])]),
        "absorbed": np.vstack([Y_in, rng.normal(size=(m, 2))]),
    }
    return X, pictures, np.arange(n_in, n_in + m)


def _replicate_scores(X: np.ndarray, rng, detector: str, scales: tuple) -> np.ndarray:
    """One fresh half-reference re-measurement, as the noise null makes them."""
    from detectors import combine, scale_scores
    n = len(X)
    ref = np.zeros(n, bool)
    ref[rng.choice(n, n // 2, replace=False)] = True
    half = tuple(max(1, s // 2) for s in scales)
    dist, idx = knn_to_reference(X, ref, max(half), return_idx=True)
    return combine(scale_scores(dist, idx, detector, half))


def section_i() -> None:
    print("\nI. Version 2 (multi-scale) and the week-5 code")
    import detectors as T
    import quality as Q
    from sklearn.neighbors import LocalOutlierFactor
    from zadu.measures import trustworthiness_continuity as tnc

    X, _, _ = load_dataset("cardio")
    X = np.asarray(X)
    errs = []
    for k in (20, 50):
        d, i, _ = get_knn("cardio", k=k, verbose=False)
        ref = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
        errs.append(float(np.max(np.abs(T.lof_from_graph(d, i, k) - ref) / ref)))
    check("I", "LOF from the neighbour graph agrees with scikit-learn", max(errs) < 1e-9,
          f"max relative error {max(errs):.1e} at k = 20 and 50")

    X, _, _ = load_dataset("ionosphere")
    X = np.asarray(X)
    Y = PCA(n_components=2, svd_solver="full").fit_transform(X)
    glob, loc = tnc.measure(X, Y, k=K, return_local=True)
    t, c = Q.local_trust_cont(X, Y, K)
    diff = max(np.abs(t - loc["local_trustworthiness"]).max(),
               np.abs(c - loc["local_continuity"]).max())
    check("I", "per-point trustworthiness and continuity agree with ZADU", diff < 1e-12,
          f"max absolute difference {diff:.1e}; T {t.mean():.4f}, C {c.mean():.4f}")

    X, _, _ = load_dataset("wbc")
    X = np.asarray(X)
    rng = np.random.default_rng(0)
    old = np.empty((len(X), 10))
    for b in range(10):
        ref = np.zeros(len(X), bool)
        ref[rng.choice(len(X), len(X) // 2, replace=False)] = True
        old[:, b] = np.log2(M.desc_ranks(knn_to_reference(X, ref, K // 2)[:, -1]))
    check("I", "the generalised null is version 1's null for one scale",
          np.array_equal(M.null_log_ranks(X, K, 10, 0), old))

    # LOF only on continuous data: on lattice data (wbc, shuttle) tied distances
    # run past the stored neighbours and LOF depends on their arbitrary order.
    for det, names in (("knn", ("wbc", "ionosphere", "fault")), ("lof", ("ionosphere", "fault"))):
        moved = []
        for name in names:
            X, _, _ = load_dataset(name)
            a = T.score_original(name, det, T.SCALES_V2)
            b = T.score_space(np.asarray(X), det, T.SCALES_V2)
            rot = T.score_space(PCA(n_components=X.shape[1], svd_solver="full")
                                .fit_transform(np.asarray(X)), det, T.SCALES_V2)
            moved.append(np.count_nonzero(M.displacement(M.desc_ranks(a), M.desc_ranks(b)))
                         + np.count_nonzero(M.displacement(M.desc_ranks(a), M.desc_ranks(rot))))
        check("I", f"v2 {det}: identity and rotation move nothing ({', '.join(names)})",
              sum(moved) == 0, f"{sum(moved)} points moved")

    X, _, _ = load_dataset("fault")
    X = np.asarray(X)
    a = T.score_space(X, "knn", T.SCALES_V2)
    b = T.score_space(PCA(n_components=2, svd_solver="full").fit_transform(X), "knn", T.SCALES_V2)
    L = M.null_log_ranks(X, K, 100, 0, "knn", T.SCALES_V2)
    ab, ba = M.summarise(a, b, L), M.summarise(b, a, L)
    mono = M.summarise(a, b ** 3 + 2.0, L)
    check("I", "v2: antisymmetry and invariance to monotone rescoring",
          np.allclose(ab["points"]["lam"], -ba["points"]["lam"])
          and np.array_equal(ab["points"]["cat"] == M.MASKED, ba["points"]["cat"] == M.PHANTOM)
          and np.array_equal(mono["points"]["cat"], ab["points"]["cat"]),
          f"{ab['n_masked']} masked / {ab['n_phantom']} phantom, swapped "
          f"{ba['n_masked']} / {ba['n_phantom']}")

    for det, names in (("knn", ("wbc", "ionosphere", "fault")), ("lof", ("ionosphere", "fault"))):
        rates = []
        for name in names:
            X, _, _ = load_dataset(name)
            X = np.asarray(X)
            L = M.null_log_ranks(X, K, 100, 0, det, T.SCALES_V2)
            rng = np.random.default_rng(12345)
            for _ in range(10):
                r = M.summarise(_replicate_scores(X, rng, det, T.SCALES_V2),
                                _replicate_scores(X, rng, det, T.SCALES_V2), L)
                rates.append((r["masking_rate"], r["phantom_rate"]))
        mr, pr = np.mean([x[0] for x in rates]), np.mean([x[1] for x in rates])
        check("I", f"v2 {det}: under the null itself, calls stay near alpha = {M.ALPHA}",
              mr <= 2 * M.ALPHA and pr <= 2 * M.ALPHA,
              f"mean masking rate {mr:.3f}, phantom rate {pr:.3f} ({', '.join(names)})")

    for det in ("knn", "lof"):
        hid_masked, vis_masked, vis_kept = [], [], []
        for seed in range(5):
            X, g = planted_fixture(seed)
            vis, hid = np.flatnonzero(g == 1), np.flatnonzero(g == 2)
            a = T.score_space(X, det, T.SCALES_V2)
            b = T.score_space(PCA(n_components=2, svd_solver="full").fit_transform(X),
                              det, T.SCALES_V2)
            cat = M.summarise(a, b, M.null_log_ranks(X, K, 100, 0, det, T.SCALES_V2),
                              m=40)["points"]["cat"]
            hid_masked.append(int((cat[hid] == M.MASKED).sum()))
            vis_masked.append(int((cat[vis] == M.MASKED).sum()))
            vis_kept.append(int((cat[vis] == M.RETAINED).sum()))
        check("I", f"v2 {det}: planted fixture, hidden outliers masked, visible retained",
              min(hid_masked) >= 15 and max(vis_masked) == 0 and min(vis_kept) == 20,
              f"hidden masked {min(hid_masked)}-{max(hid_masked)}/20, visible masked "
              f"{max(vis_masked)} at most, visible retained {min(vis_kept)}/20 at least")

    res = {("v1", p): [] for p in ("clumped", "scattered", "absorbed")}
    res.update({("v2", p): [] for p in ("clumped", "scattered", "absorbed")})
    for seed in range(3):
        X, pictures, out = clump_fixture(seed)
        for label, scales in (("v1", T.SCALES_V1), ("v2", T.SCALES_V2)):
            a = T.score_space(X, "knn", scales)
            L = M.null_log_ranks(X, K, 50, 0, "knn", scales)
            for p, Y in pictures.items():
                cat = M.summarise(a, T.score_space(Y, "knn", scales), L, m=40)["points"]["cat"][out]
                res[(label, p)].append((int((cat == M.MASKED).sum()), int((cat == M.RETAINED).sum())))
    worst = lambda key, j, f: f(x[j] for x in res[key])
    check("I", "clumped fixture: v2 keeps a far-away clump of outliers that v1 calls hidden",
          worst(("v2", "clumped"), 1, min) == 40 and worst(("v1", "clumped"), 0, min) >= 35,
          f"retained by v2 {worst(('v2', 'clumped'), 1, min)}/40 at least, masked by v1 "
          f"{worst(('v1', 'clumped'), 0, min)}/40 at least, over three seeds")
    check("I", "clumped fixture: v2 still keeps scattered outliers and still hides absorbed ones",
          worst(("v2", "scattered"), 1, min) == 40 and worst(("v2", "absorbed"), 0, min) >= 35,
          f"scattered retained {worst(('v2', 'scattered'), 1, min)}/40 at least, absorbed "
          f"masked {worst(('v2', 'absorbed'), 0, min)}/40 at least")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    if not quick:
        section_g()
    section_h()
    section_i()

    failed = [r for r in _results if not r[2]]
    print(f"\n{len(_results) - len(failed)} of {len(_results)} checks passed.")
    if failed:
        print("Failed:")
        for sec, name, _, detail in failed:
            print(f"  {sec}  {name}   ({detail})")
        sys.exit(1)
