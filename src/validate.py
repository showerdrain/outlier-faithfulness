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
  F  Measure arithmetic          small cases worked by hand.
  G  Random-projection floor     reported, not asserted. See the note there.

The planted-outlier fixture in E is a small artificial dataset used only as a
test with a known answer. It plays no part in any result, and it is not the
synthetic-data component that was dropped from the proposal.
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
from measure import signed_displacement, top_k
from neighbors import K_CACHE, get_knn, knn_score, lid_mle
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
        of = signed_displacement(s_hd, s_ld)
        kk = int(y.sum())
        a, _ = top_k(s_hd, kk)
        b, _ = top_k(s_ld, kk)
        overlap = len(set(a.tolist()) & set(b.tolist())) / kk
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
        of = signed_displacement(s_hd, s_ld)
        top_hd, _ = top_k(s_hd, 40)
        top_ld, _ = top_k(s_ld, 40)
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
          all(x["vis_ld"] == 20 and x["of_vis"] >= -0.01 for x in r),
          f"{min(x['vis_ld'] for x in r)}/20 or more stay in the top 40, "
          f"median displacement {min(x['of_vis'] for x in r):+.3f} at worst")
    check("E", "outliers in the discarded directions are masked",
          all(x["hid_hd"] >= 15 and x["hid_ld"] <= 2 for x in r),
          f"{min(x['hid_hd'] for x in r)}-{max(x['hid_hd'] for x in r)}/20 in the original "
          f"top 40, {max(x['hid_ld'] for x in r)}/20 at most after projection")
    check("E", "masked outliers carry negative displacement (sign convention)",
          all(x["of_hid"] <= -0.25 for x in r),
          f"median displacement {max(x['of_hid'] for x in r):+.3f} at best")
    print(f"     (reported) median LID: inliers {np.mean([x['lid_in'] for x in r]):.1f}, "
          f"outliers in kept plane {np.mean([x['lid_vis'] for x in r]):.1f}, "
          f"outliers in discarded directions {np.mean([x['lid_hid'] for x in r]):.1f}")


def section_f() -> None:
    print("\nF. Measure arithmetic, worked by hand")
    of = signed_displacement(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    check("F", "full reversal gives [+2/3, 0, -2/3]",
          np.allclose(of, [2 / 3, 0, -2 / 3]), f"got {np.round(of, 4).tolist()}")
    s = np.array([5.0, 3.0, 3.0, 3.0, 1.0])
    runs = {tuple(top_k(s, 2)[0].tolist()) for _ in range(20)}
    order, ties = top_k(s, 2)
    check("F", "top-k is deterministic under ties and reports them",
          len(runs) == 1 and order.tolist() == [0, 1] and ties == 3,
          f"order {order.tolist()}, boundary ties {ties}")


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

    failed = [r for r in _results if not r[2]]
    print(f"\n{len(_results) - len(failed)} of {len(_results)} checks passed.")
    if failed:
        print("Failed:")
        for sec, name, _, detail in failed:
            print(f"  {sec}  {name}   ({detail})")
        sys.exit(1)
