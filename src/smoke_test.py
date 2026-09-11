"""Step 2 smoke test.

Checks that every library imports and can actually run, not just import.
Each check is isolated, so one failure does not hide the others.

Run:  python src/smoke_test.py
"""

import sys
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 12))

results = []


def check(name, fn, required=True):
    try:
        detail = fn()
        results.append((name, "OK", detail, required))
    except Exception as exc:
        results.append((name, "FAIL", f"{type(exc).__name__}: {exc}", required))


def _pyod():
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.iforest import IForest
    s = KNN(n_neighbors=10).fit(X).decision_scores_
    LOF(n_neighbors=10).fit(X)
    IForest(random_state=0).fit(X)
    return f"kNN, LOF, IForest all scored; shape {s.shape}"


def _skdim():
    import skdim
    g = float(skdim.id.MLE().fit(X).dimension_)
    pw = np.asarray(skdim.id.lPCA().fit_pw(X, n_neighbors=20).dimension_pw_)
    return f"global MLE id {g:.1f}; pointwise ids shape {pw.shape}"


def _opentsne():
    from openTSNE import TSNE
    Y = np.asarray(TSNE(n_components=2, random_state=0, verbose=False).fit(X))
    return f"embedding {Y.shape}"


def _umap():
    import umap
    return f"embedding {umap.UMAP(n_components=2, random_state=0).fit_transform(X).shape}"


def _pacmap():
    import pacmap
    return f"embedding {pacmap.PaCMAP(n_components=2, random_state=0).fit_transform(X).shape}"


def _zadu():
    import umap
    from zadu import zadu as zd
    Y = umap.UMAP(n_components=2, random_state=0).fit_transform(X)
    out = zd.ZADU([{"id": "tnc", "params": {"k": 20}}, {"id": "stress"}], X).measure(Y)
    tnc = out[0]
    return f"trustworthiness {tnc['trustworthiness']:.3f}, continuity {tnc['continuity']:.3f}"


def _trimap():
    import trimap
    return f"embedding {trimap.TRIMAP().fit_transform(X).shape}"


check("pyod", _pyod)
check("scikit-dimension", _skdim)
check("openTSNE", _opentsne)
check("umap-learn", _umap)
check("pacmap", _pacmap)
check("zadu", _zadu)
check("trimap", _trimap, required=False)

print()
width = max(len(n) for n, *_ in results)
failed_required = 0
for name, status, detail, required in results:
    if status == "FAIL" and not required:
        status = "SKIP"
        detail = "not installed yet, expected before week 6"
    if status == "FAIL":
        failed_required += 1
    print(f"  {name:<{width}}  {status:<4}  {detail}")

print()
if failed_required:
    print(f"{failed_required} required package(s) failed. Fix before continuing.")
    sys.exit(1)
print("All required packages working.")
