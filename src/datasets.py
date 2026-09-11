"""Dataset loading with one uniform interface.

Every dataset, whatever its source, comes back as the same three things:

    X     (n, D) float64 feature matrix
    y     (n,)   int8 labels, 1 = anomaly, 0 = inlier
    meta  dict   provenance: where it came from and what was done to it

Files download once into data/raw/ and are reused afterwards.

Run directly to build the summary table:
    python src/datasets.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, asdict

import numpy as np

from config import DATA_RAW, RESULTS, DEFAULT_SEED

ADBENCH = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/"
)


@dataclass(frozen=True)
class Source:
    file: str           # filename in the ADBench Classical folder
    domain: str         # "security", "security-adjacent", or "general"
    note: str = ""


# Curated shortlist. Not every ADBench set: these were chosen to span size,
# dimensionality and anomaly rate, with the security-domain ones first.
REGISTRY: dict[str, Source] = {
    # --- security -------------------------------------------------------
    "backdoor":   Source("3_backdoor",    "security",
                         "UNSW-NB15 backdoor attacks; the main security candidate"),
    "spambase":   Source("35_SpamBase",   "security",
                         "spam; 40% positives, closer to classification than outlier detection"),
    "http":       Source("16_http",       "security",
                         "KDDCup99 HTTP; only 3 features, see note in step 3"),
    "smtp":       Source("34_smtp",       "security",
                         "KDDCup99 SMTP; only 3 features, see note in step 3"),
    # --- security-adjacent ----------------------------------------------
    "fraud":      Source("13_fraud",      "security-adjacent",
                         "credit card fraud; PCA-transformed features"),
    # --- general-purpose contrast ---------------------------------------
    "shuttle":    Source("32_shuttle",    "general", "large n, low D"),
    "satellite":  Source("30_satellite",  "general", "high anomaly rate"),
    "satimage2":  Source("31_satimage-2", "general", "same source as satellite, low rate"),
    "musk":       Source("25_musk",       "general", "high D relative to n"),
    "cardio":     Source("6_cardio",      "general", "small, moderate rate"),
    "ionosphere": Source("18_Ionosphere", "general", "small n, high rate"),
    "wbc":        Source("42_WBC",        "general", "smallest; useful for fast smoke runs"),
}


def _fetch(src: Source) -> np.lib.npyio.NpzFile:
    path = DATA_RAW / f"{src.file}.npz"
    if not path.exists():
        url = ADBENCH + f"{src.file}.npz"
        print(f"  downloading {src.file} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    return np.load(path)


def load_dataset(
    name: str,
    standardize: bool = True,
    subsample: int | None = None,
    drop_duplicates: bool = True,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one dataset.

    standardize     : zero mean, unit variance per feature. On by default
        because every downstream step (kNN, LOF, LID, all projections) uses
        Euclidean distance, which is meaningless across features on different
        scales. Constant features are left alone rather than divided by zero.
    subsample       : cap n, sampling stratified so the anomaly rate is
        preserved. Recorded in meta so a shrunken dataset is never silent.
    drop_duplicates : remove exact duplicate rows, keeping the first. On by
        default because duplicates put zero distances into the neighbour
        graph, which makes LOF degenerate and breaks the MLE local intrinsic
        dimensionality estimator outright (it takes logs of distance ratios).
        Campos et al. (2016) ship with-and-without variants for this reason.

    meta['data_hash'] identifies the exact arrays returned, after every
    transformation above. Step 4 keys the neighbour cache on it, so changing
    any option here produces a new cache entry instead of a silent stale hit.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    src = REGISTRY[name]

    d = _fetch(src)
    X = np.asarray(d["X"], dtype=np.float64)
    y = np.asarray(d["y"]).astype(np.int8).ravel()
    n_full = X.shape[0]

    n_dup = n_full - len(np.unique(X, axis=0))

    n_removed = 0
    n_conflicting = 0
    if drop_duplicates and n_dup > 0:
        _, first_idx, inverse = np.unique(X, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        for g in np.flatnonzero(np.bincount(inverse) > 1):
            if len(np.unique(y[inverse == g])) > 1:
                n_conflicting += 1
        keep = np.sort(first_idx)
        n_removed = len(X) - len(keep)
        X, y = X[keep], y[keep]
        if n_removed > 0.01 * n_full:
            print(f"  note: {name} lost {n_removed} duplicate rows "
                  f"({100 * n_removed / n_full:.1f}% of the original)")

    n_after_dedup = X.shape[0]

    if subsample is not None and subsample < n_after_dedup:
        rng = np.random.default_rng(seed)
        idx_out = np.flatnonzero(y == 1)
        idx_in = np.flatnonzero(y == 0)
        keep_out = max(1, round(subsample * len(idx_out) / n_after_dedup))
        keep_in = subsample - keep_out
        sel = np.concatenate([
            rng.choice(idx_out, size=min(keep_out, len(idx_out)), replace=False),
            rng.choice(idx_in, size=min(keep_in, len(idx_in)), replace=False),
        ])
        sel.sort()
        X, y = X[sel], y[sel]

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0          # leave constant features untouched
        X = (X - mu) / sd

    data_hash = hashlib.sha256(
        np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()
    ).hexdigest()[:16]

    meta = {
        "name": name,
        "data_hash": data_hash,
        "source_file": src.file,
        "domain": src.domain,
        "note": src.note,
        "origin": "ADBench (Han et al. 2022), Classical collection",
        "n": int(X.shape[0]),
        "n_full": int(n_full),
        "D": int(X.shape[1]),
        "n_anomalies": int(y.sum()),
        "anomaly_rate": float(y.mean()),
        "duplicate_rows_in_full": int(n_dup),
        "duplicates_removed": int(n_removed),
        "duplicate_groups_with_conflicting_labels": int(n_conflicting),
        "standardized": bool(standardize),
        "subsampled_to": subsample if subsample is not None and subsample < n_after_dedup else None,
        "seed": seed if subsample is not None else None,
    }
    return X, y, meta


def describe_all(subsample: int | None = None) -> list[dict]:
    """Load every registered dataset and return the metadata table."""
    out = []
    for name in REGISTRY:
        _, _, meta = load_dataset(name, subsample=subsample)
        out.append(meta)
    return out


if __name__ == "__main__":
    rows = describe_all()
    path = RESULTS / "dataset_table.json"
    path.write_text(json.dumps(rows, indent=2))

    hdr = (f"{'dataset':<12}{'domain':<19}{'n':>8}{'D':>6}{'anom%':>8}"
           f"{'dups cut':>10}  {'data_hash':<16}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["domain"], -r["n"])):
        print(f"{r['name']:<12}{r['domain']:<19}{r['n']:>8}{r['D']:>6}"
              f"{100 * r['anomaly_rate']:>8.2f}{r['duplicates_removed']:>10}  "
              f"{r['data_hash']:<16}")
    print(f"\nWrote {path}")
