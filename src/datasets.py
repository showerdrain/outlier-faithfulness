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
    family: str = ""    # underlying source; versions of the same data share one


# Every dataset in ADBench's Classical collection (47). Which ones the thesis
# uses is decided by src/screen.py against stated criteria, not by this list.
#
# FAMILIES. Several ADBench files are versions of one underlying dataset with
# a different class taken as the anomaly. They are not independent, so they
# may not appear on both sides of the development/benchmark split, and count
# once in any cross-dataset statistic. Links are kept conservatively: a
# family tag only ever prevents a pairing, so an unproven link costs little.
#   confirmed from the data  satimage2 and wbc are exact row subsets of
#                            satellite and breastw; landsat has a correlation
#                            matrix identical to satellite's (rescaled copy)
#   documented source only   cardio/cardiotocography (UCI CTG), thyroid/
#                            annthyroid (UCI thyroid), wdbc/wpbc (Wisconsin);
#                            columns are reordered or rescaled between files,
#                            so a quick row comparison could not confirm them
REGISTRY: dict[str, Source] = {
    # --- security -------------------------------------------------------
    "backdoor":   Source("3_backdoor",    "security",
                         "UNSW-NB15 backdoor attacks; the main security candidate"),
    "spambase":   Source("35_SpamBase",   "security",
                         "spam; 40% positives, closer to classification than outlier detection"),
    "http":       Source("16_http",       "security",
                         "KDDCup99 HTTP; only 3 features, 61% duplicate rows", "kddcup99"),
    "smtp":       Source("34_smtp",       "security",
                         "KDDCup99 SMTP; only 3 features, 25% duplicate rows", "kddcup99"),
    # --- security-adjacent ----------------------------------------------
    "fraud":      Source("13_fraud",      "security-adjacent",
                         "credit card fraud; PCA-transformed features"),
    "internetads": Source("17_InternetAds", "security-adjacent",
                          "advert images on web pages; 1,555 mostly binary features"),
    # --- general --------------------------------------------------------
    "shuttle":    Source("32_shuttle",    "general", "large n, low D"),
    "satellite":  Source("30_satellite",  "general", "high anomaly rate", "statlog-landsat"),
    "satimage2":  Source("31_satimage-2", "general",
                         "exact row subset of satellite, low rate", "statlog-landsat"),
    "landsat":    Source("19_landsat",    "general",
                         "satellite rescaled: identical correlation matrix", "statlog-landsat"),
    "musk":       Source("25_musk",       "general", "high D relative to n"),
    "cardio":     Source("6_cardio",      "general", "small, moderate rate", "ctg"),
    "cardiotocography": Source("7_Cardiotocography", "general",
                               "documented as the same UCI CTG source as cardio", "ctg"),
    "ionosphere": Source("18_Ionosphere", "general", "small n, high rate"),
    "wbc":        Source("42_WBC",        "general",
                         "smallest; useful for fast smoke runs", "wisconsin-original"),
    "breastw":    Source("4_breastw",     "general",
                         "WBC's rows are an exact subset of this",
                         "wisconsin-original"),
    "wdbc":       Source("43_WDBC",       "general",
                         "Wisconsin diagnostic: different features from WBC",
                         "wisconsin-diagnostic"),
    "wpbc":       Source("46_WPBC",       "general",
                         "Wisconsin prognostic; documented as sharing WDBC's features",
                         "wisconsin-diagnostic"),
    "thyroid":    Source("38_thyroid",    "general", "", "thyroid"),
    "annthyroid": Source("2_annthyroid",  "general", "same UCI thyroid collection", "thyroid"),
    "aloi":       Source("1_ALOI",        "general", "object images under varied lighting"),
    "campaign":   Source("5_campaign",    "general", "bank telemarketing"),
    "celeba":     Source("8_celeba",      "general", "face attributes"),
    "census":     Source("9_census",      "general", "500 features, large n"),
    "cover":      Source("10_cover",      "general", "forest cover type, large n"),
    "donors":     Source("11_donors",     "general", "large n"),
    "fault":      Source("12_fault",      "general", "steel plate faults"),
    "glass":      Source("14_glass",      "general"),
    "hepatitis":  Source("15_Hepatitis",  "general", "80 points"),
    "letter":     Source("20_letter",     "general"),
    "lymphography": Source("21_Lymphography", "general"),
    "magic":      Source("22_magic.gamma", "general", "gamma-ray telescope"),
    "mammography": Source("23_mammography", "general"),
    "mnist":      Source("24_mnist",      "general"),
    "optdigits":  Source("26_optdigits",  "general"),
    "pageblocks": Source("27_PageBlocks", "general"),
    "pendigits":  Source("28_pendigits",  "general"),
    "pima":       Source("29_Pima",       "general"),
    "skin":       Source("33_skin",       "general", "3 features"),
    "speech":     Source("36_speech",     "general", "400 features"),
    "stamps":     Source("37_Stamps",     "general"),
    "vertebral":  Source("39_vertebral",  "general"),
    "vowels":     Source("40_vowels",     "general"),
    "waveform":   Source("41_Waveform",   "general"),
    "wilt":       Source("44_Wilt",       "general"),
    "wine":       Source("45_wine",       "general"),
    "yeast":      Source("47_yeast",      "general"),
}


def family(name: str) -> str:
    """Source family of a dataset; a dataset without one is its own family."""
    return REGISTRY[name].family or name


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
    """Load every registered dataset and return the metadata table.

    Bypasses the memoised loader: holding all 47 datasets at full size in
    memory at once would need several gigabytes (census alone is 0.9 GB).
    """
    out = []
    for name in REGISTRY:
        _, _, meta = _load_uncached(name, subsample=subsample)
        out.append(meta)
    return out


# --- memoisation ------------------------------------------------------------
# Modules routinely load the same dataset more than once in a run. Loading
# backdoor costs about 8 seconds each time, so cache the result. Arrays are
# marked read-only: a shared cache that callers can mutate in place is a
# source of bugs that are very hard to trace.

from functools import lru_cache

_load_uncached = load_dataset


@lru_cache(maxsize=32)
def load_dataset(name, **kw):                      # noqa: F811
    X, y, meta = _load_uncached(name, **kw)
    X.flags.writeable = False
    y.flags.writeable = False
    return X, y, meta


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
