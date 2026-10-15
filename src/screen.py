"""Screen every ADBench Classical dataset and record the development/benchmark split.

    python src/screen.py             # all 47 (downloads what is missing; 83 MB in all)
    python src/screen.py --offline   # only datasets already in data/raw
    python src/screen.py --reuse     # re-apply the rules to the saved table

Writes results/screen_table.json and results/splits.json. Both are tracked in
git: they are the record of which data the thesis uses and why.

CRITERIA. Each is a choice to be defended; the sensitivity table shows how the
selection moves when any one threshold does.
  D    >= 10    the premise is that structure is lost compressing many
                dimensions into two; with few dimensions, nothing is lost
  AUC  >= 0.65  kNN (k = 20) in the original space. Below this the labelled
                anomalies are not geometrically outlying, so there is no outlier
                structure for a projection to preserve or destroy
  rate <= 25%   above this the task is classification, not outlier detection
  n    >= 200   with k = 20, smaller datasets make "local" span a tenth or more
                of the data, and the analyst's shortlist is a handful of points
Statistics use at most 20,000 points per dataset (stratified, seed 0, after
duplicate removal), so AUC and LID are estimates of their full-data values.

INDEPENDENCE. Datasets sharing a source family count once in the benchmark;
the member with the lowest anomaly rate is kept. Cross-dataset tests such as
Friedman's assume the datasets are independent.

THE SPLIT. The measure is developed only on datasets the screen excludes
anyway, so the benchmark loses nothing, and no dataset or source family is
used both to shape the measure and to evaluate it. The development set is
chosen by hand and checked by machine: propose_split() refuses any choice that
breaks those rules.
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from config import DATA_RAW, DEFAULT_K, DEFAULT_SEED, RESULTS
from datasets import REGISTRY, family, load_dataset
from measure import n_distinct
from neighbors import knn_graph, knn_score, lid_mle

CAP = 20_000
K = DEFAULT_K
CRITERIA = {"min_D": 10, "min_auc": 0.65, "max_rate": 25.0, "min_n": 200}
TABLE = RESULTS / "screen_table.json"
SPLITS = RESULTS / "splits.json"
SPLIT_DATE = "2026-10-15"   # change only when the split itself changes

# Hand-chosen, machine-checked; see propose_split() for the rules enforced.
# Each has a distinct job while the measure is being built.
DEVELOPMENT: dict[str, str] = {
    "shuttle": "low-LID control (median about 2): projection should lose little, so "
               "heavy reported damage means the measure is wrong. Also the tie test "
               "bed: 49,097 points share few distinct scores. Large n.",
    "wbc": "clean negative control: near-perfect detection (AUC 0.99), small, fast "
           "to iterate on. Its source family (wisconsin-original) is not in the "
           "benchmark; WDBC there is a different feature set.",
    "ionosphere": "the positive case: the heaviest LID tail found (90th percentile "
                  "31 in 32 dimensions) and the dataset PCA damages most.",
    "fault": "high ambient, low intrinsic dimension (27 features, median LID 4.8): "
             "the opposite profile to ionosphere at a similar dimensionality.",
    "magic": "scale: about 19,000 points, so a permutation null or tie policy "
             "that is only affordable on small data shows it here.",
}

CASE_STUDY: dict[str, str] = {
    "backdoor": "the only high-dimensional network-intrusion dataset that passes "
                "the screen; also analysed at full size as the security case study",
}


def screen_one(name: str) -> dict:
    """Screening statistics for one dataset.

    Floats are rounded so that re-running on another machine does not show
    up in git as changes in the fifteenth decimal place.
    """
    X, y, meta = load_dataset(name, subsample=CAP, seed=DEFAULT_SEED)
    n_dedup = meta["n_full"] - meta["duplicates_removed"]
    k = min(K, meta["n"] - 1)
    dist, _ = knn_graph(np.asarray(X), k)
    s = knn_score(dist, k)
    lid = lid_mle(dist, k)
    return {
        "name": name,
        "file": REGISTRY[name].file,
        "domain": REGISTRY[name].domain,
        "family": family(name),
        "n_full": meta["n_full"],
        "n": n_dedup,
        "n_screened": meta["n"],
        "D": meta["D"],
        "rate": round(100 * meta["anomaly_rate"], 4),
        "n_anomalies_screened": meta["n_anomalies"],
        "duplicates_pct": round(100 * meta["duplicates_removed"] / meta["n_full"], 4),
        "auc": round(float(roc_auc_score(y, s)), 6),
        "lid_median": round(float(np.nanmedian(lid)), 4),
        "lid_p90": round(float(np.nanpercentile(lid, 90)), 4),
        "distinct_score_frac": round(n_distinct(s) / len(s), 6),
        "data_hash": meta["data_hash"],
    }


def failures(r: dict, c: dict = CRITERIA) -> list[str]:
    out = []
    if r["D"] < c["min_D"]:
        out.append(f"D<{c['min_D']}")
    if r["auc"] < c["min_auc"]:
        out.append(f"AUC<{c['min_auc']}")
    if r["rate"] > c["max_rate"]:
        out.append(f"rate>{c['max_rate']:g}%")
    if r["n"] < c["min_n"]:
        out.append(f"n<{c['min_n']}")
    return out


def benchmark(rows: list[dict], c: dict = CRITERIA) -> tuple[list[dict], list[dict]]:
    """Datasets passing every criterion, one per family. Returns (kept, dropped)."""
    passing = [r for r in rows if not failures(r, c)]
    kept, dropped, best = [], [], {}
    for r in sorted(passing, key=lambda r: (r["family"], r["rate"])):
        if r["family"] in best:
            dropped.append({**r, "dropped_for": best[r["family"]]})
        else:
            best[r["family"]] = r["name"]
            kept.append(r)
    return kept, dropped


def propose_split(rows: list[dict]) -> dict:
    by = {r["name"]: r for r in rows}
    bench, dropped = benchmark(rows)
    bench_names = {r["name"] for r in bench}
    bench_families = {r["family"] for r in bench}

    problems = []
    for name in DEVELOPMENT:
        r = by.get(name)
        if r is None:
            problems.append(f"{name}: not screened")
            continue
        if not failures(r):
            problems.append(f"{name}: passes the screen, so using it for development "
                            f"would remove it from the benchmark")
        if r["auc"] < CRITERIA["min_auc"]:
            problems.append(f"{name}: AUC {r['auc']:.3f} shows no outlier structure to develop on")
        if r["family"] in bench_families:
            problems.append(f"{name}: shares family '{r['family']}' with a benchmark dataset")
    for name in CASE_STUDY:
        if name not in bench_names:
            problems.append(f"{name}: case study is not in the benchmark")
    if problems:
        raise SystemExit("Split rejected:\n  " + "\n  ".join(problems))

    dev_families = {by[n]["family"] for n in DEVELOPMENT}
    candidates = [r["name"] for r in rows
                  if failures(r) and r["auc"] >= CRITERIA["min_auc"]
                  and r["family"] not in bench_families and r["name"] not in DEVELOPMENT]
    used = bench_names | set(DEVELOPMENT)
    return {
        "status": "proposed; pending supervisor confirmation",
        "date": SPLIT_DATE,
        "criteria": CRITERIA,
        "screen_cap": CAP,
        "k": K,
        "benchmark": [{"name": r["name"], "family": r["family"], "n": r["n"], "D": r["D"],
                       "rate": round(r["rate"], 2), "auc": round(r["auc"], 3),
                       "lid_median": round(r["lid_median"], 2)} for r in bench],
        "benchmark_dropped_for_family": [{"name": r["name"], "kept_instead": r["dropped_for"]}
                                         for r in dropped],
        "development": [{"name": n, "family": by[n]["family"], "why": why,
                         "fails": failures(by[n]), "auc": round(by[n]["auc"], 3),
                         "lid_median": round(by[n]["lid_median"], 2),
                         "lid_p90": round(by[n]["lid_p90"], 2)}
                        for n, why in DEVELOPMENT.items()],
        "development_unused_candidates": candidates,
        "case_study": [{"name": n, "why": why} for n, why in CASE_STUDY.items()],
        "excluded": [{"name": r["name"], "fails": failures(r)}
                     for r in rows if r["name"] not in used and failures(r)],
        "families_on_both_sides": sorted(dev_families & bench_families),
    }


def sensitivity(rows: list[dict]) -> list[tuple[str, object, int, list[str], list[str]]]:
    base = {r["name"] for r in benchmark(rows)[0]}
    grid = {"min_D": [8, 10, 15, 20], "min_auc": [0.60, 0.65, 0.70, 0.75],
            "max_rate": [15.0, 20.0, 25.0, 30.0], "min_n": [0, 100, 200, 500]}
    out = []
    for key, values in grid.items():
        for v in values:
            c = {**CRITERIA, key: v}
            names = {r["name"] for r in benchmark(rows, c)[0]}
            out.append((key, v, len(names), sorted(names - base), sorted(base - names)))
    return out


def main() -> None:
    reuse = "--reuse" in sys.argv
    offline = "--offline" in sys.argv

    if reuse and TABLE.exists():
        rows = json.loads(TABLE.read_text())
    else:
        names = sorted(REGISTRY, key=lambda n: int(REGISTRY[n].file.split("_")[0]))
        if offline:
            names = [n for n in names if (DATA_RAW / f"{REGISTRY[n].file}.npz").exists()]
        rows = []
        for name in names:
            print(f"  screening {name} ...", flush=True)
            rows.append(screen_one(name))
        TABLE.write_text(json.dumps(rows, indent=2))

    print(f"\n{'dataset':<17}{'family':<21}{'n':>8}{'D':>6}{'rate%':>7}{'AUC':>7}"
          f"{'LID':>7}{'p90':>7}  verdict")
    print("-" * 100)
    bench_names = {r["name"] for r in benchmark(rows)[0]}
    for r in rows:
        f = failures(r)
        verdict = "BENCHMARK" if r["name"] in bench_names else (
            ", ".join(f) if f else "passes; family already represented")
        if r["name"] in DEVELOPMENT:
            verdict = "DEVELOPMENT  (" + ", ".join(f) + ")"
        fam = r["family"] if r["family"] != r["name"] else ""
        print(f"{r['name']:<17}{fam:<21}{r['n']:>8}{r['D']:>6}{r['rate']:>7.1f}"
              f"{r['auc']:>7.3f}{r['lid_median']:>7.2f}{r['lid_p90']:>7.2f}  {verdict}")

    print("\nSensitivity: benchmark size as one threshold moves (others at default)")
    for key, v, n, added, removed in sensitivity(rows):
        mark = "  <- default" if CRITERIA[key] == v else ""
        change = ""
        if added:
            change += "  +" + " +".join(added)
        if removed:
            change += "  -" + " -".join(removed)
        print(f"  {key:<9}{v!s:>6}  {n:>3} datasets{mark}{change}")

    split = propose_split(rows)
    SPLITS.write_text(json.dumps(split, indent=2))
    print(f"\nBenchmark {len(split['benchmark'])}, development {len(split['development'])}, "
          f"case study {len(split['case_study'])}, excluded {len(split['excluded'])}")
    print(f"Unused development candidates: {', '.join(split['development_unused_candidates']) or 'none'}")
    print(f"Wrote {TABLE}\nWrote {SPLITS}")


if __name__ == "__main__":
    main()
