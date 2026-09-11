"""Central path and seed configuration.

Every other module imports paths from here rather than building its own.
That way there is exactly one place to change if the layout moves, and no
module can quietly write outside the tree.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

_DIRS = (DATA_RAW, DATA_PROCESSED, CACHE, RESULTS, FIGURES)

for _d in _DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# Fallback only. Every stochastic call takes a seed argument explicitly,
# and that seed goes into the output filename. Week 8 is an analysis of
# seed variance and it is impossible if any seed upstream was implicit.
DEFAULT_SEED = 0

# Default neighbourhood size. Used as the cache key alongside the dataset
# name, so changing it produces a new cache entry rather than a stale hit.
DEFAULT_K = 20


if __name__ == "__main__":
    print(f"Repository root: {ROOT}")
    for d in _DIRS:
        print(f"  {d.relative_to(ROOT)}  exists={d.exists()}")
