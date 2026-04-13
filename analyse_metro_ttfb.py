"""
Plot TTFB CDF per client_metro from required_data.txt.
Produces two figures per metro:
  1. Full range
  2. Truncated at 200 ms
Saves to metro/figs/.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_FILE = Path("/Users/ansabni/Work/CostPerfServedTo/metro/required_data.txt")
FIG_DIR   = Path("/Users/ansabni/Work/CostPerfServedTo/metro/figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── column layout ─────────────────────────────────────────────────────────────
# The cumulative TTFB bucket upper bounds in ms (in column order).
# The last one represents > 1000 ms, treated as the [1000, 2000) bucket.
CUM_UPPER_MS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                150, 200, 300, 400, 500, 1000, 2000]

HEADER = [
    "client_metro", "region", "maprule", "is_cache_hit",
    "ttfb_le_5_ms", "ttfb_le_10_ms", "ttfb_le_15_ms", "ttfb_le_20_ms",
    "ttfb_le_25_ms", "ttfb_le_30_ms", "ttfb_le_35_ms", "ttfb_le_40_ms",
    "ttfb_le_45_ms", "ttfb_le_50_ms", "ttfb_le_55_ms", "ttfb_le_60_ms",
    "ttfb_le_65_ms", "ttfb_le_70_ms", "ttfb_le_75_ms", "ttfb_le_80_ms",
    "ttfb_le_85_ms", "ttfb_le_90_ms", "ttfb_le_95_ms", "ttfb_le_100_ms",
    "ttfb_le_150_ms", "ttfb_le_200_ms", "ttfb_le_300_ms", "ttfb_le_400_ms",
    "ttfb_le_500_ms", "ttfb_le_1000_ms", "ttfb_gt_1000_ms",
    "bytes_sent", "requests",
]

TTFB_COLS = [c for c in HEADER if c.startswith("ttfb_")]
N_BUCKETS = len(CUM_UPPER_MS)   # 27

# ── load & aggregate by client_metro ─────────────────────────────────────────
# cumulative bucket sums keyed by client_metro
cum_by_metro: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_BUCKETS, dtype=float))

with DATA_FILE.open(newline="", encoding="utf-8") as fh:
    reader = csv.reader(fh)
    for row in reader:
        if not row or row[0].strip() == "client_metro":
            continue  # skip blank / header
        metro = row[0].strip()
        # columns 4..30 inclusive are the 27 ttfb bucket counts (cumulative)
        vals = np.array([float(v) for v in row[4:4 + N_BUCKETS]], dtype=float)
        cum_by_metro[metro] += vals

print(f"Found {len(cum_by_metro)} distinct client_metro values: {sorted(cum_by_metro)}")

# ── convert cumulative counts → per-bucket counts ────────────────────────────
# lower bounds for each bucket (first bucket starts at 0)
lower_ms = np.array([0] + CUM_UPPER_MS[:-1], dtype=float)
upper_ms = np.array(CUM_UPPER_MS, dtype=float)
mid_ms   = (lower_ms + upper_ms) / 2.0

def cum_to_per_bucket(cum: np.ndarray) -> np.ndarray:
    """Cumulative → per-bucket by differencing."""
    per = np.diff(cum, prepend=0.0)
    return np.maximum(per, 0.0)   # guard against floating-point negatives

# ── plotting helper ───────────────────────────────────────────────────────────
def make_cdf_plot(metro: str, counts: np.ndarray, xlim: float | None = None):
    total = counts.sum()
    if total <= 0:
        print(f"  metro {metro}: zero requests — skipping")
        return

    # Build a dense ms-resolution series for a smooth CDF
    # expand each bucket uniformly across its ms range
    dense_ms: list[float] = []
    dense_w:  list[float] = []
    for lo, hi, cnt in zip(lower_ms, upper_ms, counts):
        if cnt <= 0:
            continue
        width = hi - lo
        # sample at every ms within the bucket
        n_pts = max(1, int(width))
        pts = np.linspace(lo, hi, n_pts, endpoint=False) + (width / n_pts / 2)
        dense_ms.extend(pts.tolist())
        dense_w.extend([cnt / n_pts] * n_pts)

    x = np.array(dense_ms, dtype=float)
    w = np.array(dense_w, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) / w.sum()

    # key percentiles
    p50_ms = x[np.searchsorted(cdf, 0.50)]
    p95_ms = x[np.searchsorted(cdf, 0.95)]
    p99_ms = x[np.searchsorted(cdf, 0.99)]

    suffix = "full" if xlim is None else f"le{int(xlim)}ms"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, cdf, color="steelblue", linewidth=2)
    for pct, val, col in [(50, p50_ms, "orange"), (95, p95_ms, "red"), (99, p99_ms, "purple")]:
        ax.axvline(val, linestyle="--", color=col, linewidth=1.2,
                   label=f"p{pct} = {val:.1f} ms")
    ax.set_xlabel("TTFB (ms)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    title_extra = f" (≤ {xlim} ms)" if xlim else ""
    ax.set_title(f"TTFB CDF — client_metro {metro}{title_extra}\n"
                 f"total requests: {int(total):,}", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0, right=xlim)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()

    out = FIG_DIR / f"ttfb_cdf_{metro}_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}  (p50={p50_ms:.1f}, p95={p95_ms:.1f}, p99={p99_ms:.1f} ms)")

# ── main ──────────────────────────────────────────────────────────────────────
for metro in sorted(cum_by_metro):
    print(f"\nmetro {metro}:")
    per_bucket = cum_to_per_bucket(cum_by_metro[metro])
    make_cdf_plot(metro, per_bucket, xlim=None)       # full range
    make_cdf_plot(metro, per_bucket, xlim=200.0)      # truncated at 200 ms

print("\nDone.")
