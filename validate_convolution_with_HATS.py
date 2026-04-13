"""
Plot TTFB CDF for every (client_metro, region) pair in
check_convolution_hats_data/required_data.txt.

Column layout (no header):
  0-26  : ttfb cumulative bucket counts (le_5 … gt_1000)
  27    : bytes_sent
  28    : requests
  29    : is_cache_hit   (1 = hit, 0 = miss)
  30    : client_metro
  31    : region
  32    : maprule

Output: one PNG per pair saved to check_convolution_hats_data/figs/
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
DATA_FILE = Path("/Users/ansabni/Work/CostPerfServedTo/check_convolution_hats_data/required_data.txt")
FIG_DIR   = Path("/Users/ansabni/Work/CostPerfServedTo/check_convolution_hats_data/figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── bucket definitions ────────────────────────────────────────────────────────
# Upper bounds in ms for the 27 cumulative buckets.
# The last bucket (gt_1000_ms) is treated as [1000, 2000).
CUM_UPPER_MS = [
     5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
    150, 200, 300, 400, 500, 1000, 2000,
]
N_BUCKETS = len(CUM_UPPER_MS)  # 27
lower_ms  = np.array([0] + CUM_UPPER_MS[:-1], dtype=float)
upper_ms  = np.array(CUM_UPPER_MS, dtype=float)

# ── accumulators keyed by (client_metro, region) ─────────────────────────────
# Each value: [cum_ttfb_array, total_requests, hit_requests]
PairKey = tuple[str, str]
cum_ttfb:   dict[PairKey, np.ndarray] = defaultdict(lambda: np.zeros(N_BUCKETS, dtype=float))
total_reqs: dict[PairKey, float]      = defaultdict(float)
hit_reqs:   dict[PairKey, float]      = defaultdict(float)

with DATA_FILE.open(newline="", encoding="utf-8") as fh:
    reader = csv.reader(fh)
    for row in reader:
        if len(row) < 33:
            continue
        # parse key fields
        client_metro  = row[30].strip()
        region        = row[31].strip()
        is_hit        = int(row[29].strip())
        requests      = float(row[28].strip())
        key: PairKey  = (client_metro, region)

        # accumulate ttfb buckets (cols 0-26)
        vals = np.array([float(v) for v in row[0:N_BUCKETS]], dtype=float)
        cum_ttfb[key]   += vals
        total_reqs[key] += requests
        hit_reqs[key]   += requests * is_hit

pairs = sorted(cum_ttfb.keys())
print(f"Found {len(pairs)} (client_metro, region) pairs")

# ── helpers ───────────────────────────────────────────────────────────────────

def cum_to_per_bucket(cum: np.ndarray) -> np.ndarray:
    """Cumulative counts → per-bucket counts."""
    return np.maximum(np.diff(cum, prepend=0.0), 0.0)


def build_dense_cdf(counts: np.ndarray):
    """
    Expand each bucket uniformly across its ms range to produce a
    dense (x_ms, cdf) pair suitable for smooth plotting.
    """
    dense_x: list[float] = []
    dense_w: list[float] = []
    for lo, hi, cnt in zip(lower_ms, upper_ms, counts):
        if cnt <= 0:
            continue
        width  = hi - lo
        n_pts  = max(1, int(width))
        pts    = np.linspace(lo, hi, n_pts, endpoint=False) + (width / n_pts / 2)
        dense_x.extend(pts.tolist())
        dense_w.extend([cnt / n_pts] * n_pts)

    if not dense_x:
        return np.array([0.0]), np.array([0.0])

    x = np.array(dense_x, dtype=float)
    w = np.array(dense_w, dtype=float)
    order = np.argsort(x)
    x, w  = x[order], w[order]
    cdf   = np.cumsum(w) / w.sum()
    return x, cdf


def percentile_from_cdf(x: np.ndarray, cdf: np.ndarray, p: float) -> float:
    idx = np.searchsorted(cdf, p / 100.0)
    return float(x[min(idx, len(x) - 1)])


# ── main plotting loop ────────────────────────────────────────────────────────
for (metro, region) in pairs:
    key    = (metro, region)
    counts = cum_to_per_bucket(cum_ttfb[key])
    total  = total_reqs[key]

    if total <= 0 or counts.sum() <= 0:
        print(f"  {metro}/{region}: no data — skipping")
        continue

    hitrate = hit_reqs[key] / total * 100.0
    x, cdf  = build_dense_cdf(counts)
    p50  = percentile_from_cdf(x, cdf, 50)
    p95  = percentile_from_cdf(x, cdf, 95)
    p99  = percentile_from_cdf(x, cdf, 99)

    print(f"  metro={metro} region={region}  "
          f"requests={int(total):,}  hitrate={hitrate:.1f}%  "
          f"p50={p50:.1f} ms  p95={p95:.1f} ms  p99={p99:.1f} ms")

    for xlim, suffix in [(None, "full"), (200.0, "le200ms")]:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, cdf, color="steelblue", linewidth=2, label="CDF")

        for pct, val, col in [(50, p50, "orange"), (95, p95, "red"), (99, p99, "purple")]:
            ax.axvline(val, linestyle="--", linewidth=1.2, color=col,
                       label=f"p{pct} = {val:.1f} ms")

        ax.set_xlabel("TTFB (ms)", fontsize=12)
        ax.set_ylabel("CDF", fontsize=12)
        xlim_str = f" (≤ {int(xlim)} ms)" if xlim else ""
        ax.set_title(
            f"TTFB CDF — client_metro={metro}, region={region}{xlim_str}\n"
            f"requests: {int(total):,}  |  hitrate: {hitrate:.1f}%",
            fontsize=11,
        )
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0, right=xlim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10)
        fig.tight_layout()

        fname = f"ttfb_cdf_metro{metro}_region{region}_{suffix}.png"
        fig.savefig(FIG_DIR / fname, dpi=150)
        plt.close(fig)

print(f"\nAll figures saved to {FIG_DIR}")
