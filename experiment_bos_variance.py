"""
Experiment: BOS TTFB P50/P95 vs hitrate, sweeping RTT/TAT variance.

For each variance level (0, 2, 4, ... 10 ms stddev), plot P50 and P95 TTFB
as a function of hitrate (1–100%) using the BOS FDS footprint descriptor.

  edge_rtt      ~ Gaussian(mean=30ms, stddev=σ)
  edge_tat_miss ~ Gaussian(mean=20ms, stddev=σ)
  edge_tat_hit    = 0ms (point mass)
"""

import sys
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from fds import FootprintDescriptor
from perf import MetroPerformance
from probability import (
    ProbabilityDensityFunction,
    PdfBucket,
    Convolution,
    gaussian_pdf,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FDS_PATH = "FDS_OtherBigFoot/bos.txt"
METRO    = "bos"

RTT_MEAN_MS      = 30.0   # edge RTT mean
TAT_MISS_MEAN_MS = 20.0   # edge TAT-miss mean

# σ values to sweep  (0 = point-mass, then 2, 4, 6, 8, 10 ms)
STDDEV_STEPS = [0, 2, 4, 6, 8, 10]

HITRATES = list(range(1, 101))   # 1 % … 100 %

MAX_PDF_MS = 1000   # upper bound for Gaussian truncation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def point_mass_pdf(mean_ms: float) -> ProbabilityDensityFunction:
    """Return a PDF that places all mass in a 1-ms bucket at mean_ms."""
    lo = max(0, int(math.floor(mean_ms)))
    return ProbabilityDensityFunction([PdfBucket(lower_ms=lo, upper_ms=lo + 1, count=1.0)])


def make_pdf(mean_ms: float, stddev_ms: float) -> ProbabilityDensityFunction:
    """Return a Gaussian PDF, or a point mass when stddev == 0."""
    if stddev_ms == 0.0:
        return point_mass_pdf(mean_ms)
    lo = max(0, int(mean_ms - 6 * stddev_ms))
    hi = min(MAX_PDF_MS, int(mean_ms + 6 * stddev_ms) + 1)
    return gaussian_pdf(mean=mean_ms, stddev=stddev_ms, lower_ms=lo, upper_ms=hi)


def zero_ms_pdf() -> ProbabilityDensityFunction:
    """Point mass at 0 ms (used for edge_tat_hit = 0)."""
    return ProbabilityDensityFunction([PdfBucket(lower_ms=0, upper_ms=1, count=1.0)])


# ---------------------------------------------------------------------------
# Load BOS FDS
# ---------------------------------------------------------------------------
descriptor = FootprintDescriptor.from_file(FDS_PATH)
print(f"Loaded BOS FDS: {len(descriptor._points_sorted_by_hitrate)} unique hitrate points "
      f"(range {descriptor._hitrates[0]:.1f}% – {descriptor._hitrates[-1]:.1f}%)")

conn = Convolution()

# ---------------------------------------------------------------------------
# Main loop: one subplot per σ
# ---------------------------------------------------------------------------
ncols = 3
nrows = math.ceil(len(STDDEV_STEPS) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), constrained_layout=True)
axes_flat = axes.flatten()

for ax_idx, stddev in enumerate(STDDEV_STEPS):
    ax = axes_flat[ax_idx]

    # Build PDFs for this σ
    rtt_pdf      = make_pdf(RTT_MEAN_MS,      stddev)
    tat_miss_pdf = make_pdf(TAT_MISS_MEAN_MS, stddev)
    tat_hit_pdf  = zero_ms_pdf()

    # Wire up MetroPerformance
    mp = MetroPerformance(name=METRO, descriptor=descriptor)
    mp.set_edge_rtt(METRO, rtt_pdf)
    mp.set_edge_tat_hit(tat_hit_pdf)
    mp.set_edge_tat_miss(tat_miss_pdf)

    p50_list, p95_list, p99_list = [], [], []
    for hr in HITRATES:
        ttfb = mp.get_ttfb_pdf(METRO, hr, conn)
        p50_list.append(ttfb.millisecond_at_percentile(50))
        p95_list.append(ttfb.millisecond_at_percentile(95))
        p99_list.append(ttfb.millisecond_at_percentile(99))

    ax.plot(HITRATES, p50_list, label="P50", linewidth=2)
    ax.plot(HITRATES, p95_list, label="P95", linewidth=2, linestyle="--")
    ax.plot(HITRATES, p99_list, label="P99", linewidth=2, linestyle=":")
    ax.set_xlabel("Hit-rate (%)")
    ax.set_ylabel("TTFB (ms)")
    sigma_label = f"σ = {stddev} ms" if stddev > 0 else "σ = 0 ms  (point mass)"
    ax.set_title(sigma_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Hide any unused subplots
for ax_idx in range(len(STDDEV_STEPS), len(axes_flat)):
    axes_flat[ax_idx].set_visible(False)

fig.suptitle(
    f"BOS TTFB P50 / P95 / P99 vs Hit-rate\n"
    f"edge_rtt ~ N({RTT_MEAN_MS}ms, σ),  edge_tat_miss ~ N({TAT_MISS_MEAN_MS}ms, σ),  edge_tat_hit = 0ms",
    fontsize=13,
)

out_path = "bos_variance_experiment.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
