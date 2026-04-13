"""Plot P50 and P95 latency distributions across metros.

Compares:
  - Gradient descent at iteration 14 (more_aggressive log)
  - Knee operating point (less_aggressive knee_analysis.txt)

Each metro is shown as a point sized by its client traffic (TRAFFIC_FROM).
Both a CDF and a scatter/strip plot are produced for P50 and P95.

Usage:
    python plot_latency_distributions.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR        = Path(__file__).resolve().parent
_GRAD_LOG        = _BASE_DIR / "US_traffic_based_penalty_more_aggressive" / "gradient_descent_metro_log.txt"
_KNEE_FILE       = _BASE_DIR / "US_traffic_based_penalty_less_aggressive" / "knee_analysis.txt"
_OUT_PNG         = _BASE_DIR / "latency_distributions.png"
_TARGET_ITER     = 14

# ---------------------------------------------------------------------------
# Load TRAFFIC_FROM from solve_for_US (consistent with other scripts)
# ---------------------------------------------------------------------------
import solve_for_US as _s

airport_info, metro_to_airport, airport_to_metro = _s.parse_metro_areas(
    _BASE_DIR / "PERF" / "metro_areas.csv"
)
traffic_lookup = _s._load_traffic_lookup()

neighborhood_from: dict[str, list[str]] = defaultdict(list)
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if traffic > 10000 and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        neighborhood_from[metro_to_airport[bw_metro]].append(metro_to_airport[asn_metro])

traffic_lookup_by_airport: dict[tuple[str, str], float] = {}
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        traffic_lookup_by_airport[
            (metro_to_airport[asn_metro], metro_to_airport[bw_metro])
        ] = traffic

metros = list(airport_info.keys())
TRAFFIC_FROM: dict[str, float] = defaultdict(float)
for metro in metros:
    for from_metro in neighborhood_from.get(metro, []):
        t = traffic_lookup_by_airport.get((from_metro, metro), 0.0)
        TRAFFIC_FROM[from_metro] += t

TRAFFIC: dict[str, float] = {m: TRAFFIC_FROM[m] for m in metros if TRAFFIC_FROM[m] > 0}

# ---------------------------------------------------------------------------
# Parse iteration 14 from gradient log
# ---------------------------------------------------------------------------
_ITER_RE  = re.compile(r"^iteration=(\d+)")
_METRO_RE = re.compile(
    r"^(\w+),\s*disk_mb=[\d.]+,\s*hitrate=[\d.]+,\s*cost=[\d.]+,"
    r"\s*perf_penalty=[\d.]+,\s*p50=([\d.]+),\s*p95=([\d.]+)"
)

grad_data: dict[str, dict] = {}   # metro -> {p50, p95}
in_target = False

with open(_GRAD_LOG) as fh:
    for line in fh:
        line = line.rstrip()
        m = _ITER_RE.match(line)
        if m:
            if in_target:
                break          # done with iteration 14
            in_target = (int(m.group(1)) == _TARGET_ITER)
            continue
        if in_target:
            m2 = _METRO_RE.match(line)
            if m2:
                metro, p50, p95 = m2.group(1), float(m2.group(2)), float(m2.group(3))
                grad_data[metro] = {"p50": p50, "p95": p95}

print(f"Parsed {len(grad_data)} metros from iteration {_TARGET_ITER} of gradient log.")

# ---------------------------------------------------------------------------
# Parse knee_analysis.txt  (P50/P95 in the Knee columns)
# ---------------------------------------------------------------------------
_KNEE_RE = re.compile(
    r"^\s+(\w+)\s+[\d,]+\s+[\d,]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)"
)

# More robust: split on whitespace, pick columns by position
# Line structure (disk metros):
#   Metro  Traffic  Disk(MB)  HR%  Cost$/mo  p50  p95  Penalty  ...
# Line structure (no-disk metros):
#   Metro  Traffic  0  0.0  0.00  p50  p95  Penalty  ...
# We detect the section and parse accordingly.

knee_data: dict[str, dict] = {}

with open(_KNEE_FILE) as fh:
    for line in fh:
        line = line.rstrip()
        if line.startswith("---") or line.startswith("Metro") or line.startswith("TOTAL"):
            continue
        if "(no local disk" in line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        metro = parts[0]
        if not metro.isalpha():
            continue
        # Disk metros: Metro Traffic Disk HR Cost p50 p95 Penalty Disk HR Cost p50 p95 Penalty % Src
        # No-disk:    Metro Traffic 0    0.0 0.00 p50 p95 Penalty 0    0.0 0.00 p50 p95 Penalty ...
        # Column 5 (0-indexed) = knee_p50, column 6 = knee_p95
        try:
            p50 = float(parts[5])
            p95 = float(parts[6])
        except (ValueError, IndexError):
            continue
        knee_data[metro] = {"p50": p50, "p95": p95}

print(f"Parsed {len(knee_data)} metros from knee_analysis.txt.")

# ---------------------------------------------------------------------------
# Build common metro set with valid latency data and positive traffic
# ---------------------------------------------------------------------------
# Exclude metros with no perf model (TUS, ANC — p50=p95=0 in grad log)
common_metros = sorted(
    m for m in TRAFFIC
    if m in grad_data and m in knee_data
    and not (grad_data[m]["p50"] == 0.0 and grad_data[m]["p95"] == 0.0)
    and not (knee_data[m]["p50"] == 0.0 and knee_data[m]["p95"] == 0.0)
)

print(f"Common metros with valid data: {len(common_metros)}")

# Arrays aligned to common_metros
grad_p50  = np.array([grad_data[m]["p50"]  for m in common_metros])
grad_p95  = np.array([grad_data[m]["p95"]  for m in common_metros])
knee_p50  = np.array([knee_data[m]["p50"]  for m in common_metros])
knee_p95  = np.array([knee_data[m]["p95"]  for m in common_metros])
weights   = np.array([TRAFFIC[m]           for m in common_metros])
weights_n = weights / weights.sum()          # normalised for CDF

# Traffic-weighted average helper
def wavg(vals: np.ndarray) -> float:
    return float(np.dot(vals, weights_n))

# ---------------------------------------------------------------------------
# Sort by traffic for CDF
# ---------------------------------------------------------------------------
order = np.argsort(weights)[::-1]   # heaviest first (not needed for CDF but useful)

# For CDF: sort each series independently
def traffic_weighted_cdf(vals: np.ndarray, wts: np.ndarray):
    """Return (sorted_vals, cumulative_traffic_fraction)."""
    idx   = np.argsort(vals)
    sv    = vals[idx]
    sw    = wts[idx] / wts.sum()
    cdf   = np.cumsum(sw)
    return sv, cdf

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
GRAD_COL = "#2176AE"    # blue  — gradient iter 14
KNEE_COL = "#E87722"    # orange — knee

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Latency Distribution — Gradient Descent (iter {_TARGET_ITER}, more-aggressive)  vs  Knee (less-aggressive)",
    fontsize=13, fontweight="bold", y=1.01
)

# ── helper: draw one CDF panel ────────────────────────────────────────────
def draw_cdf(ax, grad_vals, knee_vals, metric_label):
    gx, gy = traffic_weighted_cdf(grad_vals, weights)
    kx, ky = traffic_weighted_cdf(knee_vals,  weights)

    ax.step(gx, gy * 100, where="post", color=GRAD_COL, lw=2,
            label=f"Gradient iter {_TARGET_ITER}  (wtd-avg {wavg(grad_vals):.1f} ms)")
    ax.step(kx, ky * 100, where="post", color=KNEE_COL, lw=2, linestyle="--",
            label=f"Knee  (wtd-avg {wavg(knee_vals):.1f} ms)")

    # Mark 50th / 75th / 90th percentile lines
    for pct, ls in [(50, ":"), (75, "-."), (90, "--")]:
        ax.axhline(pct, color="grey", lw=0.6, linestyle=ls, alpha=0.5)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 200,
                pct + 1, f"{pct}%", va="bottom", ha="right",
                fontsize=7, color="grey")

    ax.set_xlabel(f"{metric_label} latency (ms)", fontsize=11)
    ax.set_ylabel("Cumulative traffic fraction (%)", fontsize=11)
    ax.set_title(f"{metric_label} CDF (traffic-weighted)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)

draw_cdf(axes[0], grad_p50, knee_p50, "P50")
draw_cdf(axes[1], grad_p95, knee_p95, "P95")

# After drawing, add percentile text at the right edge now that xlim is set
for ax, grad_vals, knee_vals, label in [
    (axes[0], grad_p50, knee_p50, "P50"),
    (axes[1], grad_p95, knee_p95, "P95"),
]:
    xmax = max(grad_vals.max(), knee_vals.max()) * 1.02
    ax.set_xlim(left=0, right=xmax)
    for pct, ls in [(50, ":"), (75, "-."), (90, "--")]:
        ax.text(xmax * 0.99, pct + 1, f"{pct}%",
                va="bottom", ha="right", fontsize=7, color="grey")

plt.tight_layout()
fig.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved plot → {_OUT_PNG}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
header = f"{'Metro':>5}  {'Traffic':>8}  {'Grad-P50':>8}  {'Knee-P50':>8}  {'ΔP50':>7}  {'Grad-P95':>8}  {'Knee-P95':>8}  {'ΔP95':>7}"
print()
print(header)
print("-" * len(header))
for m in sorted(common_metros, key=lambda x: -TRAFFIC[x]):
    gp50 = grad_data[m]["p50"]
    kp50 = knee_data[m]["p50"]
    gp95 = grad_data[m]["p95"]
    kp95 = knee_data[m]["p95"]
    print(f"{m:>5}  {TRAFFIC[m]:>8,.0f}  {gp50:>8.1f}  {kp50:>8.1f}  {kp50-gp50:>+7.1f}  {gp95:>8.1f}  {kp95:>8.1f}  {kp95-gp95:>+7.1f}")

print("-" * len(header))
print(f"{'Wtd-avg':>5}  {'':>8}  {wavg(grad_p50):>8.2f}  {wavg(knee_p50):>8.2f}  {wavg(knee_p50)-wavg(grad_p50):>+7.2f}  {wavg(grad_p95):>8.2f}  {wavg(knee_p95):>8.2f}  {wavg(knee_p95)-wavg(grad_p95):>+7.2f}")
