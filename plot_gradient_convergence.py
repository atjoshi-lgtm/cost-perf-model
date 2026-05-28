"""Parse gradient_descent_metro_log.txt iteration by iteration.

For each iteration compute:
  - total cost ($/mo)
  - traffic-weighted P50 latency (ms)
  - traffic-weighted P95 latency (ms)

Traffic weights are the client Mbps for each metro (TRAFFIC_FROM), derived
directly from the traffic lookup — so zero-disk metros whose traffic is served
by neighbours are correctly included with their actual client Mbps as weight.

Outputs:
  - a text table  → <folder>/convergence_by_iteration.txt
  - a 3-panel plot → <folder>/convergence_by_iteration.png
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate files
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent

def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                        help="Folder containing gradient_descent_metro_log.txt")
    parser.add_argument("--geo", type=str, default="NA",
                        help="Macroarea (e.g. NA, EMEA, LA, APAC)")
    parser.add_argument("--bucket", type=str, default="AkamaiHD",
                        help="FDS bucket name (e.g. AkamaiHD, OtherBigFoot)")
    parser.add_argument("--traffic-threshold", type=float, default=10000.0,
                        help="Minimum traffic (Mbps) to include a network edge")
    return parser.parse_args()

_args = _get_args()
_GEO  = _args.geo
_BUCKET = _args.bucket
_TRAFFIC_THRESHOLD = _args.traffic_threshold

folder_arg = _args.dir
LOG_FOLDER = Path(folder_arg) if Path(folder_arg).is_absolute() else _BASE_DIR / folder_arg
LOG_PATH   = LOG_FOLDER / "gradient_descent_metro_log.txt"

if not LOG_PATH.exists():
    sys.exit(f"File not found: {LOG_PATH}")

# ---------------------------------------------------------------------------
# Load traffic weights from solve_for_US infrastructure.
# Use TRAFFIC_FROM[metro] = total Mbps that metro's clients send outward.
# This is non-zero for every metro that has users, including zero-disk metros
# whose traffic is served by neighbours.
# ---------------------------------------------------------------------------
import solve_for_US as _s

_s._GEO    = _GEO
_s._BUCKET = _BUCKET

airport_info, metro_to_airport, airport_to_metro = _s.parse_metro_areas(
    _BASE_DIR / "PERF" / "metro_areas.csv", _GEO
)
traffic_lookup = _s._load_traffic_lookup()

neighborhood_from: dict[str, list[str]] = defaultdict(list)
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if traffic > _TRAFFIC_THRESHOLD and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
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

# TRAFFIC[metro] = client Mbps weight (non-zero for all active metros)
TRAFFIC: dict[str, float] = {m: TRAFFIC_FROM[m] for m in metros if TRAFFIC_FROM[m] > 0}

# ---------------------------------------------------------------------------
# Build per-region traffic weights (EMEA only — no-op for other geos)
# Mirrors _load_emea_airport_regions / _EMEA_COUNTRY_THRESHOLDS in solve_for_US.py
# ---------------------------------------------------------------------------
_REGION_TRAFFIC: dict[str, dict[str, float]] = {}  # region -> {metro: Mbps}

if _GEO == "EMEA":
    airport_regions, airport_countries = _s._load_emea_airport_regions()
    _COUNTRY_REGION_OVERRIDE: dict[str, str] = {
        "South Africa": "South Africa",  # treated as its own sub-region in plots
    }
    for metro, w in TRAFFIC.items():
        country = airport_countries.get(metro)
        region  = airport_regions.get(metro)
        # Apply country-level override (e.g. South Africa shown separately)
        if country and country in _COUNTRY_REGION_OVERRIDE:
            bucket = _COUNTRY_REGION_OVERRIDE[country]
        elif region:
            bucket = region
        else:
            bucket = "Other"
        _REGION_TRAFFIC.setdefault(bucket, {})[metro] = w

EMEA_REGIONS = sorted(_REGION_TRAFFIC.keys())  # e.g. ["Africa", "Europe", "Middle East", "South Africa"]

# ---------------------------------------------------------------------------
# Parse every iteration from the log
# ---------------------------------------------------------------------------
_ITER_RE  = re.compile(r"^iteration=(\d+)")
_METRO_RE = re.compile(
    r"^(\w+),\s*disk_mb=([\d.]+),\s*hitrate=([\d.]+),\s*cost=([\d.]+),"
    r"\s*perf_penalty=([\d.]+),\s*p50=([\d.]+),\s*p95=([\d.]+)"
)

# iterations[n] = { metro: {cost, p50, p95} }
iterations: dict[int, dict[str, dict]] = {}
current_iter: int | None = None

with LOG_PATH.open(encoding="utf-8") as fh:
    for raw_line in fh:
        line = raw_line.strip()
        m = _ITER_RE.match(line)
        if m:
            current_iter = int(m.group(1))
            iterations[current_iter] = {}
            continue
        m2 = _METRO_RE.match(line)
        if m2 and current_iter is not None:
            metro = m2.group(1)
            iterations[current_iter][metro] = dict(
                cost = float(m2.group(4)),
                p50  = float(m2.group(6)),
                p95  = float(m2.group(7)),
            )

if not iterations:
    sys.exit("No iteration data found in log.")

print(f"Parsed {len(iterations)} iterations, {len(next(iter(iterations.values())))} metros each.")

# ---------------------------------------------------------------------------
# Compute per-iteration summary
# ---------------------------------------------------------------------------
def weighted_avg(values: dict[str, float], weights: dict[str, float]) -> float:
    # Exclude metros with no perf model (p50/p95 logged as exactly 0.0, e.g. TUS, ANC)
    active = {m: v for m, v in values.items() if v > 0.0}
    total_w = sum(weights.get(m, 0.0) for m in active)
    if total_w == 0:
        return 0.0
    return sum(active[m] * weights.get(m, 0.0) for m in active) / total_w

records: list[dict] = []
for it_num in sorted(iterations):
    metros_data = iterations[it_num]
    total_cost = sum(d["cost"] for d in metros_data.values())
    p50_dict   = {m: d["p50"] for m, d in metros_data.items()}
    p95_dict   = {m: d["p95"] for m, d in metros_data.items()}
    wavg_p50   = weighted_avg(p50_dict, TRAFFIC)
    wavg_p95   = weighted_avg(p95_dict, TRAFFIC)

    region_p50: dict[str, float] = {}
    region_p95: dict[str, float] = {}
    for region, region_weights in _REGION_TRAFFIC.items():
        region_p50[region] = weighted_avg(p50_dict, region_weights)
        region_p95[region] = weighted_avg(p95_dict, region_weights)

    records.append(dict(
        iteration  = it_num,
        total_cost = total_cost,
        wavg_p50   = wavg_p50,
        wavg_p95   = wavg_p95,
        region_p50 = region_p50,
        region_p95 = region_p95,
    ))

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
HDR = f"{'Iter':>5}  {'Total Cost ($/mo)':>18}  {'Wtd-avg P50 (ms)':>18}  {'Wtd-avg P95 (ms)':>18}"
SEP = "-" * len(HDR)
lines = [SEP, HDR, SEP]
for r in records:
    lines.append(
        f"{r['iteration']:>5}  {r['total_cost']:>18.2f}  {r['wavg_p50']:>18.2f}  {r['wavg_p95']:>18.2f}"
    )
lines.append(SEP)

table = "\n".join(lines)
print(table)

out_txt = LOG_FOLDER / "convergence_by_iteration.txt"
out_txt.write_text(table + "\n", encoding="utf-8")
print(f"\nSaved table to {out_txt}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters  = [r["iteration"]  for r in records]
    costs  = [r["total_cost"] for r in records]
    p50s   = [r["wavg_p50"]   for r in records]
    p95s   = [r["wavg_p95"]   for r in records]

    # Number of panels: always 3 overall + 2 per-region panels if EMEA
    n_panels = 3 + (2 if EMEA_REGIONS else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4 * n_panels), sharex=True)
    fig.suptitle(
        f"Gradient-Descent Convergence by Iteration  [{_GEO} / {_BUCKET}]",
        fontsize=13, fontweight="bold",
    )

    # Panel 0: total cost
    axes[0].plot(iters, costs, marker="o", markersize=3, color="steelblue", linewidth=1.5)
    axes[0].set_ylabel("Total Cost ($/mo)")
    axes[0].set_title("Total Cost")
    axes[0].grid(True, alpha=0.3)

    # Panel 1: overall weighted P50
    axes[1].plot(iters, p50s, marker="o", markersize=3, color="darkorange", linewidth=1.5, label="All")
    axes[1].set_ylabel("Wtd-avg P50 (ms)")
    axes[1].set_title("Traffic-Weighted P50 Latency (overall)")
    axes[1].grid(True, alpha=0.3)

    # Panel 2: overall weighted P95
    axes[2].plot(iters, p95s, marker="o", markersize=3, color="seagreen", linewidth=1.5, label="All")
    axes[2].set_ylabel("Wtd-avg P95 (ms)")
    axes[2].set_title("Traffic-Weighted P95 Latency (overall)")
    axes[2].grid(True, alpha=0.3)

    if EMEA_REGIONS:
        _REGION_COLORS = {
            "Europe":       "royalblue",
            "Middle East":  "darkorange",
            "Africa":       "firebrick",
            "South Africa": "purple",
            "Other":        "gray",
        }
        _REGION_THRESHOLDS_P50 = {
            "Europe":       24.0,
            "Middle East":  45.0,
            "Africa":       75.0,
            "South Africa": 35.0,
        }
        _REGION_THRESHOLDS_P95 = {
            "Europe":       105.0,
            "Middle East":  180.0,
            "Africa":       220.0,
            "South Africa": 200.0,
        }

        # Panel 3: per-region P50
        ax_p50r = axes[3]
        for region in EMEA_REGIONS:
            vals = [r["region_p50"].get(region, 0.0) for r in records]
            color = _REGION_COLORS.get(region, "gray")
            ax_p50r.plot(iters, vals, marker="o", markersize=3, linewidth=1.5,
                         color=color, label=region)
            if region in _REGION_THRESHOLDS_P50:
                ax_p50r.axhline(_REGION_THRESHOLDS_P50[region], color=color,
                                linestyle="--", linewidth=0.8, alpha=0.6)
        ax_p50r.set_ylabel("Wtd-avg P50 (ms)")
        ax_p50r.set_title("Traffic-Weighted P50 Latency by Region  (dashed = target)")
        ax_p50r.legend(fontsize=8)
        ax_p50r.grid(True, alpha=0.3)

        # Panel 4: per-region P95
        ax_p95r = axes[4]
        for region in EMEA_REGIONS:
            vals = [r["region_p95"].get(region, 0.0) for r in records]
            color = _REGION_COLORS.get(region, "gray")
            ax_p95r.plot(iters, vals, marker="o", markersize=3, linewidth=1.5,
                         color=color, label=region)
            if region in _REGION_THRESHOLDS_P95:
                ax_p95r.axhline(_REGION_THRESHOLDS_P95[region], color=color,
                                linestyle="--", linewidth=0.8, alpha=0.6)
        ax_p95r.set_ylabel("Wtd-avg P95 (ms)")
        ax_p95r.set_title("Traffic-Weighted P95 Latency by Region  (dashed = target)")
        ax_p95r.legend(fontsize=8)
        ax_p95r.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    out_png = LOG_FOLDER / "convergence_by_iteration.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot   to {out_png}")
except ImportError:
    print("matplotlib not available — skipping plot.")
