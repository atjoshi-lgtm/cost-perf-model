"""
Plot TTFB CDF for 4 metros as a function of hitrate (0–100 step 10),
using only in-metro RTT and traffic. One figure per metro → PERF_ANALYSIS/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from solve_for_US import (
    parse_metro_areas,
    assign_parent_metros,
    get_metro_tiers,
    load_smoothed_fds_for_all_metros_threaded,
    load_mch_performance_models_threaded,
    ALL_METROS,
    MCH_METROS,
    _FDS2_DIR,
)
from analyse import (
    get_rtt_pdf,
    get_edge_tat_pdf,
    get_midgress_rtt_pdf,
    get_parent_tat_pdf,
    client_metro_ids,
)
from probability import Convolution
from perf_with_mch import MetroPerformanceWithMCH

# ── config ────────────────────────────────────────────────────────────────────
METROS_OF_INTEREST = ALL_METROS#["BOS", "IAH", "HNL", "SLC"]
HITRATES = list(range(0, 101, 20))   # 0, 20, 40, … 100
OUTPUT_DIR = BASE_DIR / "PERF_ANALYSIS"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── load shared data ──────────────────────────────────────────────────────────
print("Loading metro areas …")
airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(
    BASE_DIR / "PERF" / "metro_areas.csv"
)
parent_assignment = assign_parent_metros(airport_info)
metro_tiers = get_metro_tiers()

print("Loading footprint descriptors …")
fds_metros = []
for fd_path in _FDS2_DIR.glob("*.txt"):
    metro = fd_path.stem.upper()
    if metro in airport_info:
        fds_metros.append((metro, airport_info[metro]))

FDS_BY_METRO = load_smoothed_fds_for_all_metros_threaded(
    ALL_METROS,
    airport_info,
    metro_tiers,
    fds_metros,
)
print(f"Loaded {len(FDS_BY_METRO)} footprint descriptors")

print("Loading MCH performance models …")
MCH_PERFORMANCE_MODELS = load_mch_performance_models_threaded(MCH_METROS, airport_to_metro)
print(f"Loaded {len(MCH_PERFORMANCE_MODELS)} MCH models")

# ── helper ────────────────────────────────────────────────────────────────────

def build_perf_model(metro_code: str) -> MetroPerformanceWithMCH:
    """Build MetroPerformanceWithMCH using only the in-metro edge RTT."""
    metro_name = airport_to_metro[metro_code]
    parent_code = parent_assignment[metro_code][0]
    parent_model = MCH_PERFORMANCE_MODELS[parent_code]
    descriptor = FDS_BY_METRO[metro_code]

    model = MetroPerformanceWithMCH(
        name=metro_code,
        parent_model=parent_model,
        descriptor=descriptor,
    )
    model.set_edge_tat_hit(get_edge_tat_pdf(metro_name, cache_hit_type=1))
    model.set_edge_tat_miss(get_edge_tat_pdf(metro_name, cache_hit_type=0))
    model.set_mch_rtt(get_midgress_rtt_pdf(airport_to_metro[parent_code], metro_name))

    # In-metro RTT only: client metro == serving metro
    if metro_name in client_metro_ids:
        rtt_pdf = get_rtt_pdf(metro_name, client_metro_ids[metro_name])
        model.set_edge_rtt(metro_code, rtt_pdf)
    else:
        print(f"  WARNING: {metro_name} not in client_metro_ids")

    return model


def cdf_arrays(pdf):
    """Return (x_ms, cdf) numpy arrays for a ProbabilityDensityFunction."""
    s = pdf.probability_series.astype(float)
    s = s[s > 0].sort_index()
    total = float(s.sum())
    if total <= 0:
        return np.array([0.0]), np.array([0.0])
    return s.index.to_numpy(dtype=float), (s.cumsum() / total).values


# ── main loop ─────────────────────────────────────────────────────────────────
conn = Convolution()
colors = cm.viridis(np.linspace(0.05, 0.95, len(HITRATES)))

for metro_code in METROS_OF_INTEREST:
    if metro_code not in airport_to_metro:
        print(f"Skipping {metro_code}: not in metro_areas.csv"); continue
    if metro_code not in FDS_BY_METRO:
        print(f"Skipping {metro_code}: no footprint descriptor"); continue

    metro_name = airport_to_metro[metro_code]
    parent_code = parent_assignment[metro_code][0]
    print(f"\n{metro_code} ({metro_name}), parent={parent_code}")

    model = build_perf_model(metro_code)
    if metro_code not in model.edge_rtt:
        print(f"  Skipping: no in-metro edge RTT"); continue

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, hitrate in enumerate(HITRATES):
        try:
            ttfb_pdf = model.get_ttfb_pdf(
                from_metro=metro_code,
                hitrate=float(hitrate),
                conn=conn,
            )
            x, cdf = cdf_arrays(ttfb_pdf)
            p50 = ttfb_pdf.millisecond_at_percentile(50)
            p95 = ttfb_pdf.millisecond_at_percentile(95)
            ax.plot(x, cdf, color=colors[i], linewidth=1.8,
                    label=f"hit={hitrate}%  (p50={p50} ms, p95={p95} ms)")
        except Exception as e:
            print(f"  hitrate={hitrate}%: {e}")

    ax.set_xlabel("TTFB (ms)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(
        f"TTFB CDF vs Cache Hitrate — {metro_code} ({metro_name})\n"
        f"(in-metro traffic only, parent={parent_code})",
        fontsize=13,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")

    out_path = OUTPUT_DIR / f"ttfb_cdf_hitrate_{metro_code}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")

print("\nDone.")
