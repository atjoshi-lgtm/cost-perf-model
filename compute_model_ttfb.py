"""Compute model-based TTFB distributions for 4 scenarios using RTT + TAT convolutions.

Scenarios:
  1. users=IAD (Washington_DC)  → edge=PHL (Philadelphia) → MCH=LGA (New_York),  hitrate=27.2%
  2. users=IAH (Houston)        → edge=DFW (Dallas)       → MCH=DFW (Dallas),     hitrate=25.0%
  3. users=SLC (Salt_Lake_City) → edge=DEN (Denver)       → MCH=DEN (Denver),     hitrate=22.9%
  4. users=IND (Indianapolis)   → edge=ORD (Chicago)      → MCH=ORD (Chicago),    hitrate=21.4%

Output: figs saved to check_convolution_hats_data/figs_model/
"""

from __future__ import annotations

import os
import sys
import sqlite3
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

sys.path.insert(0, BASE_DIR)

from probability import ProbabilityDensityFunction, PdfBucket, Convolution
from perf_with_mch import MetroPerformanceWithMCH, MCHPerformanceModel

# ---------------------------------------------------------------------------
# Metro name / ID mappings (from PERF/metro_areas.csv)
# ---------------------------------------------------------------------------
client_metro_ids: dict[str, int] = {}
with open(os.path.join(BASE_DIR, "PERF", "metro_areas.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        metro_id = int(row[0])
        metro_name = row[1]
        client_metro_ids[metro_name] = metro_id

# ---------------------------------------------------------------------------
# DB query helpers (mirrored from analyse.py)
# ---------------------------------------------------------------------------
_DB_PATH = os.path.join(BASE_DIR, "PERF", "perf_data.db")
_DATES = "'2026-02-07', '2026-02-08', '2026-02-09'"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH)


def get_rtt_pdf(metro_name: str, client_metro_id: int) -> ProbabilityDensityFunction:
    """Edge RTT from client metro ``client_metro_id`` to edge metro ``metro_name``."""
    import pandas as pd
    query = f"""
    SELECT * FROM netopt_perf_edge_rtt_ansabni
    WHERE region_metro = '{metro_name}' AND client_metro = '{client_metro_id}'
    AND pdate in ({_DATES})"""
    conn = _connect()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    if df.empty:
        print(f"  [WARN] No RTT data for edge={metro_name}, client_metro_id={client_metro_id}")
        return ProbabilityDensityFunction([])
    return ProbabilityDensityFunction.from_dataframe(df)


def get_edge_tat_pdf(metro_name: str, cache_hit_type: int = 1) -> ProbabilityDensityFunction:
    """Edge TAT (cache-hit type 1) at edge metro ``metro_name``."""
    import pandas as pd
    query = f"""
    SELECT * FROM netopt_perf_edge_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type = {cache_hit_type}
    AND pdate in ({_DATES})"""
    conn = _connect()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    if df.empty:
        print(f"  [WARN] No edge TAT data for edge={metro_name}")
        return ProbabilityDensityFunction([])
    return ProbabilityDensityFunction.from_dataframe(df)


def get_midgress_rtt_pdf(parent_metro: str, child_metro: str) -> ProbabilityDensityFunction:
    """Midgress RTT from child edge metro to parent MCH metro."""
    import pandas as pd
    query = f"""
    SELECT * FROM netopt_perf_midgress_rtt_ansabni
    WHERE parent_metro = '{parent_metro}' AND child_metro = '{child_metro}'
    AND pdate in ({_DATES})"""
    conn = _connect()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    if df.empty:
        print(f"  [WARN] No midgress RTT data for parent={parent_metro}, child={child_metro}")
        return ProbabilityDensityFunction([])
    return ProbabilityDensityFunction.from_dataframe(df)


def get_parent_tat_pdf(metro_name: str) -> ProbabilityDensityFunction:
    """Parent (MCH) TAT at metro ``metro_name`` (cache_hit_type != 2)."""
    import pandas as pd
    query = f"""
    SELECT * FROM netopt_perf_midgress_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type != 2
    AND pdate in ({_DATES})"""
    conn = _connect()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    if df.empty:
        print(f"  [WARN] No parent TAT data for MCH={metro_name}")
        return ProbabilityDensityFunction([])
    return ProbabilityDensityFunction.from_dataframe(df)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------
# Each entry: (label, user_airport, user_metro_name, edge_airport, edge_metro_name, mch_airport, mch_metro_name, hitrate)
# For scenario 1: user specified EWR as MCH but EWR is not an MCH metro — LGA (New_York) is the
#                 New York area MCH, so we use LGA/New_York.
SCENARIOS = [
    {
        "label":          "IAD→PHL→LGA",
        "display":        "Users: IAD (Washington DC)  →  Edge: PHL (Philadelphia)  →  MCH: LGA (New York)",
        "user_airport":   "IAD",
        "user_metro":     "Washington_DC",
        "edge_airport":   "PHL",
        "edge_metro":     "Philadelphia",
        "mch_airport":    "LGA",
        "mch_metro":      "New_York",
        "hitrate":        27.2,
    },
    {
        "label":          "IAH→DFW→DFW",
        "display":        "Users: IAH (Houston)  →  Edge: DFW (Dallas)  →  MCH: DFW (Dallas)",
        "user_airport":   "IAH",
        "user_metro":     "Houston",
        "edge_airport":   "DFW",
        "edge_metro":     "Dallas",
        "mch_airport":    "DFW",
        "mch_metro":      "Dallas",
        "hitrate":        25.0,
    },
    {
        "label":          "SLC→DEN→DEN",
        "display":        "Users: SLC (Salt Lake City)  →  Edge: DEN (Denver)  →  MCH: DEN (Denver)",
        "user_airport":   "SLC",
        "user_metro":     "Salt_Lake_City",
        "edge_airport":   "DEN",
        "edge_metro":     "Denver",
        "mch_airport":    "DEN",
        "mch_metro":      "Denver",
        "hitrate":        22.9,
    },
    {
        "label":          "IND→ORD→ORD",
        "display":        "Users: IND (Indianapolis)  →  Edge: ORD (Chicago)  →  MCH: ORD (Chicago)",
        "user_airport":   "IND",
        "user_metro":     "Indianapolis",
        "edge_airport":   "ORD",
        "edge_metro":     "Chicago",
        "mch_airport":    "ORD",
        "mch_metro":      "Chicago",
        "hitrate":        21.4,
    },
]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(BASE_DIR, "check_convolution_hats_data", "figs_model")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
conv = Convolution()

for sc in SCENARIOS:
    label       = sc["label"]
    display     = sc["display"]
    user_ap     = sc["user_airport"]
    user_name   = sc["user_metro"]
    edge_ap     = sc["edge_airport"]
    edge_name   = sc["edge_metro"]
    mch_ap      = sc["mch_airport"]
    mch_name    = sc["mch_metro"]
    hitrate     = sc["hitrate"]

    print(f"\n=== {label} (hitrate={hitrate}%) ===")

    # 1. Build MCH performance model
    print(f"  Loading MCH TAT for {mch_name}...")
    mch_tat = get_parent_tat_pdf(mch_name)
    mch_model = MCHPerformanceModel(name=mch_ap)
    mch_model.set_mch_tat(mch_tat)

    # 2. Build edge performance model
    #    descriptor not used for TTFB computation — pass None
    edge_model = MetroPerformanceWithMCH(
        name=edge_ap,
        parent_model=mch_model,
        descriptor=None,
    )

    # Edge TAT (cache hit)
    print(f"  Loading edge TAT for {edge_name}...")
    edge_tat = get_edge_tat_pdf(edge_name, cache_hit_type=1)
    edge_model.set_edge_tat_hit(edge_tat)

    # Midgress RTT: edge → MCH
    print(f"  Loading midgress RTT {mch_name} ← {edge_name}...")
    mch_rtt = get_midgress_rtt_pdf(mch_name, edge_name)
    edge_model.set_mch_rtt(mch_rtt)

    # Edge RTT: user metro → edge metro
    user_metro_id = client_metro_ids.get(user_name)
    if user_metro_id is None:
        print(f"  [WARN] user metro '{user_name}' not found in client_metro_ids — skipping scenario")
        continue
    print(f"  Loading edge RTT for {edge_name} ← {user_name} (id={user_metro_id})...")
    edge_rtt = get_rtt_pdf(edge_name, user_metro_id)
    edge_model.set_edge_rtt(user_ap, edge_rtt)

    # 3. Compute TTFB PDF
    print(f"  Computing TTFB PDF...")
    try:
        ttfb_pdf = edge_model.get_ttfb_pdf(from_metro=user_ap, hitrate=hitrate, conn=conv)
    except Exception as e:
        print(f"  [ERROR] Failed to compute TTFB: {e}")
        continue

    # 4. Extract CDF using millisecond-resolution probability_series
    series = ttfb_pdf.probability_series.astype(float)
    series = series[series > 0].sort_index()
    total = float(series.sum())
    if total <= 0:
        print(f"  [WARN] Zero total count in TTFB PDF for scenario {label}")
        continue

    cdf_x = series.index.to_numpy(dtype=float)
    cdf_y = (series.cumsum() / total).values

    # Compute percentiles
    p50 = ttfb_pdf.millisecond_at_percentile(50)
    p75 = ttfb_pdf.millisecond_at_percentile(75)
    p95 = ttfb_pdf.millisecond_at_percentile(95)
    p99 = ttfb_pdf.millisecond_at_percentile(99)

    print(f"  p50={p50:.1f}ms  p75={p75:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")

    # 5. Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cdf_x, cdf_y * 100, linewidth=2, color="steelblue", label="Model TTFB CDF")

    # Mark percentiles
    for pval, pname, color in [(p50, "p50", "green"), (p75, "p75", "orange"), (p95, "p95", "red"), (p99, "p99", "darkred")]:
        ax.axvline(pval, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(pval + 1, 5 + ({"p50": 0, "p75": 8, "p95": 16, "p99": 24}[pname]),
                f"{pname}={pval:.0f}ms", color=color, fontsize=8)

    ax.set_xlabel("TTFB (ms)")
    ax.set_ylabel("CDF (%)")
    ax.set_title(f"Model TTFB CDF\n{display}\nHitrate={hitrate}%", fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.4)

    # Annotate component source note
    note_lines = [
        f"Edge RTT:    {user_name} → {edge_name}",
        f"Edge TAT:    {edge_name} (cache hit)",
        f"MCH RTT:     {edge_name} → {mch_name}",
        f"MCH TAT:     {mch_name}",
        f"Hitrate:     {hitrate}%",
    ]
    ax.text(
        0.98, 0.35, "\n".join(note_lines),
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"model_ttfb_{label.replace('→', '_to_')}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")

print("\nDone.")
