"""Evaluate cost and performance of every US metro operating at its FDS knee point.

For metros that don't have a knee file the nearest-metro FDS fallback (same
logic as solve_for_US.py) is used and the knee of the proxy metro is applied.

Output: a formatted table printed to stdout and saved to knee_analysis.txt.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cost import CaribouCostCalculator
from fds import FootprintDescriptor
from perf_with_mch import MCHPerformanceModel, MetroPerformanceWithMCH
from probability import Convolution, weighted_pdf_sum
from analyse import get_rtt_pdf, get_edge_tat_pdf, get_midgress_rtt_pdf, get_parent_tat_pdf, client_metro_ids

# Re-use helpers from solve_for_US without re-running its __main__ block
import solve_for_US as _s

_BASE_DIR = Path(__file__).resolve().parent
_KNEE_DIR = _BASE_DIR / "KneeData"

# ---------------------------------------------------------------------------
# Load knee data (bytes) for each FDS metro
# ---------------------------------------------------------------------------

def load_knee_bytes(metro: str) -> float | None:
    """Return the knee cache size in bytes for *metro*, or None if unavailable."""
    path = _KNEE_DIR / f"{metro.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data["knee"])


# ---------------------------------------------------------------------------
# Bootstrap shared infrastructure (mirrors solve_for_US.__main__)
# ---------------------------------------------------------------------------

airport_info, metro_to_airport, airport_to_metro = _s.parse_metro_areas(
    _BASE_DIR / "PERF" / "metro_areas.csv"
)
metro_tiers   = _s.get_metro_tiers()
parent_assignment = _s.assign_parent_metros(airport_info)

traffic_lookup = _s._load_traffic_lookup()

neighborhood_to   = defaultdict(list)
neighborhood_from = defaultdict(list)
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if traffic > 10000 and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        neighborhood_to[metro_to_airport[asn_metro]].append(metro_to_airport[bw_metro])
        neighborhood_from[metro_to_airport[bw_metro]].append(metro_to_airport[asn_metro])

traffic_lookup_by_airport: dict[tuple[str, str], float] = {}
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        traffic_lookup_by_airport[
            (metro_to_airport[asn_metro], metro_to_airport[bw_metro])
        ] = traffic

metros = list(airport_info.keys())

INCOMING_TRAFFIC: dict[str, float] = defaultdict(float)
TRAFFIC_FROM:     dict[str, float] = defaultdict(float)
for metro in metros:
    for from_metro in neighborhood_from.get(metro, []):
        t = traffic_lookup_by_airport.get((from_metro, metro), 0.0)
        INCOMING_TRAFFIC[metro] += t
        TRAFFIC_FROM[from_metro] += t

# FDS descriptors
fds_metros = []
for fd_path in (_BASE_DIR / "FDS2").glob("*.txt"):
    code = fd_path.stem.upper()
    if code in airport_info:
        fds_metros.append((code, airport_info[code]))

FDS_BY_METRO = _s.load_smoothed_fds_for_all_metros_threaded(
    metros, airport_info, metro_tiers, fds_metros
)

# MCH and edge performance models
MCH_PERFORMANCE_MODELS = _s.load_mch_performance_models_threaded(
    _s.MCH_METROS, airport_to_metro
)
PERFORMANCE_MODELS = _s.load_performance_models_threaded(
    metros,
    FDS_BY_METRO,
    airport_to_metro,
    parent_assignment,
    neighborhood_from,
    MCH_PERFORMANCE_MODELS,
)

# Cost-optimal points (for comparison)
COST_OPTIMAL_POINTS = _s.load_cost_optimal_points_threaded(
    metros,
    FDS_BY_METRO,
    INCOMING_TRAFFIC,
    metro_tiers,
    airport_to_metro,
    _s.MCH_METROS,
)

# ---------------------------------------------------------------------------
# Determine knee disk for every metro
# ---------------------------------------------------------------------------
# Knee files exist for FDS metros.  For other metros the same nearest-FDS
# fallback that solve_for_US uses for the FDS is applied: walk fds_metros
# sorted by distance.

def _knee_bytes_with_fallback(metro: str) -> tuple[float | None, str]:
    """Return (knee_bytes, source_metro) using the nearest-metro fallback."""
    kb = load_knee_bytes(metro)
    if kb is not None:
        return kb, metro

    # Walk neighbours the same way get_smoothed_fd_for_metro does
    visited = {metro}
    candidate = _s.get_nearest_metro(airport_info, metro, fds_metros)
    while candidate and candidate not in visited:
        visited.add(candidate)
        kb = load_knee_bytes(candidate)
        if kb is not None:
            return kb, candidate
        candidate = _s.get_nearest_metro(airport_info, candidate, fds_metros)
    return None, metro


KNEE_DISK_BYTES: dict[str, float] = {}
KNEE_SOURCE:     dict[str, str]   = {}

for metro in metros:
    if metro not in FDS_BY_METRO:
        continue
    if INCOMING_TRAFFIC.get(metro, 0.0) == 0:
        continue  # No traffic → no disk, same as cost-optimal
    kb, src = _knee_bytes_with_fallback(metro)
    if kb is None:
        continue
    descriptor = FDS_BY_METRO[metro]
    # Clamp to the range of the FDS
    min_cache = min(p.cache_space for p in descriptor._points_sorted_by_cache)
    max_cache = max(p.cache_space for p in descriptor._points_sorted_by_cache)
    KNEE_DISK_BYTES[metro] = max(min_cache, min(kb, max_cache))
    KNEE_SOURCE[metro] = src

# ---------------------------------------------------------------------------
# Evaluate cost & performance at the knee
# ---------------------------------------------------------------------------

cost_model = CaribouCostCalculator()

def _replication_factor(metro: str) -> int:
    name = airport_to_metro.get(metro)
    tier = metro_tiers.get(name, 2)
    return 5 if tier == 0 else (3 if tier == 1 else 2)


# Hitrates at the knee
TRY_HITRATES: dict[str, float] = {}
for metro, disk_bytes in KNEE_DISK_BYTES.items():
    TRY_HITRATES[metro] = FDS_BY_METRO[metro].hitrate_for_cache(int(disk_bytes))

# Costs at the knee
KNEE_COST: dict[str, float] = {}
for metro, disk_bytes in KNEE_DISK_BYTES.items():
    rf = _replication_factor(metro)
    KNEE_COST[metro] = _s.compute_replicated_total_cost_model_b(
        cost_model=cost_model,
        total_disk_required_tb=disk_bytes,
        hitrate_fraction=TRY_HITRATES[metro] / 100.0,
        incoming_traffic_mbps=INCOMING_TRAFFIC.get(metro, 0.0),
        replication_factor=rf,
        is_mch_in_metro=metro in _s.MCH_METROS,
    )

# Performance at the knee
conn = Convolution()
KNEE_PERF: dict[str, tuple[float, float]] = {}
for metro in metros:
    if metro not in PERFORMANCE_MODELS:
        continue
    p50, p95 = _s.compute_performance_for_metro(
        metro,
        neighborhood_to,
        PERFORMANCE_MODELS,
        TRY_HITRATES,
        traffic_lookup_by_airport,
        conn,
    )
    KNEE_PERF[metro] = (p50, p95)

# Perf penalty at the knee
KNEE_PENALTY: dict[str, float] = {
    metro: _s.penalty_function(
        KNEE_PERF[metro][0], KNEE_PERF[metro][1],
        TRAFFIC_FROM.get(metro, 0.0) / 1000.0
    )
    for metro in KNEE_PERF
}

# Performance at the cost-optimal point
OPT_HITRATES: dict[str, float] = {
    metro: float(COST_OPTIMAL_POINTS[metro]["hitrate"])
    for metro in COST_OPTIMAL_POINTS
}
OPT_PERF: dict[str, tuple[float, float]] = {}
for metro in metros:
    if metro not in PERFORMANCE_MODELS:
        continue
    p50, p95 = _s.compute_performance_for_metro(
        metro,
        neighborhood_to,
        PERFORMANCE_MODELS,
        OPT_HITRATES,
        traffic_lookup_by_airport,
        conn,
    )
    OPT_PERF[metro] = (p50, p95)

# Perf penalty at the cost-optimal point
OPT_PENALTY: dict[str, float] = {
    metro: _s.penalty_function(
        OPT_PERF[metro][0], OPT_PERF[metro][1],
        TRAFFIC_FROM.get(metro, 0.0) / 1000.0
    )
    for metro in OPT_PERF
}

# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

MB  = 1024 * 1024     # for reference only (disk values in this file are already in MB)

rows = []
for metro in sorted(metros):
    if metro not in PERFORMANCE_MODELS:
        continue

    no_disk = INCOMING_TRAFFIC.get(metro, 0.0) == 0  # served by neighbours — no local disk

    knee_bytes  = KNEE_DISK_BYTES.get(metro, 0.0) if not no_disk else 0.0
    knee_mb     = knee_bytes          # already in MB
    knee_hr     = TRY_HITRATES.get(metro, 0.0) if not no_disk else 0.0
    knee_cost   = KNEE_COST.get(metro, 0.0)     if not no_disk else 0.0
    knee_p50, knee_p95 = KNEE_PERF.get(metro, (0.0, 0.0))
    knee_src    = KNEE_SOURCE.get(metro, "-") if not no_disk else "-"

    opt = COST_OPTIMAL_POINTS.get(metro, {})
    if no_disk:
        opt_mb, opt_hr, opt_cost = 0.0, 0.0, 0.0
    else:
        opt_mb   = opt.get("disk", 0.0)    # already in MB
        opt_hr   = opt.get("hitrate", 0.0)
        opt_cost = opt.get("cost", 0.0)
    opt_p50, opt_p95 = OPT_PERF.get(metro, (0.0, 0.0))

    # Only include the row if the metro has some traffic (client or server)
    if INCOMING_TRAFFIC.get(metro, 0.0) == 0 and TRAFFIC_FROM.get(metro, 0.0) == 0:
        continue

    delta_cost_pct = ((knee_cost - opt_cost) / opt_cost * 100) if opt_cost else 0.0

    rows.append({
        "metro":        metro,
        "knee_mb":      knee_mb,
        "knee_hr":      knee_hr,
        "knee_cost":    knee_cost,
        "knee_p50":     knee_p50,
        "knee_p95":     knee_p95,
        "knee_penalty": KNEE_PENALTY.get(metro, 0.0),
        "opt_mb":       opt_mb,
        "opt_hr":       opt_hr,
        "opt_cost":     opt_cost,
        "opt_p50":      opt_p50,
        "opt_p95":      opt_p95,
        "opt_penalty":  OPT_PENALTY.get(metro, 0.0),
        "delta_cost":   delta_cost_pct,
        "knee_src":     knee_src,
        "traffic_mbps": INCOMING_TRAFFIC.get(metro, 0.0),
        "traffic_from": TRAFFIC_FROM.get(metro, 0.0),
        "no_disk":      no_disk,
    })

# Sort: metros with disk first (by incoming traffic desc), then no-disk metros (by client traffic desc)
rows.sort(key=lambda r: (r["no_disk"], -r["traffic_mbps"] if not r["no_disk"] else -r["traffic_from"]))

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

HDR = (
    f"{'Metro':>5}  {'Traffic':>8}  "
    f"{'─────────────────── Knee ───────────────────':^70}  "
    f"{'──────────────── Cost-Optimal ────────────────':^70}  "
    f"{'ΔCost%':>7}  {'KneeSrc':>7}"
)
SEP = "-" * len(HDR)
COL = (
    f"{'':>5}  {'Mbps':>8}  "
    f"{'Disk(MB)':>10}  {'HR%':>6}  {'Cost$/mo':>10}  {'p50 ms':>8}  {'p95 ms':>8}  {'Penalty':>10}  "
    f"{'Disk(MB)':>10}  {'HR%':>6}  {'Cost$/mo':>10}  {'p50 ms':>8}  {'p95 ms':>8}  {'Penalty':>10}  "
    f"{'%':>7}  {'':>7}"
)

lines = [SEP, HDR, COL, SEP]

prev_no_disk = False
for r in rows:
    # Insert a separator when switching from disk-deploying to no-disk metros
    if r["no_disk"] and not prev_no_disk:
        lines.append(SEP)
        lines.append(f"  (no local disk — traffic served by neighbouring metros, Traffic column = client Mbps)")
        lines.append(SEP)
    prev_no_disk = r["no_disk"]

    # For no-disk metros show their client traffic in the Traffic column
    traffic_display = r["traffic_from"] if r["no_disk"] else r["traffic_mbps"]
    line = (
        f"{r['metro']:>5}  {traffic_display:>8.0f}  "
        f"{r['knee_mb']:>10.0f}  {r['knee_hr']:>6.1f}  {r['knee_cost']:>10.2f}  "
        f"{r['knee_p50']:>8.1f}  {r['knee_p95']:>8.1f}  {r['knee_penalty']:>10.2f}  "
        f"{r['opt_mb']:>10.0f}  {r['opt_hr']:>6.1f}  {r['opt_cost']:>10.2f}  "
        f"{r['opt_p50']:>8.1f}  {r['opt_p95']:>8.1f}  {r['opt_penalty']:>10.2f}  "
        f"{r['delta_cost']:>+7.1f}  {r['knee_src']:>7}"
    )
    lines.append(line)

# Totals
total_knee_cost    = sum(r["knee_cost"]    for r in rows)
total_knee_penalty = sum(r["knee_penalty"] for r in rows)
total_opt_cost     = sum(r["opt_cost"]     for r in rows)
total_opt_penalty  = sum(r["opt_penalty"]  for r in rows)
total_delta_pct = (total_knee_cost - total_opt_cost) / total_opt_cost * 100 if total_opt_cost else 0.0

lines.append(SEP)
lines.append(
    f"{'TOTAL':>5}  {sum(r['traffic_mbps'] for r in rows):>8.0f}  "
    f"{'':>10}  {'':>6}  {total_knee_cost:>10.2f}  "
    f"{'':>8}  {'':>8}  {total_knee_penalty:>10.2f}  "
    f"{'':>10}  {'':>6}  {total_opt_cost:>10.2f}  "
    f"{'':>8}  {'':>8}  {total_opt_penalty:>10.2f}  "
    f"{total_delta_pct:>+7.1f}  {'':>7}"
)
lines.append(SEP)

output = "\n".join(lines)
print(output)

out_path = _BASE_DIR / "knee_analysis.txt"
out_path.write_text(output + "\n", encoding="utf-8")
print(f"\nSaved to {out_path}")
