"""solve.py — like solve_for_US.py but uses per-metro knee P50/P95 as penalty targets.

Instead of fixed regional latency thresholds, the penalty function uses the
P50 and P95 observed at the FDS knee point for each metro as its targets.
Everything else (cost model, gradient descent, logging) is identical to
solve_for_US.py.

Usage:
    python3.11 solve.py --geo EMEA --bucket OtherBigFoot --traffic-threshold 5000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent

def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo",               type=str,   default="NA")
    parser.add_argument("--bucket",            type=str,   default="AkamaiHD")
    parser.add_argument("--traffic-threshold", type=float, default=20000.0)
    return parser.parse_args()

args = _get_args()
_GEO              = args.geo
_BUCKET           = args.bucket
_TRAFFIC_THRESHOLD = args.traffic_threshold

# ---------------------------------------------------------------------------
# Bootstrap solve_for_US helpers (must set module globals BEFORE importing)
# ---------------------------------------------------------------------------
import solve_for_US as _s

_s._GEO     = _GEO
_s._BUCKET  = _BUCKET
_s._FDS2_DIR = _BASE_DIR / f"FDS_{_BUCKET}"

from solve_for_US import (
    parse_metro_areas, get_metro_tiers, assign_parent_metros,
    _load_traffic_lookup, get_nearest_metro,
    load_smoothed_fds_for_all_metros_threaded,
    load_mch_performance_models_threaded,
    load_performance_models_threaded,
    load_cost_optimal_points_threaded,
    load_gradients_threaded,
    evaluate_state,
    compute_performance_for_metro,
    compute_replicated_total_cost_model_b,
    replication_factor_for_metro,
    log_iteration_state,
    MCH_METROS, ALL_METROS,
)
from cost import CaribouCostCalculator
from probability import Convolution, weighted_pdf_sum

# ---------------------------------------------------------------------------
# Setup (identical to solve_for_US.__main__)
# ---------------------------------------------------------------------------
airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(
    _BASE_DIR / "PERF" / "metro_areas.csv", _GEO
)
metro_tiers       = get_metro_tiers()
parent_assignment = assign_parent_metros(airport_info)
traffic_lookup    = _load_traffic_lookup()

neighborhood_to   = defaultdict(list)
neighborhood_from = defaultdict(list)
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if traffic > _TRAFFIC_THRESHOLD and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
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

fds_metros = []
for fd_path in (_BASE_DIR / f"FDS_{_BUCKET}").glob("*.txt"):
    code = fd_path.stem.upper()
    if code in airport_info:
        fds_metros.append((code, airport_info[code]))
print(f"Metros with FDS files: {[m for m, _ in fds_metros]}")

FDS_BY_METRO = load_smoothed_fds_for_all_metros_threaded(
    ALL_METROS, airport_info, metro_tiers, fds_metros
)
MCH_PERFORMANCE_MODELS = load_mch_performance_models_threaded(MCH_METROS, airport_to_metro)
PERFORMANCE_MODELS = load_performance_models_threaded(
    ALL_METROS, FDS_BY_METRO, airport_to_metro, parent_assignment,
    neighborhood_from, MCH_PERFORMANCE_MODELS,
)
COST_OPTIMAL_POINTS = load_cost_optimal_points_threaded(
    metros, FDS_BY_METRO, INCOMING_TRAFFIC, metro_tiers, airport_to_metro, MCH_METROS,
)

# ---------------------------------------------------------------------------
# Load knee disk values and compute knee P50/P95 for every metro
# ---------------------------------------------------------------------------
_KNEE_DIR = _BASE_DIR / f"KneeData_{_BUCKET}"

def _load_knee_bytes(metro: str) -> float | None:
    path = _KNEE_DIR / f"{metro.lower()}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("knee") is None:
        return None
    return float(data["knee"])

def _knee_bytes_with_fallback(metro: str) -> float | None:
    kb = _load_knee_bytes(metro)
    if kb is not None:
        return kb
    visited = {metro}
    candidate = get_nearest_metro(airport_info, metro, fds_metros)
    while candidate and candidate not in visited:
        visited.add(candidate)
        kb = _load_knee_bytes(candidate)
        if kb is not None:
            print(f"  Knee fallback: {metro} → {candidate}")
            return kb
        candidate = get_nearest_metro(airport_info, candidate, fds_metros)
    return None

print("\nLoading knee disk values...")
KNEE_HITRATES: dict[str, float] = {}
for metro in metros:
    if metro not in FDS_BY_METRO:
        continue
    kb = _knee_bytes_with_fallback(metro)
    if kb is None:
        continue
    descriptor = FDS_BY_METRO[metro]
    min_cache = min(p.cache_space for p in descriptor._points_sorted_by_cache)
    max_cache = max(p.cache_space for p in descriptor._points_sorted_by_cache)
    knee_mb = max(min_cache, min(kb, max_cache))
    KNEE_HITRATES[metro] = min(max(descriptor.hitrate_for_cache(knee_mb), 0.0), 100.0)

print(f"Knee hitrates loaded for {len(KNEE_HITRATES)} metros.")

print("\nComputing knee P50/P95 for each metro (these become penalty targets)...")
conn_knee = Convolution()
# Per-metro knee P50/P95 targets: dict[metro] = (p50_ms, p95_ms)
KNEE_PERF_TARGETS: dict[str, tuple[float, float]] = {}

for metro in metros:
    if metro not in PERFORMANCE_MODELS:
        continue
    p50, p95 = compute_performance_for_metro(
        metro,
        neighborhood_to,
        PERFORMANCE_MODELS,
        KNEE_HITRATES,
        traffic_lookup_by_airport,
        conn_knee,
    )
    if p50 > 0.0 or p95 > 0.0:
        KNEE_PERF_TARGETS[metro] = (p50, p95)
        print(f"  {metro}: knee P50={p50:.2f} ms, knee P95={p95:.2f} ms")
    else:
        print(f"  {metro}: no perf data at knee — will use 0 targets (no penalty)")

# ---------------------------------------------------------------------------
# Custom penalty function using knee P50/P95 as targets
# ---------------------------------------------------------------------------
def penalty_function(p50: float, p95: float, traffic_gbps: float = 1.0, metro: str = "") -> float:
    """Penalty relative to the metro's own knee-point latency targets."""
    if metro in KNEE_PERF_TARGETS:
        p50_target, p95_target = KNEE_PERF_TARGETS[metro]
    else:
        # No knee data: no penalty (target = current value, or 0 if no perf)
        p50_target, p95_target = p50, p95
    return (
        2 * traffic_gbps * (max(p50 - p50_target, 0) ** 2)
        + 2 * traffic_gbps * max(p95 - p95_target, 0)
    )

# Patch into the solve_for_US module so that compute_perf_penalty_for_metro
# (which calls penalty_function by name in its own module scope) picks up ours.
_s.penalty_function = penalty_function

# Also patch evaluate_state's penalty calls by wrapping it:
_orig_evaluate_state = evaluate_state

def _patched_evaluate_state(
    metros_arg, disk_provisioned, descriptors, incoming_traffic,
    metro_tiers_arg, airport_to_metro_arg, mch_metros, neighborhood_to_arg,
    performance_models, traffic_lookup_by_airport_arg, traffic_from,
):
    try_hitrates, cost_by_metro, _old_penalty, performance_stats = _orig_evaluate_state(
        metros_arg, disk_provisioned, descriptors, incoming_traffic,
        metro_tiers_arg, airport_to_metro_arg, mch_metros, neighborhood_to_arg,
        performance_models, traffic_lookup_by_airport_arg, traffic_from,
    )
    # Recompute penalty using our knee-based penalty_function
    perf_penalty: dict[str, float] = {}
    for metro in metros_arg:
        p50, p95 = performance_stats.get(metro, (0.0, 0.0))
        traffic_gbps = traffic_from.get(metro, 0.0) / 1000.0
        perf_penalty[metro] = penalty_function(p50, p95, traffic_gbps, metro=metro)
    return try_hitrates, cost_by_metro, perf_penalty, performance_stats

# ---------------------------------------------------------------------------
# Starting point: cost-optimal disk allocation
# ---------------------------------------------------------------------------
print("\nCost-optimal starting point for each metro:")
DISK_PROVISIONED = {metro: float(COST_OPTIMAL_POINTS[metro]["disk"]) for metro in COST_OPTIMAL_POINTS}
TRY_HITRATES     = {metro: float(COST_OPTIMAL_POINTS[metro]["hitrate"]) for metro in COST_OPTIMAL_POINTS}
COST             = {metro: float(COST_OPTIMAL_POINTS[metro]["cost"]) for metro in COST_OPTIMAL_POINTS}

conn = Convolution()
PERF_PENALTY: dict[str, float] = defaultdict(float)
cost_optimal_rows = []
for metro in metros:
    if metro not in PERFORMANCE_MODELS:
        continue
    p50, p95 = compute_performance_for_metro(
        metro, neighborhood_to, PERFORMANCE_MODELS, TRY_HITRATES,
        traffic_lookup_by_airport, conn,
    )
    PERF_PENALTY[metro] = penalty_function(p50, p95, TRAFFIC_FROM[metro] / 1000.0, metro=metro)
    print(
        f"  {metro}: P50={p50:.2f} ms  P95={p95:.2f} ms  penalty={PERF_PENALTY[metro]:.4f}"
        + (f"  [knee target P50={KNEE_PERF_TARGETS[metro][0]:.2f} P95={KNEE_PERF_TARGETS[metro][1]:.2f}]"
           if metro in KNEE_PERF_TARGETS else "")
    )
    cost_optimal_rows.append({
        "metro": metro, "disk_mb": DISK_PROVISIONED.get(metro, 0.0),
        "hitrate": TRY_HITRATES.get(metro, 0.0), "cost": COST.get(metro, 0.0),
        "p50_ms": p50, "p95_ms": p95, "perf_penalty": PERF_PENALTY[metro],
    })

csv_path = _BASE_DIR / f"cost_optimal_starting_points_{_GEO}_{_BUCKET}_knee_targets.csv"
with open(csv_path, "w") as f:
    f.write("metro,disk_mb,hitrate,cost,p50_ms,p95_ms,perf_penalty,knee_p50_target,knee_p95_target\n")
    for row in cost_optimal_rows:
        kt = KNEE_PERF_TARGETS.get(row["metro"], (0.0, 0.0))
        f.write(
            f"{row['metro']},{row['disk_mb']:.2f},{row['hitrate']:.2f},"
            f"{row['cost']:.2f},{row['p50_ms']:.4f},{row['p95_ms']:.4f},"
            f"{row['perf_penalty']:.4f},{kt[0]:.4f},{kt[1]:.4f}\n"
        )
print(f"\nCost-optimal starting points written to {csv_path}")

# ---------------------------------------------------------------------------
# Gradient descent loop (identical to solve_for_US.py)
# ---------------------------------------------------------------------------
gradient_step    = 5 * 1_000_000   # 5 TB in MB — probe step
TB_IN_MB         = 1_000_000
PER_METRO_STEP_TB = 10

active_metros = [m for m in metros if m in FDS_BY_METRO and m in PERFORMANCE_MODELS]
per_metro_log_path = _BASE_DIR / f"gradient_descent_metro_log_knee_{_GEO}_{_BUCKET}.txt"
summary_log_path   = _BASE_DIR / f"gradient_descent_summary_log_knee_{_GEO}_{_BUCKET}.txt"

per_metro_log_path.write_text("", encoding="utf-8")
summary_log_path.write_text("", encoding="utf-8")

iteration = 0
while True:
    iteration += 1
    gradients = load_gradients_threaded(
        active_metros, gradient_step, FDS_BY_METRO, DISK_PROVISIONED,
        TRY_HITRATES, COST, PERF_PENALTY, INCOMING_TRAFFIC, metro_tiers,
        airport_to_metro, MCH_METROS, neighborhood_to, PERFORMANCE_MODELS,
        traffic_lookup_by_airport, TRAFFIC_FROM,
    )

    grad_values   = {m: gradients.get(m, {}).get("overall_gradient", 0.0) for m in active_metros}
    abs_grads     = {m: abs(g) for m, g in grad_values.items()}
    total_abs_grad = sum(abs_grads.values())

    num_negative = sum(1 for g in grad_values.values() if g < 0)
    budget_mb    = max(num_negative, 1) * PER_METRO_STEP_TB * TB_IN_MB

    updated_disk = dict(DISK_PROVISIONED)
    for metro in active_metros:
        g = grad_values[metro]
        if g == 0.0 or total_abs_grad == 0.0:
            continue
        descriptor = FDS_BY_METRO[metro]
        current_disk    = DISK_PROVISIONED.get(metro, 0.0)
        min_cache_space = min(p.cache_space for p in descriptor._points_sorted_by_cache)
        max_cache_space = max(p.cache_space for p in descriptor._points_sorted_by_cache)
        weight    = abs_grads[metro] / total_abs_grad
        delta_mb  = math.ceil(weight * budget_mb / TB_IN_MB) * TB_IN_MB
        if g < 0:
            updated_disk[metro] = min(current_disk + delta_mb, max_cache_space)
        else:
            updated_disk[metro] = max(current_disk - delta_mb, min_cache_space)

    DISK_PROVISIONED = updated_disk
    TRY_HITRATES, COST, PERF_PENALTY, performance_stats = _patched_evaluate_state(
        active_metros, DISK_PROVISIONED, FDS_BY_METRO, INCOMING_TRAFFIC,
        metro_tiers, airport_to_metro, MCH_METROS, neighborhood_to,
        PERFORMANCE_MODELS, traffic_lookup_by_airport, TRAFFIC_FROM,
    )
    HITRATES = dict(TRY_HITRATES)

    total_cost    = sum(COST.get(m, 0.0) for m in active_metros)
    total_penalty = sum(PERF_PENALTY.get(m, 0.0) for m in active_metros)
    combined      = total_cost + total_penalty

    log_iteration_state(
        iteration, active_metros, DISK_PROVISIONED, HITRATES, COST,
        PERF_PENALTY, performance_stats, gradients,
        total_cost, total_penalty, combined,
        per_metro_log_path, summary_log_path,
    )

    print(f"\nIteration {iteration}")
    for metro in active_metros:
        p50, p95 = performance_stats.get(metro, (0.0, 0.0))
        g = gradients.get(metro, {})
        print(
            f"  {metro}: disk={DISK_PROVISIONED.get(metro,0):.0f} MB  "
            f"hr={HITRATES.get(metro,0):.1f}%  cost={COST.get(metro,0):.2f}  "
            f"penalty={PERF_PENALTY.get(metro,0):.4f}  "
            f"P50={p50:.2f} ms  P95={p95:.2f} ms  "
            f"grad={g.get('overall_gradient',0):.4f}"
        )
    print(
        f"  → total_cost={total_cost:.2f}  total_penalty={total_penalty:.4f}  "
        f"objective={combined:.4f}"
    )
