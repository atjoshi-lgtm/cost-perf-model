"""Parse the last iteration from a gradient_descent_metro_log.txt, extract disk
values per metro, then re-evaluate cost, p50, p95 and perf_penalty using the
current cost model and performance models.

Usage:
    python analyse_last_iteration.py [<folder>]

Default folder: US_traffic_based_penalty
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve the log folder from the command line (or default)
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
folder_arg = sys.argv[1] if len(sys.argv) > 1 else "US_traffic_based_penalty_more_aggressive"
LOG_FOLDER = _BASE_DIR / folder_arg
LOG_PATH   = LOG_FOLDER / "gradient_descent_metro_log.txt"

if not LOG_PATH.exists():
    sys.exit(f"Log file not found: {LOG_PATH}")

# ---------------------------------------------------------------------------
# Parse the last iteration block from the log
# ---------------------------------------------------------------------------
# Each iteration block starts with "iteration=<N>" and ends just before the
# next "iteration=" line or EOF.

_ITER_RE  = re.compile(r"^iteration=(\d+)")
_METRO_RE = re.compile(
    r"^(\w+),\s*disk_mb=([\d.]+),\s*hitrate=([\d.]+),\s*cost=([\d.]+),"
    r"\s*perf_penalty=([\d.]+),\s*p50=([\d.]+),\s*p95=([\d.]+)"
)

def parse_last_iteration(log_path: Path) -> dict[str, float]:
    """Return {metro: disk_mb} from the last iteration block."""
    blocks: dict[int, dict[str, float]] = {}
    current_iter: int | None = None

    with log_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            m = _ITER_RE.match(line)
            if m:
                current_iter = int(m.group(1))
                blocks[current_iter] = {}
                continue
            m2 = _METRO_RE.match(line)
            if m2 and current_iter is not None:
                metro = m2.group(1)
                blocks[current_iter][metro] = float(m2.group(2))

    if not blocks:
        sys.exit("No iteration data found in log file.")

    last_iter = max(blocks)
    print(f"Last iteration in log: {last_iter}  ({len(blocks[last_iter])} metros)")
    return blocks[last_iter]

DISK_FROM_LOG: dict[str, float] = parse_last_iteration(LOG_PATH)

# ---------------------------------------------------------------------------
# Bootstrap shared infrastructure (mirrors solve_for_US.__main__)
# ---------------------------------------------------------------------------
import solve_for_US as _s
from cost import CaribouCostCalculator
from probability import Convolution

airport_info, metro_to_airport, airport_to_metro = _s.parse_metro_areas(
    _BASE_DIR / "PERF" / "metro_areas.csv"
)
metro_tiers       = _s.get_metro_tiers()
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

fds_metros = []
for fd_path in (_BASE_DIR / "FDS2").glob("*.txt"):
    code = fd_path.stem.upper()
    if code in airport_info:
        fds_metros.append((code, airport_info[code]))

FDS_BY_METRO = _s.load_smoothed_fds_for_all_metros_threaded(
    metros, airport_info, metro_tiers, fds_metros
)

MCH_PERFORMANCE_MODELS = _s.load_mch_performance_models_threaded(
    _s.MCH_METROS, airport_to_metro
)
PERFORMANCE_MODELS = _s.load_performance_models_threaded(
    metros, FDS_BY_METRO, airport_to_metro, parent_assignment,
    neighborhood_from, MCH_PERFORMANCE_MODELS,
)

# ---------------------------------------------------------------------------
# Build disk_provisioned from the log (only for metros present in the log)
# ---------------------------------------------------------------------------
DISK_PROVISIONED: dict[str, float] = {}
for metro, disk_mb in DISK_FROM_LOG.items():
    if metro in FDS_BY_METRO:
        DISK_PROVISIONED[metro] = disk_mb

# Zero-disk metros: present in the metro list but not in the log.
# Their traffic is served by neighbours; they still have p50/p95/penalty.
NODISK_METROS: list[str] = [
    m for m in metros
    if m not in DISK_PROVISIONED
    and m in PERFORMANCE_MODELS
    and (INCOMING_TRAFFIC.get(m, 0.0) > 0 or TRAFFIC_FROM.get(m, 0.0) > 0)
]

# ---------------------------------------------------------------------------
# Re-evaluate cost, hitrate, p50, p95, perf_penalty
# ---------------------------------------------------------------------------
def _replication_factor(metro: str) -> int:
    name = airport_to_metro.get(metro)
    tier = metro_tiers.get(name, 2)
    return 5 if tier == 0 else (3 if tier == 1 else 2)

cost_model = CaribouCostCalculator()

TRY_HITRATES: dict[str, float] = {}
COST:         dict[str, float] = {}
for metro, disk in DISK_PROVISIONED.items():
    descriptor = FDS_BY_METRO[metro]
    TRY_HITRATES[metro] = descriptor.hitrate_for_cache(int(disk))
    rf = _replication_factor(metro)
    COST[metro] = _s.compute_replicated_total_cost_model_b(
        cost_model=cost_model,
        total_disk_required_tb=disk,
        hitrate_fraction=TRY_HITRATES[metro] / 100.0,
        incoming_traffic_mbps=INCOMING_TRAFFIC.get(metro, 0.0),
        replication_factor=rf,
        is_mch_in_metro=metro in _s.MCH_METROS,
    )

conn = Convolution()

# For performance computation we need hitrates for ALL metros (including
# zero-disk ones), so that compute_performance_for_metro can look them up.
ALL_HITRATES: dict[str, float] = dict(TRY_HITRATES)   # copy disk metros
for metro in NODISK_METROS:
    ALL_HITRATES[metro] = 0.0  # no local cache → 0% hitrate

PERF:         dict[str, tuple[float, float]] = {}
PERF_PENALTY: dict[str, float]              = {}

# Disk-deploying metros
for metro in DISK_PROVISIONED:
    if metro not in PERFORMANCE_MODELS:
        continue
    p50, p95 = _s.compute_performance_for_metro(
        metro, neighborhood_to, PERFORMANCE_MODELS,
        ALL_HITRATES, traffic_lookup_by_airport, conn,
    )
    PERF[metro] = (p50, p95)
    PERF_PENALTY[metro] = _s.penalty_function(
        p50, p95, TRAFFIC_FROM.get(metro, 0.0) / 1000.0
    )

# Zero-disk metros
for metro in NODISK_METROS:
    p50, p95 = _s.compute_performance_for_metro(
        metro, neighborhood_to, PERFORMANCE_MODELS,
        ALL_HITRATES, traffic_lookup_by_airport, conn,
    )
    PERF[metro] = (p50, p95)
    PERF_PENALTY[metro] = _s.penalty_function(
        p50, p95, TRAFFIC_FROM.get(metro, 0.0) / 1000.0
    )

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
MB = 1024 * 1024

HDR = (
    f"{'Metro':>5}  {'Traffic':>8}  "
    f"{'Disk(MB)':>10}  {'HR%':>6}  {'Cost$/mo':>10}  "
    f"{'p50 ms':>8}  {'p95 ms':>8}  {'PerfPenalty':>12}"
)
SEP = "-" * len(HDR)
lines = [SEP, HDR, SEP]

active = sorted(
    DISK_PROVISIONED.keys(),
    key=lambda m: INCOMING_TRAFFIC.get(m, 0.0),
    reverse=True,
)

for metro in active:
    disk_mb  = DISK_PROVISIONED[metro] / MB
    hr       = TRY_HITRATES.get(metro, 0.0)
    cost     = COST.get(metro, 0.0)
    p50, p95 = PERF.get(metro, (0.0, 0.0))
    penalty  = PERF_PENALTY.get(metro, 0.0)
    traffic  = INCOMING_TRAFFIC.get(metro, 0.0)
    lines.append(
        f"{metro:>5}  {traffic:>8.0f}  "
        f"{disk_mb:>10.1f}  {hr:>6.1f}  {cost:>10.2f}  "
        f"{p50:>8.2f}  {p95:>8.2f}  {penalty:>12.2f}"
    )

total_cost    = sum(COST.get(m, 0.0) for m in active)
total_penalty = sum(PERF_PENALTY.get(m, 0.0) for m in active)
lines.append(SEP)
lines.append(
    f"{'TOTAL':>5}  {sum(INCOMING_TRAFFIC.get(m,0) for m in active):>8.0f}  "
    f"{'':>10}  {'':>6}  {total_cost:>10.2f}  "
    f"{'':>8}  {'':>8}  {total_penalty:>12.2f}"
)
lines.append(SEP)

# Zero-disk metros
nodisk_sorted = sorted(NODISK_METROS, key=lambda m: TRAFFIC_FROM.get(m, 0.0), reverse=True)
if nodisk_sorted:
    lines.append("")
    lines.append(SEP)
    lines.append(f"  (no local disk — served by neighbours, Traffic column = client Mbps)")
    lines.append(SEP)
    lines.append(HDR)
    lines.append(SEP)
    for metro in nodisk_sorted:
        p50, p95 = PERF.get(metro, (0.0, 0.0))
        penalty  = PERF_PENALTY.get(metro, 0.0)
        traffic  = TRAFFIC_FROM.get(metro, 0.0)
        lines.append(
            f"{metro:>5}  {traffic:>8.0f}  "
            f"{'0':>10}  {'0.0':>6}  {'0.00':>10}  "
            f"{p50:>8.2f}  {p95:>8.2f}  {penalty:>12.2f}"
        )
    nodisk_penalty = sum(PERF_PENALTY.get(m, 0.0) for m in nodisk_sorted)
    lines.append(SEP)
    lines.append(
        f"{'TOTAL':>5}  {sum(TRAFFIC_FROM.get(m,0) for m in nodisk_sorted):>8.0f}  "
        f"{'':>10}  {'':>6}  {'0.00':>10}  "
        f"{'':>8}  {'':>8}  {nodisk_penalty:>12.2f}"
    )
    lines.append(SEP)
    total_penalty += nodisk_penalty

lines.append(f"\nObjective (cost + penalty) = {total_cost + total_penalty:.2f}")

output = "\n".join(lines)
print(output)

out_path = LOG_FOLDER / "last_iteration_analysis.txt"
out_path.write_text(output + "\n", encoding="utf-8")
print(f"\nSaved to {out_path}")
