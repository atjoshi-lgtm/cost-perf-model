"""Compute traffic-weighted P50 and P95 latency from knee_analysis.txt.

For each of the three modes (Knee, Cost-Optimal), reports:
  - traffic-weighted P50 latency (ms)
  - traffic-weighted P95 latency (ms)
  - total cost ($/mo)

Zero-disk metros are included — traffic weights come from TRAFFIC_FROM
(same method as plot_gradient_convergence.py), so every metro with client
traffic is correctly weighted, regardless of whether it deploys disk.

Metros with no perf model (p50 = p95 = 0, e.g. TUS, ANC) are excluded.

Usage:
    python summarise_knee_analysis.py [<knee_analysis.txt path>]

Default: looks in US_traffic_based_penalty_less_aggressive/knee_analysis.txt
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the file
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent

if len(sys.argv) > 1:
    knee_path = Path(sys.argv[1])
    if not knee_path.is_absolute():
        knee_path = _BASE_DIR / knee_path
else:
    knee_path = _BASE_DIR / "US_traffic_based_penalty_less_aggressive" / "knee_analysis.txt"

if not knee_path.exists():
    sys.exit(f"File not found: {knee_path}")

# ---------------------------------------------------------------------------
# Load traffic weights via solve_for_US (mirrors plot_gradient_convergence.py)
# TRAFFIC_FROM[metro] = total client Mbps sent by that metro's users.
# Non-zero for every metro with users, including zero-disk metros.
# ---------------------------------------------------------------------------
import solve_for_US as _s

airport_info, metro_to_airport, airport_to_metro = _s.parse_metro_areas(
    _BASE_DIR / "PERF" / "metro_areas.csv"
)
traffic_lookup = _s._load_traffic_lookup()

neighborhood_from: dict[str, list[str]] = defaultdict(list)
for (asn_metro, bw_metro), traf in traffic_lookup.items():
    if traf > 10000 and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        neighborhood_from[metro_to_airport[bw_metro]].append(metro_to_airport[asn_metro])

traffic_lookup_by_airport: dict[tuple[str, str], float] = {}
for (asn_metro, bw_metro), traf in traffic_lookup.items():
    if asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        traffic_lookup_by_airport[
            (metro_to_airport[asn_metro], metro_to_airport[bw_metro])
        ] = traf

metros = list(airport_info.keys())
TRAFFIC_FROM: dict[str, float] = defaultdict(float)
for metro in metros:
    for from_metro in neighborhood_from.get(metro, []):
        t = traffic_lookup_by_airport.get((from_metro, metro), 0.0)
        TRAFFIC_FROM[from_metro] += t

# TRAFFIC[metro] = client Mbps weight
TRAFFIC: dict[str, float] = {m: TRAFFIC_FROM[m] for m in metros if TRAFFIC_FROM[m] > 0}

# ---------------------------------------------------------------------------
# Parse knee_analysis.txt  (disk values and p50/p95 only — traffic from above)
# ---------------------------------------------------------------------------
# Disk-metro and no-disk-metro lines share the same column layout:
#   parts[0]=metro  [4]=knee_cost  [5]=knee_p50  [6]=knee_p95
#            [10]=opt_cost  [11]=opt_p50  [12]=opt_p95

records: list[dict] = []

_in_nodisk = False
for raw in knee_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("-"):
        continue
    if "no local disk" in line:
        _in_nodisk = True
        continue
    if line.startswith("Metro") or line.startswith("Mbps"):
        continue
    if line.startswith("TOTAL"):
        break

    parts = line.split()
    if len(parts) < 13:
        continue

    metro     = parts[0]
    knee_p50  = float(parts[5])
    knee_p95  = float(parts[6])
    knee_cost = float(parts[4])
    opt_p50   = float(parts[11])
    opt_p95   = float(parts[12])
    opt_cost  = float(parts[10])

    # Skip metros with no perf model (both p50 and p95 are 0)
    if knee_p50 == 0.0 and knee_p95 == 0.0:
        continue

    # Use TRAFFIC_FROM weight — consistent with plot_gradient_convergence.py
    weight = TRAFFIC.get(metro, 0.0)
    if weight == 0.0:
        continue   # no client traffic at all — skip (e.g. TUS, ANC)

    records.append(dict(
        metro     = metro,
        weight    = weight,
        knee_p50  = knee_p50,
        knee_p95  = knee_p95,
        knee_cost = knee_cost,
        opt_p50   = opt_p50,
        opt_p95   = opt_p95,
        opt_cost  = opt_cost,
        no_disk   = _in_nodisk,
    ))

if not records:
    sys.exit("No data parsed from file.")

# ---------------------------------------------------------------------------
# Compute weighted averages
# ---------------------------------------------------------------------------
def wavg(values: list[float], weights: list[float]) -> float:
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w

weights = [r["weight"] for r in records]

wavg_knee_p50 = wavg([r["knee_p50"] for r in records], weights)
wavg_knee_p95 = wavg([r["knee_p95"] for r in records], weights)
wavg_opt_p50  = wavg([r["opt_p50"]  for r in records], weights)
wavg_opt_p95  = wavg([r["opt_p95"]  for r in records], weights)

total_knee_cost = sum(r["knee_cost"] for r in records)
total_opt_cost  = sum(r["opt_cost"]  for r in records)

n_disk   = sum(1 for r in records if not r["no_disk"])
n_nodisk = sum(1 for r in records if r["no_disk"])

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
SEP = "-" * 62
HDR = f"  {'Metric':<34}  {'Knee':>10}  {'Cost-Opt':>10}"

lines = [
    f"\nSource: {knee_path}",
    f"Metros: {len(records)} total  ({n_disk} disk-deploying, {n_nodisk} zero-disk)",
    SEP, HDR, SEP,
    f"  {'Total cost ($/mo)':<34}  {total_knee_cost:>10.2f}  {total_opt_cost:>10.2f}",
    f"  {'Wtd-avg P50 ms (all metros)':<34}  {wavg_knee_p50:>10.2f}  {wavg_opt_p50:>10.2f}",
    f"  {'Wtd-avg P95 ms (all metros)':<34}  {wavg_knee_p95:>10.2f}  {wavg_opt_p95:>10.2f}",
    SEP,
]

# Break out disk-only and no-disk-only weighted averages
disk_recs   = [r for r in records if not r["no_disk"]]
nodisk_recs = [r for r in records if r["no_disk"]]

if disk_recs:
    dw = [r["weight"] for r in disk_recs]
    lines += [
        f"  {'  Disk-deploying metros only:':<34}",
        f"  {'  Wtd-avg P50 ms':<34}  {wavg([r['knee_p50'] for r in disk_recs], dw):>10.2f}  {wavg([r['opt_p50'] for r in disk_recs], dw):>10.2f}",
        f"  {'  Wtd-avg P95 ms':<34}  {wavg([r['knee_p95'] for r in disk_recs], dw):>10.2f}  {wavg([r['opt_p95'] for r in disk_recs], dw):>10.2f}",
    ]

if nodisk_recs:
    nw = [r["weight"] for r in nodisk_recs]
    lines += [
        f"  {'  Zero-disk metros only:':<34}",
        f"  {'  Wtd-avg P50 ms':<34}  {wavg([r['knee_p50'] for r in nodisk_recs], nw):>10.2f}  {wavg([r['opt_p50'] for r in nodisk_recs], nw):>10.2f}",
        f"  {'  Wtd-avg P95 ms':<34}  {wavg([r['knee_p95'] for r in nodisk_recs], nw):>10.2f}  {wavg([r['opt_p95'] for r in nodisk_recs], nw):>10.2f}",
    ]

lines.append(SEP)
lines.append("  Note: weighted by TRAFFIC_FROM (client Mbps); no-perf-model metros excluded.")

output = "\n".join(lines)
print(output)

out_path = knee_path.parent / "knee_analysis_summary.txt"
out_path.write_text(output + "\n", encoding="utf-8")
print(f"\nSaved to {out_path}")
