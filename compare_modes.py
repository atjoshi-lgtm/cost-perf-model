"""Compare three operating modes side-by-side for every US metro.

Sources
-------
* knee_analysis.txt           – Knee point  AND  Cost-Optimal point
* last_iteration_analysis.txt – Gradient-descent last iteration (all metros,
                                including zero-disk ones with p50/p95)

Disk units
----------
knee_analysis   : Disk(MB) column is in MB  (large numbers like 206471000 MB ≈ 197 GB)
last_iteration  : Disk(MB) column is in GB  (e.g. 220.3 GB)

All disk values are normalised to GB in the output.
Averages are traffic-weighted (by client Mbps).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate files – check next to this script, then in the less-aggressive folder
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [_HERE, _HERE / "US_traffic_based_penalty_more_aggressive"]

def _find(name: str) -> Path:
    for d in _CANDIDATES:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {name} in {_CANDIDATES}")

KNEE_PATH = _find("knee_analysis.txt")
ITER_PATH = _find("last_iteration_analysis.txt")

# ---------------------------------------------------------------------------
# Parse knee_analysis.txt
# ---------------------------------------------------------------------------
# Disk(MB) values from that file are in MB  →  divide by 1024 to get GB.
# No-disk metros have disk=0.

knee_rows:  dict[str, dict] = {}   # metro → {traffic, knee_disk_gb, knee_hr, knee_cost, knee_p50, knee_p95,
                                   #           opt_disk_gb, opt_hr, opt_cost, opt_p50, opt_p95, no_disk}

_knee_text = KNEE_PATH.read_text(encoding="utf-8")
_in_nodisk = False

for raw in _knee_text.splitlines():
    line = raw.strip()
    if not line or line.startswith("-") or line.startswith("Metro") or line.startswith("Mbps"):
        continue
    if "no local disk" in line:
        _in_nodisk = True
        continue
    if line.startswith("TOTAL"):
        break

    parts = line.split()
    metro = parts[0]

    if _in_nodisk:
        # Format: metro traffic 0 0.0 0.00 p50 p95 penalty 0 0.0 0.00 p50 p95 penalty delta src
        # idx:      0     1    2  3   4    5   6    7      8  9   10   11  12   13      14   15
        traffic = float(parts[1])
        p50_knee = float(parts[5])
        p95_knee = float(parts[6])
        p50_opt  = float(parts[11])
        p95_opt  = float(parts[12])
        knee_rows[metro] = dict(
            traffic=traffic, no_disk=True,
            knee_disk_gb=0.0, knee_hr=0.0, knee_cost=0.0,
            knee_p50=p50_knee, knee_p95=p95_knee,
            opt_disk_gb=0.0,  opt_hr=0.0,  opt_cost=0.0,
            opt_p50=p50_opt,  opt_p95=p95_opt,
        )
    else:
        # Format: metro traffic disk_mb hr cost p50 p95 penalty disk_mb hr cost p50 p95 penalty delta src
        traffic     = float(parts[1])
        knee_disk_mb= float(parts[2])
        knee_hr     = float(parts[3])
        knee_cost   = float(parts[4])
        knee_p50    = float(parts[5])
        knee_p95    = float(parts[6])
        opt_disk_mb = float(parts[8])
        opt_hr      = float(parts[9])
        opt_cost    = float(parts[10])
        opt_p50     = float(parts[11])
        opt_p95     = float(parts[12])
        knee_rows[metro] = dict(
            traffic=traffic, no_disk=False,
            knee_disk_gb=knee_disk_mb / 1024,
            knee_hr=knee_hr, knee_cost=knee_cost,
            knee_p50=knee_p50, knee_p95=knee_p95,
            opt_disk_gb=opt_disk_mb / 1024,
            opt_hr=opt_hr, opt_cost=opt_cost,
            opt_p50=opt_p50, opt_p95=opt_p95,
        )

# ---------------------------------------------------------------------------
# Parse last_iteration_analysis.txt
# ---------------------------------------------------------------------------
# Disk(MB) column is in GB for disk metros (e.g. 220.3 → 220.3 GB).
# Zero-disk metros have disk=0.0, cost=0.0, but real p50/p95.
# The file also has a "(no local disk …)" separator section — parse all rows.

iter_rows: dict[str, dict] = {}

_iter_text = ITER_PATH.read_text(encoding="utf-8")
for raw in _iter_text.splitlines():
    line = raw.strip()
    if (not line or line.startswith("-") or line.startswith("Metro")
            or line.startswith("TOTAL") or line.startswith("Objective")
            or line.startswith("(")):
        continue
    parts = line.split()
    if len(parts) < 7:
        continue
    metro = parts[0]
    iter_rows[metro] = dict(
        traffic = float(parts[1]),
        disk_gb = float(parts[2]),   # GB for disk metros, 0.0 for no-disk
        hr      = float(parts[3]),
        cost    = float(parts[4]),
        p50     = float(parts[5]),
        p95     = float(parts[6]),
    )

# Extract totals from the files
def _extract_total_cost(text: str, label: str) -> float:
    m = re.search(r"TOTAL\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)", text)
    return 0.0

# Reparse totals from TOTAL line
def _parse_knee_totals(text: str):
    for line in text.splitlines():
        if line.strip().startswith("TOTAL"):
            parts = line.split()
            # TOTAL traffic [blank] [blank] knee_cost [blank] [blank] knee_pen [blank] [blank] opt_cost ...
            # actual format: TOTAL  traffic  ''  ''  knee_cost  ''  ''  knee_pen  ''  ''  opt_cost  ''  ''  opt_pen  delta
            # easier to just sum from rows
            break

knee_total_cost  = sum(r["knee_cost"] for r in knee_rows.values())
opt_total_cost   = sum(r["opt_cost"]  for r in knee_rows.values())
iter_total_cost  = sum(r["cost"]      for r in iter_rows.values())

# ---------------------------------------------------------------------------
# All metros, ordered: disk-deploying by descending traffic, then no-disk
# ---------------------------------------------------------------------------

disk_metros   = sorted([m for m, r in knee_rows.items() if not r["no_disk"]],
                        key=lambda m: -knee_rows[m]["traffic"])
nodisk_metros = sorted([m for m, r in knee_rows.items() if r["no_disk"]],
                        key=lambda m: -knee_rows[m]["traffic"])
all_metros = disk_metros + nodisk_metros

# Traffic weight for each metro (client Mbps — the meaningful weighting unit)
def _weight(m: str) -> float:
    return knee_rows[m]["traffic"]   # INCOMING for disk metros, TRAFFIC_FROM for no-disk metros

total_weight = sum(_weight(m) for m in all_metros)

def _wavg_p50(mode_p50: dict[str, float]) -> float:
    return sum(_weight(m) * mode_p50[m] for m in mode_p50) / sum(_weight(m) for m in mode_p50)

def _wavg_p95(mode_p95: dict[str, float]) -> float:
    return sum(_weight(m) * mode_p95[m] for m in mode_p95) / sum(_weight(m) for m in mode_p95)

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

MB_LINE = "#" * 110
SEP     = "-" * 110

def hdr(title: str) -> str:
    return f"\n{MB_LINE}\n  {title}\n{MB_LINE}"

lines: list[str] = []

# ── DISK (GB) ───────────────────────────────────────────────────────────────
lines.append(hdr("DISK  (GB)"))
lines.append(f"{'Metro':>5}  {'Traffic':>8}  {'Gradient':>10}  {'Knee':>10}  {'Cost-Opt':>10}")
lines.append(SEP)
for m in disk_metros:
    r  = knee_rows[m]
    it = iter_rows.get(m)
    it_disk = f"{it['disk_gb']:>10.1f}" if it else f"{'—':>10}"
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {it_disk}  {r['knee_disk_gb']:>10.1f}  {r['opt_disk_gb']:>10.1f}"
    )
lines.append(SEP)
lines.append(f"  (no local disk — served by neighbours, Traffic = client Mbps)")
lines.append(SEP)
for m in nodisk_metros:
    r = knee_rows[m]
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {'0':>10}  {'0':>10}  {'0':>10}"
    )

# ── COST ($/mo) ─────────────────────────────────────────────────────────────
lines.append(hdr("COST  ($/mo)"))
lines.append(f"{'Metro':>5}  {'Traffic':>8}  {'Gradient':>10}  {'Knee':>10}  {'Cost-Opt':>10}")
lines.append(SEP)
for m in disk_metros:
    r  = knee_rows[m]
    it = iter_rows.get(m)
    it_cost = f"{it['cost']:>10.2f}" if it else f"{'—':>10}"
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {it_cost}  {r['knee_cost']:>10.2f}  {r['opt_cost']:>10.2f}"
    )
lines.append(SEP)
lines.append(
    f"{'TOTAL':>5}  {'':>8}  {iter_total_cost:>10.2f}  {knee_total_cost:>10.2f}  {opt_total_cost:>10.2f}"
)
lines.append(f"  (no-disk metros have zero cost in all modes)")

# ── P50 (ms) ────────────────────────────────────────────────────────────────
lines.append(hdr("P50  (ms)"))
lines.append(f"{'Metro':>5}  {'Traffic':>8}  {'Gradient':>10}  {'Knee':>10}  {'Cost-Opt':>10}")
lines.append(SEP)
for m in disk_metros:
    r  = knee_rows[m]
    it = iter_rows.get(m)
    it_p50 = f"{it['p50']:>10.2f}" if it else f"{'—':>10}"
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {it_p50}  {r['knee_p50']:>10.1f}  {r['opt_p50']:>10.1f}"
    )
lines.append(SEP)
lines.append(f"  (no local disk — served by neighbours, Traffic = client Mbps)")
lines.append(SEP)
for m in nodisk_metros:
    r  = knee_rows[m]
    it = iter_rows.get(m)
    it_p50 = f"{it['p50']:>10.2f}" if it else f"{'—':>10}"
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {it_p50}  {r['knee_p50']:>10.1f}  {r['opt_p50']:>10.1f}"
    )

# Traffic-weighted averages — build per-mode p50 dicts over all metros
iter_p50  = {m: iter_rows[m]["p50"]       for m in all_metros if m in iter_rows}
knee_p50  = {m: knee_rows[m]["knee_p50"]  for m in all_metros}
opt_p50   = {m: knee_rows[m]["opt_p50"]   for m in all_metros}

wavg_iter_p50 = _wavg_p50(iter_p50)
wavg_knee_p50 = _wavg_p50(knee_p50)
wavg_opt_p50  = _wavg_p50(opt_p50)

lines.append(SEP)
lines.append(
    f"{'WAVG':>5}  {'all':>8}  {wavg_iter_p50:>10.2f}  {wavg_knee_p50:>10.2f}  {wavg_opt_p50:>10.2f}"
)
lines.append(f"  (traffic-weighted average across all {len(all_metros)} metros)")

# ── P95 (ms) ────────────────────────────────────────────────────────────────
lines.append(hdr("P95  (ms)"))
lines.append(f"{'Metro':>5}  {'Traffic':>8}  {'Gradient':>10}  {'Knee':>10}  {'Cost-Opt':>10}")
lines.append(SEP)
for m in disk_metros:
    r  = knee_rows[m]
    it = iter_rows.get(m)
    it_p95 = f"{it['p95']:>10.2f}" if it else f"{'—':>10}"
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {it_p95}  {r['knee_p95']:>10.1f}  {r['opt_p95']:>10.1f}"
    )
lines.append(SEP)
lines.append(f"  (no local disk — served by neighbours, Traffic = client Mbps)")
lines.append(SEP)
for m in nodisk_metros:
    r  = knee_rows[m]
    it = iter_rows.get(m)
    it_p95 = f"{it['p95']:>10.2f}" if it else f"{'—':>10}"
    lines.append(
        f"{m:>5}  {r['traffic']:>8.0f}  {it_p95}  {r['knee_p95']:>10.1f}  {r['opt_p95']:>10.1f}"
    )

iter_p95  = {m: iter_rows[m]["p95"]       for m in all_metros if m in iter_rows}
knee_p95  = {m: knee_rows[m]["knee_p95"]  for m in all_metros}
opt_p95   = {m: knee_rows[m]["opt_p95"]   for m in all_metros}

wavg_iter_p95 = _wavg_p95(iter_p95)
wavg_knee_p95 = _wavg_p95(knee_p95)
wavg_opt_p95  = _wavg_p95(opt_p95)

lines.append(SEP)
lines.append(
    f"{'WAVG':>5}  {'all':>8}  {wavg_iter_p95:>10.2f}  {wavg_knee_p95:>10.2f}  {wavg_opt_p95:>10.2f}"
)
lines.append(f"  (traffic-weighted average across all {len(all_metros)} metros)")

# ── SUMMARY ─────────────────────────────────────────────────────────────────
lines.append(hdr("SUMMARY"))
lines.append(f"  {'Metric':<36}  {'Gradient':>12}  {'Knee':>12}  {'Cost-Opt':>12}")
lines.append(SEP)
lines.append(f"  {'Total cost ($/mo)':<36}  {iter_total_cost:>12.2f}  {knee_total_cost:>12.2f}  {opt_total_cost:>12.2f}")
lines.append(f"  {'Wtd-avg P50 ms (all metros)':<36}  {wavg_iter_p50:>12.2f}  {wavg_knee_p50:>12.2f}  {wavg_opt_p50:>12.2f}")
lines.append(f"  {'Wtd-avg P95 ms (all metros)':<36}  {wavg_iter_p95:>12.2f}  {wavg_knee_p95:>12.2f}  {wavg_opt_p95:>12.2f}")
lines.append(MB_LINE)
lines.append(f"  Notes:")
lines.append(f"   • Gradient = last iteration of gradient-descent ({len(iter_rows)} metros total)")
lines.append(f"   • Knee     = operate every metro at its FDS knee point")
lines.append(f"   • Cost-Opt = cost-minimising point given the traffic-weighted penalty objective")
lines.append(f"   • No-disk metros ({len(nodisk_metros)}) have 0 disk/cost; p50/p95 from neighbour-served model")
lines.append(f"   • Weighted average uses client Mbps as weight for each metro")
lines.append(MB_LINE)

output = "\n".join(lines)
print(output)

out_path = KNEE_PATH.parent / "mode_comparison.txt"
out_path.write_text(output + "\n", encoding="utf-8")
print(f"\nSaved to {out_path}")
