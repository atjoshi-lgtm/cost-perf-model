"""Debug script: sweep the footprint descriptor curve in 1 TB increments and compute P50/P95 TTFB.

All traffic is routed to the given metro itself (self-serving), so the edge RTT is the
RTT from that metro's clients to that metro's edge servers.

Usage:
    python3.11 debug_ttfb_vs_disk.py --airport DOH --geo EMEA --bucket OtherBigFoot

Figures are saved to DEBUG_TTFB_DST/<airport>/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent

# ── constants matching solve_for_US.py ────────────────────────────────────────
ALL_MCH_METROS = [
    "ORD", "DFW", "LGA", "IAD", "ATL", "MIA", "SEA", "SJC", "LAX", "LON",
    "FRA", "RIO", "PAR", "AMS", "MIL", "TYO", "OSA", "SIN", "HKG", "MAD",
    "SYD", "GRU", "BOS", "DEN", "STO", "BOM", "MAA", "MEL", "BUE", "SCL",
    "MEX", "QRO", "CGK",
]

TB_IN_MB = 1_000_000      # 1 TB in MB
STEP_TB   = 1             # sweep step size
STEP_US   = 10            # microsecond bucket resolution for convolution

# Mirrors solve_for_US.py — remap airport codes to their FDS filename stems
FDS_EXCEPTIONS: dict[str, str] = {"LGA": "EWR_LGA"}


# ── arg parsing ───────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", required=True, help="Airport code to debug (e.g. DOH)")
    parser.add_argument("--geo",     default="EMEA", help="Macroarea (NA, EMEA, LA, APAC)")
    parser.add_argument("--bucket",  default="OtherBigFoot", help="FDS bucket name")
    return parser.parse_args()


# ── metro topology helpers ─────────────────────────────────────────────────────
def load_geo_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    p = _BASE_DIR / "PERF" / "geo_mapping.csv"
    if not p.exists():
        return mapping
    with p.open() as f:
        f.readline()
        for line in f:
            parts = [x.strip().strip('"') for x in line.strip().split(',')]
            if len(parts) >= 3:
                mapping[parts[0]] = parts[2]
    return mapping


def parse_metro_areas(geo: str):
    airport_info: dict[str, dict] = {}
    metro_to_airport: dict[str, str] = {}
    airport_to_metro: dict[str, str] = {}
    mch_metros: list[str] = []
    geo_mapping = load_geo_mapping()

    with (_BASE_DIR / "PERF" / "metro_areas.csv").open() as f:
        f.readline()
        for line in f:
            parts = [x.strip().strip('"') for x in line.strip().split(',')]
            if len(parts) != 8:
                continue
            _id, metro_area, lat, lon, code, country, state, _ = parts
            if geo_mapping.get(metro_area) != geo:
                continue
            airport_info[code] = {
                "metro_area": metro_area,
                "latitude": float(lat),
                "longitude": float(lon),
            }
            metro_to_airport[metro_area] = code
            airport_to_metro[code] = metro_area
            if code in ALL_MCH_METROS:
                mch_metros.append(code)

    return airport_info, metro_to_airport, airport_to_metro, mch_metros


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_parent(airport: str, airport_info: dict, mch_metros: list[str]) -> str | None:
    if airport in mch_metros:
        return airport
    best, best_d = None, float("inf")
    for mch in mch_metros:
        if mch not in airport_info:
            continue
        d = haversine(
            airport_info[airport]["latitude"], airport_info[airport]["longitude"],
            airport_info[mch]["latitude"],    airport_info[mch]["longitude"],
        )
        if d < best_d:
            best_d, best = d, mch
    return best


# ── FDS loading ───────────────────────────────────────────────────────────────
def _fds_path(airport: str, bucket: str) -> Path:
    """Return the FDS file path for an airport, applying known key exceptions."""
    fds_dir = _BASE_DIR / f"FDS_{bucket}"
    metro_key = FDS_EXCEPTIONS.get(airport, airport).lower()
    return fds_dir / f"{metro_key}.txt"


def _read_fd_text(airport: str, bucket: str) -> str | None:
    p = _fds_path(airport, bucket)
    return p.read_text(encoding="utf-8") if p.exists() else None


def load_fds(airport: str, bucket: str, airport_info: dict):
    """Load FDS for airport, falling back to the nearest metro that has a file."""
    from fds import FootprintDescriptor

    fds_dir = _BASE_DIR / f"FDS_{bucket}"

    # Build list of metros that actually have an FDS file (mirrors solve_for_US.py)
    fds_metros: list[tuple[str, dict]] = []
    for fd_path in fds_dir.glob("*.txt"):
        code = fd_path.stem.upper()
        # Reverse-map exceptions (e.g. EWR_LGA -> LGA)
        for orig, mapped in FDS_EXCEPTIONS.items():
            if mapped.lower() == fd_path.stem.lower():
                code = orig
                break
        if code in airport_info:
            fds_metros.append((code, airport_info[code]))

    fd_text = _read_fd_text(airport, bucket)
    source = airport

    if fd_text is None:
        print(f"No FDS file for {airport} in FDS_{bucket}, searching nearest metro...")
        visited: set[str] = {airport}
        candidate = find_nearest_fds_metro(airport_info, airport, fds_metros, visited)
        while candidate is not None and candidate not in visited:
            visited.add(candidate)
            fd_text = _read_fd_text(candidate, bucket)
            if fd_text is not None:
                source = candidate
                print(f"  Using FDS from nearest metro: {candidate}")
                break
            candidate = find_nearest_fds_metro(airport_info, candidate, fds_metros, visited)

    if fd_text is None:
        raise FileNotFoundError(
            f"No FDS file found for {airport} or any nearby metro in FDS_{bucket}/"
        )

    descriptor = FootprintDescriptor.from_text(fd_text).smooth_by_cache_bucket(10 * 1024)
    return descriptor, source


def find_nearest_fds_metro(
    airport_info: dict,
    airport: str,
    fds_metros: list[tuple[str, dict]],
    exclude: set[str],
) -> str | None:
    if airport not in airport_info:
        return None
    info = airport_info[airport]
    best, best_d = None, float("inf")
    for code, fds_info in fds_metros:
        if code in exclude:
            continue
        d = haversine(
            info["latitude"], info["longitude"],
            fds_info["latitude"], fds_info["longitude"],
        )
        if d < best_d:
            best_d, best = d, code
    return best


# ── plotting ──────────────────────────────────────────────────────────────────
def save_fig(fig, path: Path, title: str):
    import matplotlib.pyplot as plt
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_ttfb_curve(disk_tb_list, p50_list, p95_list, out_dir: Path, airport: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(disk_tb_list, p50_list, marker='o', markersize=3, label="P50 TTFB (ms)", color="steelblue")
    ax.plot(disk_tb_list, p95_list, marker='s', markersize=3, label="P95 TTFB (ms)", color="crimson")
    ax.set_xlabel("Disk (TB)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{airport}: P50/P95 TTFB vs Disk Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir / "ttfb_vs_disk.png", f"{airport} TTFB vs Disk")


def plot_hitrate_curve(disk_tb_list, hitrate_list, out_dir: Path, airport: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(disk_tb_list, hitrate_list, marker='o', markersize=3, color="green")
    ax.set_xlabel("Disk (TB)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title(f"{airport}: Hit Rate vs Disk Size")
    ax.grid(True, alpha=0.3)
    save_fig(fig, out_dir / "hitrate_vs_disk.png", f"{airport} Hit Rate vs Disk")


def plot_pdf_cdf(pdf, title: str, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    pdf.plot(ax=ax, title=title)
    save_fig(fig, path, title)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    airport = args.airport.upper()

    from analyse import get_rtt_pdf, get_edge_tat_pdf, get_midgress_rtt_pdf, get_parent_tat_pdf, client_metro_ids
    from perf_with_mch import MetroPerformanceWithMCH, MCHPerformanceModel
    from probability import Convolution, weighted_pdf_sum

    out_dir = _BASE_DIR / "DEBUG_TTFB_DST" / airport
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    # ── topology ──────────────────────────────────────────────────────────────
    airport_info, metro_to_airport, airport_to_metro, mch_metros = parse_metro_areas(args.geo)

    if airport not in airport_to_metro:
        raise SystemExit(f"Airport {airport} not found in geo={args.geo}")

    metro_name = airport_to_metro[airport]
    parent_airport = find_parent(airport, airport_info, mch_metros)
    if parent_airport is None:
        raise SystemExit(f"Could not find a parent MCH for {airport}")
    parent_name = airport_to_metro.get(parent_airport, parent_airport)
    print(f"Metro:  {airport} ({metro_name})")
    print(f"Parent: {parent_airport} ({parent_name})")

    # ── PDFs ──────────────────────────────────────────────────────────────────
    client_id = client_metro_ids.get(metro_name)
    if client_id is None:
        raise SystemExit(f"No client_metro_id for {metro_name}. Cannot get edge RTT.")

    print(f"\nLoading PDFs...")
    edge_rtt_pdf   = get_rtt_pdf(metro_name, client_id)
    edge_tat_pdf   = get_edge_tat_pdf(metro_name, cache_hit_type=1)
    midgress_pdf   = get_midgress_rtt_pdf(parent_name, metro_name)
    parent_tat_pdf = get_parent_tat_pdf(metro_name)

    # Save the raw input PDFs for inspection
    print("\nSaving input distribution plots...")
    if edge_rtt_pdf.total_count() > 0:
        plot_pdf_cdf(edge_rtt_pdf,  f"Edge RTT: {metro_name} clients → {airport}",  out_dir / "input_edge_rtt.png")
    else:
        print(f"  WARNING: No edge RTT data for {metro_name}")

    if edge_tat_pdf.total_count() > 0:
        plot_pdf_cdf(edge_tat_pdf,  f"Edge TAT (cache hit) at {airport}",           out_dir / "input_edge_tat.png")
    else:
        print(f"  WARNING: No edge TAT data for {metro_name}")

    if midgress_pdf.total_count() > 0:
        plot_pdf_cdf(midgress_pdf,  f"Midgress RTT: {airport} → MCH {parent_airport}", out_dir / "input_midgress_rtt.png")
    else:
        print(f"  WARNING: No midgress RTT data for {metro_name} → {parent_name}")

    if parent_tat_pdf.total_count() > 0:
        plot_pdf_cdf(parent_tat_pdf, f"Parent MCH TAT at {airport} (via {parent_airport})", out_dir / "input_parent_tat.png")
    else:
        print(f"  WARNING: No parent TAT data for {metro_name}")

    # ── FDS ───────────────────────────────────────────────────────────────────
    print(f"\nLoading footprint descriptor for {airport} (bucket={args.bucket})...")
    descriptor, fds_source = load_fds(airport, args.bucket, airport_info)
    if fds_source != airport:
        print(f"  (using FDS from {fds_source} as fallback)")

    max_cache_mb = max(pt.cache_space for pt in descriptor._points_sorted_by_cache)
    max_cache_tb = max_cache_mb / TB_IN_MB
    print(f"FDS max cache space: {max_cache_tb:.1f} TB\n")

    # ── Build MCH + metro performance models ──────────────────────────────────
    mch_model = MCHPerformanceModel(name=parent_airport)
    mch_model.set_mch_tat(parent_tat_pdf)

    metro_model = MetroPerformanceWithMCH(
        name=airport,
        parent_model=mch_model,
        descriptor=descriptor,
    )
    metro_model.set_edge_tat_hit(edge_tat_pdf)
    metro_model.set_mch_rtt(midgress_pdf)
    # Route all traffic to self: the "from" metro is also this airport
    metro_model.set_edge_rtt(airport, edge_rtt_pdf)

    conn = Convolution()

    # ── Sweep disk from 0 to max in 1 TB steps ────────────────────────────────
    print("Sweeping disk sizes...")
    disk_tb_list:  list[float] = []
    hitrate_list:  list[float] = []
    p50_list:      list[float] = []
    p95_list:      list[float] = []

    disk_tb = 0.0
    while disk_tb <= max_cache_tb + STEP_TB:
        disk_mb = disk_tb * TB_IN_MB
        hitrate = min(max(descriptor.hitrate_for_cache(disk_mb), 0.0), 100.0)

        ttfb_pdf = metro_model.get_ttfb_pdf(
            from_metro=airport,
            hitrate=hitrate,
            conn=conn,
        )
        micro_pdf = ttfb_pdf.to_microsecond_pdf(step_us=STEP_US)
        p50 = micro_pdf.millisecond_at_percentile(50) / 1000.0
        p95 = micro_pdf.millisecond_at_percentile(95) / 1000.0

        disk_tb_list.append(disk_tb)
        hitrate_list.append(hitrate)
        p50_list.append(p50)
        p95_list.append(p95)

        print(f"  Disk={disk_tb:6.1f} TB  Hitrate={hitrate:5.1f}%  P50={p50:.2f} ms  P95={p95:.2f} ms")
        disk_tb += STEP_TB

    # ── Plot summary curves ───────────────────────────────────────────────────
    print("\nSaving summary plots...")
    plot_ttfb_curve(disk_tb_list, p50_list, p95_list, out_dir, airport)
    plot_hitrate_curve(disk_tb_list, hitrate_list, out_dir, airport)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "ttfb_vs_disk.csv"
    with csv_path.open("w") as f:
        f.write("disk_tb,hitrate_pct,p50_ms,p95_ms\n")
        for d, h, p50, p95 in zip(disk_tb_list, hitrate_list, p50_list, p95_list):
            f.write(f"{d:.1f},{h:.4f},{p50:.4f},{p95:.4f}\n")
    print(f"  Saved: ttfb_vs_disk.csv")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
