"""Debug script to visualize traffic routing and latency distributions for a given metro.

Usage:
    python debug_metro.py --airport LON --geo EMEA --bucket OtherBigFoot

For each serving metro that the given airport routes traffic to, this script plots:
  1. A bar chart of traffic volumes (Mbps) to each serving metro
  2. The edge RTT CDF between the source metro and each serving metro
  3. The edge TAT CDF at each serving metro (cache hits)
  4. The midgress RTT CDF between each serving metro and its parent MCH
  5. The parent MCH TAT CDF for each serving metro

All figures are saved to DEBUG/<airport-code>/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", type=str, required=True, help="Airport code of the metro to debug (e.g. LON)")
    parser.add_argument("--geo", type=str, default="EMEA", help="Geo/macroarea (e.g. NA, EMEA, LA, APAC)")
    parser.add_argument("--bucket", type=str, default="OtherBigFoot", help="Traffic bucket name")
    parser.add_argument("--traffic-threshold", type=float, default=0.0, help="Minimum traffic (Mbps) to include a serving metro")
    return parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_geo_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    csv_path = _BASE_DIR / "PERF" / "geo_mapping.csv"
    if not csv_path.exists():
        return mapping
    with csv_path.open() as f:
        f.readline()
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(',')]
            if len(parts) >= 3:
                mapping[parts[0]] = parts[2]
    return mapping


def parse_metro_areas(target_geo: str) -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    airport_info: dict[str, dict] = {}
    metro_to_airport: dict[str, str] = {}
    airport_to_metro: dict[str, str] = {}
    geo_mapping = load_geo_mapping()

    with (_BASE_DIR / "PERF" / "metro_areas.csv").open() as f:
        f.readline()
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(',')]
            if len(parts) != 8:
                continue
            _id, metro_area, lat, lon, airport_code, country, state, max_dist = parts
            if geo_mapping.get(metro_area) != target_geo:
                continue
            airport_info[airport_code] = {
                "metro_area": metro_area,
                "latitude": float(lat),
                "longitude": float(lon),
            }
            metro_to_airport[metro_area] = airport_code
            airport_to_metro[airport_code] = metro_area
    return airport_info, metro_to_airport, airport_to_metro


def load_traffic_lookup(bucket: str) -> dict[tuple[str, str], float]:
    csv_path = _BASE_DIR / "SERVEDFROM_DATA" / f"served_from_{bucket}.csv"
    lookup: dict[tuple[str, str], float] = {}
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return lookup

    def _parse(line: str) -> list[str]:
        return [p.strip().strip('"') for p in line.strip().lstrip("\ufeff").split('","')]

    with csv_path.open(encoding="utf-8") as f:
        header = _parse(f.readline())
        try:
            asn_idx = header.index("asn_metro")
            bw_idx = header.index("bw_metro")
            traffic_idx = header.index("traffic_mbps")
        except ValueError:
            asn_idx, bw_idx, traffic_idx = 0, 1, 2
        for line in f:
            if not line.strip():
                continue
            row = _parse(line)
            lookup[(row[asn_idx], row[bw_idx])] = float(row[traffic_idx])
    return lookup


def assign_parent(airport_code: str, airport_info: dict[str, dict]) -> str | None:
    """Re-use the same nearest-MCH logic as solve_for_US.py."""
    from analyse import client_metro_ids
    import math

    ALL_MCH = [
        "ORD", "DFW", "LGA", "IAD", "ATL", "MIA", "SEA", "SJC", "LAX", "LON",
        "FRA", "RIO", "PAR", "AMS", "MIL", "TYO", "OSA", "SIN", "HKG", "MAD",
        "SYD", "GRU", "BOS", "DEN", "STO", "BOM", "MAA", "MEL", "BUE", "SCL",
        "MEX", "QRO", "CGK",
    ]

    if airport_code not in airport_info:
        return None

    lat = airport_info[airport_code]["latitude"]
    lon = airport_info[airport_code]["longitude"]

    best_mch = None
    best_dist = float("inf")
    for mch in ALL_MCH:
        if mch not in airport_info:
            continue
        mlat = airport_info[mch]["latitude"]
        mlon = airport_info[mch]["longitude"]
        dist = math.sqrt((lat - mlat) ** 2 + (lon - mlon) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_mch = mch
    return best_mch


# ── plotting helpers ──────────────────────────────────────────────────────────

def _query_request_count(query: str) -> int:
    """Run a COUNT(*) / SUM(total_requests) variant of a query and return the total."""
    import sqlite3
    import re
    # Wrap the query to sum total_requests
    count_query = f"SELECT SUM(total_requests) FROM ({query})"
    db_path = _BASE_DIR / "PERF" / "perf_data.db"
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(count_query)
        result = cursor.fetchone()
        conn.close()
        if result and result[0] is not None:
            return int(result[0])
    except Exception as e:
        print(f"    Could not fetch request count: {e}")
    return 0


def _get_rtt_request_count(metro_name: str, client_metro_id: int) -> int:
    query = f"""SELECT total_requests FROM netopt_perf_edge_rtt_ansabni
    WHERE region_metro = '{metro_name}' AND client_metro = '{client_metro_id}'
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""
    return _query_request_count(query)


def _get_edge_tat_request_count(metro_name: str, cache_hit_type: int = 1) -> int:
    query = f"""SELECT total_requests FROM netopt_perf_edge_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type = {cache_hit_type}
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""
    return _query_request_count(query)


def _get_midgress_rtt_request_count(parent_metro: str, child_metro: str) -> int:
    query = f"""SELECT total_requests FROM netopt_perf_midgress_rtt_ansabni
    WHERE parent_metro = '{parent_metro}' AND child_metro = '{child_metro}'
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""
    return _query_request_count(query)


def _get_parent_tat_request_count(metro_name: str) -> int:
    query = f"""SELECT total_requests FROM netopt_perf_midgress_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type != 2
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""
    return _query_request_count(query)


def _save(fig, path: Path, title: str, request_count: int = 0):
    import matplotlib.pyplot as plt
    count_str = f"{request_count:,}" if request_count > 0 else "N/A"
    fig.suptitle(f"{title}\n(based on {count_str} requests)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_traffic_bar(serving_metros: list[str], traffic: dict[str, float], out_dir: Path, airport: str):
    import matplotlib.pyplot as plt

    labels = serving_metros
    values = [traffic.get(m, 0.0) / 1000.0 for m in labels]  # Gbps

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    bars = ax.bar(labels, values, color="steelblue")
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xlabel("Serving Metro")
    ax.set_ylabel("Traffic (Gbps)")
    ax.set_title(f"Traffic from {airport} to each serving metro")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, out_dir / "traffic_routing.png", f"{airport} → Serving Metros")


def plot_pdf(pdf, title: str, path: Path, request_count: int = 0):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    pdf.plot(ax=ax, title=title)
    _save(fig, path, title, request_count=request_count)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    airport = args.airport.upper()

    from analyse import get_rtt_pdf, get_edge_tat_pdf, get_midgress_rtt_pdf, get_parent_tat_pdf, client_metro_ids

    out_dir = _BASE_DIR / "DEBUG" / airport
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDebug output directory: {out_dir}\n")

    # ── Load metro topology ───────────────────────────────────────────────────
    airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(args.geo)

    if airport not in airport_to_metro:
        print(f"WARNING: {airport} not found in geo={args.geo}. Continuing anyway with all metros.")
        # Allow the script to work even if the airport isn't in the filtered geo
        with (_BASE_DIR / "PERF" / "metro_areas.csv").open() as f:
            f.readline()
            for line in f:
                parts = [p.strip().strip('"') for p in line.strip().split(',')]
                if len(parts) != 8:
                    continue
                _id, metro_area, lat, lon, airport_code, country, state, _ = parts
                airport_info[airport_code] = {"metro_area": metro_area, "latitude": float(lat), "longitude": float(lon)}
                metro_to_airport[metro_area] = airport_code
                airport_to_metro[airport_code] = metro_area

    metro_name = airport_to_metro.get(airport, airport)
    print(f"Metro: {airport} ({metro_name}), Geo: {args.geo}, Bucket: {args.bucket}\n")

    # ── Traffic routing ───────────────────────────────────────────────────────
    traffic_lookup = load_traffic_lookup(args.bucket)
    traffic_lookup_by_airport: dict[tuple[str, str], float] = {}
    for (asn, bw), t in traffic_lookup.items():
        if asn in metro_to_airport and bw in metro_to_airport:
            traffic_lookup_by_airport[(metro_to_airport[asn], metro_to_airport[bw])] = t

    serving_metros = sorted(
        [bw for (asn, bw), t in traffic_lookup_by_airport.items()
         if asn == airport and t > args.traffic_threshold],
        key=lambda m: -traffic_lookup_by_airport.get((airport, m), 0.0),
    )

    if not serving_metros:
        print(f"No serving metros found for {airport} above {args.traffic_threshold} Mbps threshold.")
        return

    print(f"Serving metros for {airport}: {serving_metros}\n")

    # 1. Traffic bar chart
    print("Plotting traffic routing bar chart...")
    plot_traffic_bar(
        serving_metros,
        {m: traffic_lookup_by_airport.get((airport, m), 0.0) for m in serving_metros},
        out_dir,
        airport,
    )

    # 2–5. Per serving-metro distributions
    for serving in serving_metros:
        serving_metro_name = airport_to_metro.get(serving, serving)
        traffic_gbps = traffic_lookup_by_airport.get((airport, serving), 0.0) / 1000.0
        print(f"\n  [{serving}] ({serving_metro_name}) — {traffic_gbps:.1f} Gbps")

        # 2. Edge RTT: client metro (airport) → serving metro
        client_id = client_metro_ids.get(metro_name)
        if client_id is not None:
            print(f"    Edge RTT: {metro_name} (id={client_id}) → {serving_metro_name}")
            rtt_pdf = get_rtt_pdf(serving_metro_name, client_id)
            if rtt_pdf.total_count() > 0:
                req_count = _get_rtt_request_count(serving_metro_name, client_id)
                plot_pdf(
                    rtt_pdf,
                    f"Edge RTT: {airport} clients → {serving} ({traffic_gbps:.1f}G)",
                    out_dir / f"edge_rtt_{airport}_to_{serving}.png",
                    request_count=req_count,
                )
            else:
                print(f"    No edge RTT data for {metro_name} → {serving_metro_name}")
        else:
            print(f"    No client_metro_id found for {metro_name}, skipping edge RTT")

        # 3. Edge TAT (cache hits) at the serving metro
        print(f"    Edge TAT (cache hit) at {serving_metro_name}")
        edge_tat_pdf = get_edge_tat_pdf(serving_metro_name, cache_hit_type=1)
        if edge_tat_pdf.total_count() > 0:
            req_count = _get_edge_tat_request_count(serving_metro_name, cache_hit_type=1)
            plot_pdf(
                edge_tat_pdf,
                f"Edge TAT (cache hit) at {serving} ({traffic_gbps:.1f}G)",
                out_dir / f"edge_tat_{serving}.png",
                request_count=req_count,
            )
        else:
            print(f"    No edge TAT data for {serving_metro_name}")

        # 4. Midgress RTT: serving metro → its parent MCH
        parent = assign_parent(serving, airport_info)
        if parent:
            parent_name = airport_to_metro.get(parent, parent)
            print(f"    Midgress RTT: {serving_metro_name} → parent {parent} ({parent_name})")
            midgress_pdf = get_midgress_rtt_pdf(parent_name, serving_metro_name)
            if midgress_pdf.total_count() > 0:
                req_count = _get_midgress_rtt_request_count(parent_name, serving_metro_name)
                plot_pdf(
                    midgress_pdf,
                    f"Midgress RTT: {serving} → MCH {parent} ({parent_name})",
                    out_dir / f"midgress_rtt_{serving}_to_{parent}.png",
                    request_count=req_count,
                )
            else:
                print(f"    No midgress RTT data for {serving_metro_name} → {parent_name}")

            # 5. Parent MCH TAT at the serving metro (as seen by its MCH parent)
            print(f"    Parent MCH TAT for {serving_metro_name}")
            parent_tat_pdf = get_parent_tat_pdf(serving_metro_name)
            if parent_tat_pdf.total_count() > 0:
                req_count = _get_parent_tat_request_count(serving_metro_name)
                plot_pdf(
                    parent_tat_pdf,
                    f"Parent MCH TAT for {serving} (via MCH {parent})",
                    out_dir / f"parent_tat_{serving}.png",
                    request_count=req_count,
                )
            else:
                print(f"    No parent TAT data for {serving_metro_name}")
        else:
            print(f"    Could not determine parent MCH for {serving}")

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
