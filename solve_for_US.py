# Wohoo, we are going to discover the disk needs of each metro in the US! We are going to be using multithreading to do it!
# Let's begin. from __future__ import annotations

from pathlib import Path
import random
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from cost import CaribouCostCalculator, CaribouCostBreakdown
from fds import FootprintDescriptor
from perf_with_mch import MetroPerformanceWithMCH, MCHPerformanceModel
from probability import Convolution, gaussian_pdf, weighted_pdf_sum
from analyse import *
from itertools import product
from collections import defaultdict

import sys
import argparse

INT32_MAX = 2**31 - 1
_BASE_DIR = Path(__file__).resolve().parent
_FDS_DIR = _BASE_DIR / "FDS"

def get_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--geo", type=str, default="NA", help="The macroarea (geo) to filter by (e.g., NA, EMEA, LA, APAC)")
    parser.add_argument("--bucket", type=str, default="AkamaiHD", help="The bucket name (e.g., AkamaiHD, OtherBigFoot)")
    parser.add_argument("--traffic-threshold", type=float, default=20000.0, help="Minimum traffic (in Mbps) required to consider a network edge for optimization.")
    return parser.parse_args(args_list)

# Fallback values if imported, but populated if run directly
_GEO = "NA"
_BUCKET = "AkamaiHD"
_FDS2_DIR = _BASE_DIR / f"FDS_{_BUCKET}"


FDS_EXCEPTIONS = { "LGA" : "EWR_LGA"}

ALL_MCH_METROS = [
    "ORD", "DFW", "LGA", "IAD", "ATL", "MIA", "SEA", "SJC", "LAX", "LON", 
    "FRA", "RIO", "PAR", "AMS", "MIL", "TYO", "OSA", "SIN", "HKG", "MAD", 
    "SYD", "GRU", "BOS", "DEN", "STO", "BOM", "MAA", 
    "MEL", "BUE", "SCL", "MEX", "QRO", "CGK"
]
MCH_METROS = [] # This will be populated dynamically based on geo

ALL_METROS = []

def load_geo_mapping() -> dict[str, str]:
    mapping = {}
    csv_path = _BASE_DIR / "PERF" / "geo_mapping.csv"
    if not csv_path.exists():
        return mapping
    
    with csv_path.open("r", encoding="utf-8") as f:
        # Columns: metro_area, country, macroarea
        header = f.readline()
        for line in f:
            parts = [part.strip().strip('"') for part in line.strip().split(',')]
            if len(parts) >= 3:
                metro_area = parts[0]
                macroarea = parts[2]
                mapping[metro_area] = macroarea
    return mapping

def parse_metro_areas(file_path: Path, target_geo: str = "NA") -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    global MCH_METROS
    airport_info = {}
    metro_to_airport = {}
    airport_to_metro = {}
    
    geo_mapping = load_geo_mapping()

    with file_path.open() as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 8:
                continue  # Skip malformed lines

            # Strip the quotes from each part
            parts = [part.strip().strip('"') for part in parts]

            id, metro_area, latitude, longitude, airport_code, country, state, max_distance = parts
            
            # Determine geo for this metro
            metro_geo = geo_mapping.get(metro_area)
            
            # Use geo from the macroarea instead of treating 'US' as a geo
            if metro_geo != target_geo:
                continue
                
            if airport_code in ALL_MCH_METROS and airport_code not in MCH_METROS:
                MCH_METROS.append(airport_code)

            ALL_METROS.append(airport_code)

            airport_info[airport_code] = {
                'id': id,
                'metro_area': metro_area,
                'latitude': float(latitude),
                'longitude': float(longitude),
                'country': country,
                'state': state,
                'max_distance_from_center': float(max_distance)
            }
            metro_to_airport[metro_area] = airport_code
            airport_to_metro[airport_code] = metro_area

    return airport_info, metro_to_airport, airport_to_metro

def get_metro_tiers():
    metro_tiers = {}
    with (_BASE_DIR / "PERF" / "metro_tiers.csv").open() as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            # Each line looks like: '"Doha"': '"2"', can you remove the quotes and parse it correctly?
            parts = [part.strip().strip('"') for part in parts]
            if len(parts) != 2:
                continue  # Skip malformed lines

            metro_area, tier = parts
            metro_tiers[metro_area] = int(tier)

    return metro_tiers

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Radius of Earth in kilometers
    radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius_km * c

def get_nearest_metro(airport_info: dict[str, dict], given_metro: str, fds_metros: list[tuple[str, dict]]) -> str | None:
    if given_metro not in airport_info:
        return None

    given_metro_info = airport_info[given_metro]

    nearest_metro = None
    nearest_distance = float('inf')

    for metro, info in fds_metros:

        if metro == given_metro:
            continue  # Skip the given metro itself

        # Calculate distance using the Haversine formula
        distance = haversine_distance(
            given_metro_info['latitude'], given_metro_info['longitude'],
            info['latitude'], info['longitude']
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_metro = metro

    return nearest_metro

def assign_parent_metros(airport_info: dict) -> dict[str, str]:
    parent_assignment = {}
    for metro in ALL_METROS:
        if metro in MCH_METROS:
            parent_assignment[metro] = (metro, 0)  # A metro in MCH_METROS is its own parent
        else:
            for mch_metro in MCH_METROS:
                if mch_metro in airport_info and metro in airport_info:
                    distance = haversine_distance(
                        airport_info[metro]['latitude'], airport_info[metro]['longitude'],
                        airport_info[mch_metro]['latitude'], airport_info[mch_metro]['longitude']
                    )
                    if metro not in parent_assignment or distance < parent_assignment[metro][1]:
                        parent_assignment[metro] = (mch_metro, distance)

    return parent_assignment

def _load_traffic_lookup() -> dict[tuple[str, str], float]:
    csv_path = _BASE_DIR / "SERVEDFROM_DATA" / f"served_from_{_BUCKET}.csv"
    lookup: dict[tuple[str, str], float] = {}

    def _parse_csv_line(line: str) -> list[str]:
        cleaned_line = line.strip().lstrip("\ufeff")
        return [part.strip().strip('"') for part in cleaned_line.split('","')]

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        header_line = csv_file.readline()
        if not header_line:
            return lookup
        normalized_header = _parse_csv_line(header_line)
        asn_idx = normalized_header.index("asn_metro")
        bw_idx = normalized_header.index("bw_metro")
        traffic_idx = normalized_header.index("traffic_mbps")
        for line in csv_file:
            if not line.strip():
                continue
            normalized_row = _parse_csv_line(line)
            lookup[(normalized_row[asn_idx], normalized_row[bw_idx])] = float(normalized_row[traffic_idx])
    return lookup

# Write a function that plots the neighborhood dictionary as a graph. Each metro is a node and there is an edge between two metros if they are in each other's neighborhood.
# The weight of the edge is the traffic between the two metros. You know the latitude and longitude of each metro from the airport_info dictionary, 
# so you can use that to position the nodes on the graph. Use matplotlib to plot the graph.
def plot_neighborhood_graph(neighborhood: dict[str, list[str]], airport_info: dict[str, dict], traffic_lookup: dict[tuple[str, str], float]):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()

    # Add nodes with positions based on latitude and longitude
    for metro in neighborhood:
        lat = airport_info[metro]['latitude']
        lon = airport_info[metro]['longitude']
        G.add_node(metro, pos=(lon, lat))

    # Add edges with weights based on traffic
    for metro, neighbors in neighborhood.items():
        for neighbor in neighbors:
            traffic = traffic_lookup.get((metro, neighbor), 0) + traffic_lookup.get((neighbor, metro), 0)
            G.add_edge(metro, neighbor, weight=traffic)

    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10)
    
    # Draw edge labels with traffic weights
    edge_labels = {edge: f"{weight:.1f} Mbps" for edge, weight in weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Neighborhood Graph of Metros")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


def build_mch_performance_model(metro_code: str, metro_name: str) -> tuple[str, MCHPerformanceModel]:
    """Build the parent MCH performance model for a single MCH metro."""

    mch_performance_model = MCHPerformanceModel(name=metro_code)
    tat_pdf = get_parent_tat_pdf(metro_name)
    mch_performance_model.set_mch_tat(tat_pdf)
    return metro_code, mch_performance_model


def load_mch_performance_models_threaded(mch_metros: list[str], airport_to_metro: dict[str, str]) -> dict[str, MCHPerformanceModel]:
    """Load MCH performance models in parallel using Python threads."""

    if not mch_metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    models: dict[str, MCHPerformanceModel] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Here, I want to convert the metro names in MCH_METROS to the corresponding metro area names using the airport_to_metro mapping, and then build the MCH performance model for each metro area name. I will store the results in a dictionary where the keys are the original metro names from MCH_METROS and the values are the corresponding MCHPerformanceModel instances.

        future_to_metro = {
            executor.submit(build_mch_performance_model, metro, airport_to_metro[metro]): metro
            for metro in mch_metros if metro in airport_to_metro
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            built_metro, model = future.result()
            models[built_metro] = model
            print(f"Loaded MCH performance model for {metro}")

    return models


def _fds_path_for_metro(metro: str) -> Path:
    metro_key = FDS_EXCEPTIONS.get(metro, metro).lower()
    return _FDS2_DIR / f"{metro_key}.txt"


def _read_fd_text_for_metro(metro: str) -> str | None:
    """Read raw FD text for a metro from the local FDS2 directory, if available."""

    fd_path = _fds_path_for_metro(metro)
    if not fd_path.exists():
        return None
    return fd_path.read_text(encoding="utf-8")


def get_smoothed_fd_for_metro(
    metro: str,
    airport_info: dict[str, dict],
    metro_tiers: dict[str, int],
    fds_metros: list[tuple[str, dict]],
    bucket_size_mb: int = 10 * 1024,
) -> FootprintDescriptor:
    """Load and smooth a metro FD, falling back to the nearest metro FD when needed."""

    fd_text = _read_fd_text_for_metro(metro)
    source_metro = metro
    visited: set[str] = {metro}


    if fd_text is None:
        fallback_metro = get_nearest_metro(airport_info, metro, fds_metros)
        print(f"No FD found for {metro}, trying nearest metro {fallback_metro}")
        while fallback_metro is not None and fallback_metro not in visited:
            visited.add(fallback_metro)
            fd_text = _read_fd_text_for_metro(fallback_metro)
            if fd_text is not None:
                source_metro = fallback_metro
                break
            fallback_metro = get_nearest_metro(airport_info, fallback_metro, fds_metros)

    print(f"FD text for {metro} loaded from {source_metro}, fallback metros tried: {visited - {source_metro}}")

    if fd_text is None:
        raise FileNotFoundError(f"No Footprint Descriptor available for metro {metro} or its nearest fallback metros")

    descriptor = FootprintDescriptor.from_text(fd_text).smooth_by_cache_bucket(bucket_size_mb)

    if source_metro != metro:
        print(f"Using fallback FD from {source_metro} for metro {metro}")

    return descriptor


def load_smoothed_fds_for_all_metros(
    metros: list[str],
    airport_info: dict[str, dict],
    metro_tiers: dict[str, int],
    fds_metros: list[tuple[str, dict]],
    bucket_size_mb: int = 10 * 1024,
) -> dict[str, FootprintDescriptor]:
    """Load and smooth an FD for every metro in the provided list."""

    descriptors: dict[str, FootprintDescriptor] = {}
    for metro in metros:
        if metro not in airport_info:
            continue
        descriptors[metro] = get_smoothed_fd_for_metro(
            metro,
            airport_info,
            metro_tiers,
            fds_metros,
            bucket_size_mb=bucket_size_mb,
        )
    return descriptors


def load_smoothed_fds_for_all_metros_threaded(
    metros: list[str],
    airport_info: dict[str, dict],
    metro_tiers: dict[str, int],
    fds_metros: list[tuple[str, dict]],
    bucket_size_mb: int = 10 * 1024,
) -> dict[str, FootprintDescriptor]:
    """Load and smooth footprint descriptors for all metros in parallel."""

    if not metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    descriptors: dict[str, FootprintDescriptor] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_metro = {
            executor.submit(
                get_smoothed_fd_for_metro,
                metro,
                airport_info,
                metro_tiers,
                fds_metros,
                bucket_size_mb,
            ): metro
            for metro in metros
            if metro in airport_info
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            descriptors[metro] = future.result()
            print(f"Loaded smoothed footprint descriptor for {metro}")

    return descriptors


def build_metro_performance_model(
    metro: str,
    descriptors: dict[str, FootprintDescriptor],
    airport_to_metro: dict[str, str],
    parent_assignment: dict[str, tuple[str, float]],
    neighborhood_from: dict[str, list[str]],
    mch_performance_models: dict[str, MCHPerformanceModel],
) -> tuple[str, MetroPerformanceWithMCH]:
    """Build a MetroPerformanceWithMCH for a single metro."""

    metro_name = airport_to_metro[metro]
    parent_metro = parent_assignment[metro][0]
    parent_model = mch_performance_models[parent_metro]
    descriptor = descriptors[metro]
    metro_performance = MetroPerformanceWithMCH(
        name=metro,
        parent_model=parent_model,
        descriptor=descriptor,
    )

    edge_tat_hit = get_edge_tat_pdf(metro_name)
    metro_performance.set_edge_tat_hit(edge_tat_hit)

    parent_name = airport_to_metro[parent_metro]
    rtt_cache_miss = get_midgress_rtt_pdf(parent_name, metro_name)
    metro_performance.set_mch_rtt(rtt_cache_miss)

    for from_metro in neighborhood_from.get(metro, []):
        if from_metro not in airport_to_metro:
            continue
        edge_rtt_pdf = get_rtt_pdf(metro_name, client_metro_ids[airport_to_metro[from_metro]])
        metro_performance.set_edge_rtt(from_metro, edge_rtt_pdf)

    return metro, metro_performance

def load_performance_models_threaded(
    metros: list[str],
    descriptors: dict[str, FootprintDescriptor],
    airport_to_metro: dict[str, str],
    parent_assignment: dict[str, tuple[str, float]],
    neighborhood_from: dict[str, list[str]],
    mch_performance_models: dict[str, MCHPerformanceModel],
) -> dict[str, MetroPerformanceWithMCH]:
    """Load per-metro performance models in parallel using Python threads."""

    if not metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    models: dict[str, MetroPerformanceWithMCH] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_metro = {
            executor.submit(
                build_metro_performance_model,
                metro,
                descriptors,
                airport_to_metro,
                parent_assignment,
                neighborhood_from,
                mch_performance_models,
            ): metro
            for metro in metros
            if metro in descriptors and metro in airport_to_metro and metro in parent_assignment
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            built_metro, model = future.result()
            models[built_metro] = model
            print(f"Loaded metro performance model for {metro}")

    return models


def compute_cost_optimal_point(
    metro: str,
    descriptor: FootprintDescriptor,
    incoming_traffic_mbps: float,
    replication_factor: int,
    is_mch_metro: bool,
    cost_model: CaribouCostCalculator,
) -> tuple[str, dict[str, float | int | CaribouCostBreakdown]]:
    """Compute the minimum-cost cache point for a metro."""

    min_cost = float("inf")
    best_hitrate = 0
    best_disk = 0.0
    best_breakdown: CaribouCostBreakdown | None = None
    for hitrate in range(1, 101):
        disk_required = descriptor.nearest_cache_for_hitrate(hitrate)
        total_cost = compute_replicated_total_cost_model_b(
            cost_model=cost_model,
            total_disk_required_tb=disk_required,
            hitrate_fraction=float(hitrate) / 100.0,
            incoming_traffic_mbps=incoming_traffic_mbps,
            replication_factor=replication_factor,
            is_mch_in_metro=is_mch_metro,
        )

        if best_breakdown is None or total_cost < min_cost:
            min_cost = total_cost
            best_hitrate = hitrate
            best_disk = disk_required
            best_breakdown = cost_model.compute_monthly_cost(
                total_disk_required_tb=disk_required,
                hitrate_fraction=float(hitrate) / 1000.0,
                total_traffic_mbps=incoming_traffic_mbps / max(1, replication_factor),
                free_disk_tb=0.0,
                is_mch_in_metro=is_mch_metro,
            )

    return metro, {
        "disk": best_disk,
        "hitrate": best_hitrate,
        "cost": min_cost,
        "breakdown": best_breakdown,
    }


def load_cost_optimal_points_threaded(
    metros: list[str],
    descriptors: dict[str, FootprintDescriptor],
    incoming_traffic: dict[str, float],
    metro_tiers: dict[str, int],
    airport_to_metro: dict[str, str],
    mch_metros: list[str],
) -> dict[str, dict[str, float | int | CaribouCostBreakdown]]:
    """Compute cost-optimal points for metros in parallel."""

    if not metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    cost_model = CaribouCostCalculator(geo=_GEO, traffic_class=_BUCKET)
    results: dict[str, dict[str, float | int | CaribouCostBreakdown]] = {}

    def _replication_factor_for_metro(metro: str) -> int:
        metro_name = airport_to_metro.get(metro)
        tier = metro_tiers.get(metro_name, 2)   
        if tier == 0:
            return 5
        if tier == 1:
            return 3
        return 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_metro = {
            executor.submit(
                compute_cost_optimal_point,
                metro,
                descriptors[metro],
                incoming_traffic.get(metro, 0.0),
                _replication_factor_for_metro(metro),
                metro in mch_metros,
                cost_model,
            ): metro
            for metro in metros
            if metro in descriptors
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            built_metro, result = future.result()
            results[built_metro] = result
            print(
                f"Cost-optimal point for {metro}: disk={result['disk']}, "
                f"hitrate={result['hitrate']}%, cost={result['cost']:.2f}"
            )

    return results


def replication_factor_for_metro(
    metro: str,
    metro_tiers: dict[str, int],
    airport_to_metro: dict[str, str],
) -> int:
    metro_name = airport_to_metro.get(metro)
    tier = metro_tiers.get(metro_name, 2)
    if tier == 0:
        return 5
    if tier == 1:
        return 3
    return 2


def compute_replicated_total_cost_model_b(
    *,
    cost_model: CaribouCostCalculator,
    total_disk_required_tb: float,
    hitrate_fraction: float,
    incoming_traffic_mbps: float,
    replication_factor: int,
    is_mch_in_metro: bool,
) -> float:
    """Model B: fixed replica costs scale with replicas, traffic costs use split traffic."""

    effective_replication_factor = max(1, replication_factor)
    split_traffic_mbps = incoming_traffic_mbps / effective_replication_factor
    per_replica_breakdown = cost_model.compute_monthly_cost(
        total_disk_required_tb=total_disk_required_tb,
        hitrate_fraction=hitrate_fraction,
        total_traffic_mbps=split_traffic_mbps,
        free_disk_tb=0.0,
        is_mch_in_metro=is_mch_in_metro,
    )

    fixed_cost = per_replica_breakdown.depreciation_cost + per_replica_breakdown.colocation_cost
    traffic_cost = per_replica_breakdown.midgress_cost + per_replica_breakdown.parent_service_cost
    return (effective_replication_factor * fixed_cost) + (effective_replication_factor * traffic_cost)


def compute_performance_for_metro(
    metro: str,
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    try_hitrates: dict[str, float],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    conn: Convolution,
    step_us: int = 10,
) -> tuple[float, float]:
    ttfb_pdfs = []
    weights = []

    for to_metro in neighborhood_to.get(metro, []):
        if to_metro not in performance_models:
            continue

        hitrate = try_hitrates.get(to_metro, 0.0)
        pdf = performance_models[to_metro].get_ttfb_pdf(
            from_metro=metro,
            hitrate=hitrate,
            conn=conn,
        )
        ttfb_pdfs.append(pdf.to_microsecond_pdf(step_us=step_us))
        weights.append(traffic_lookup_by_airport.get((metro, to_metro), 0.0))

    if not ttfb_pdfs or not any(weight > 0 for weight in weights):
        return 0.0, 0.0

    try:
        combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
    except ValueError:
        return 0.0, 0.0
    p50 = combined_pdf.millisecond_at_percentile(50) / 1000.0
    p95 = combined_pdf.millisecond_at_percentile(95) / 1000.0
    return p50, p95


_EMEA_AIRPORT_REGION:   dict[str, str] = {}
_EMEA_AIRPORT_COUNTRY:  dict[str, str] = {}

def _load_emea_airport_regions() -> tuple[dict[str, str], dict[str, str]]:
    global _EMEA_AIRPORT_REGION, _EMEA_AIRPORT_COUNTRY
    if _EMEA_AIRPORT_REGION:
        return _EMEA_AIRPORT_REGION, _EMEA_AIRPORT_COUNTRY
    csv_path = _BASE_DIR / "PERF" / "emea_airports.csv"
    if not csv_path.exists():
        return _EMEA_AIRPORT_REGION, _EMEA_AIRPORT_COUNTRY
    with csv_path.open(encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = [p.strip() for p in line.strip().split(',')]
            if len(parts) >= 4:
                code, city, country, region = parts[0], parts[1], parts[2], parts[3]
                _EMEA_AIRPORT_REGION[code]  = region
                _EMEA_AIRPORT_COUNTRY[code] = country
    return _EMEA_AIRPORT_REGION, _EMEA_AIRPORT_COUNTRY

# Penalty thresholds by EMEA sub-region: (p50_target_ms, p95_target_ms)
_EMEA_REGION_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Europe":      (24.0,  105.0),
    "Middle East": (45.0,  180.0),
    "Africa":      (75.0,  220.0),
}
# Country-level overrides (applied before the region lookup)
_EMEA_COUNTRY_THRESHOLDS: dict[str, tuple[float, float]] = {
    "South Africa": (35.0, 200.0),
}
_DEFAULT_THRESHOLDS = (24.0, 105.0)

def penalty_function(p50: float, p95: float, traffic_gbps: float = 1.0, metro: str = "") -> float:
    if _GEO == "EMEA" and metro:
        regions, countries = _load_emea_airport_regions()
        country = countries.get(metro)
        if country and country in _EMEA_COUNTRY_THRESHOLDS:
            p50_target, p95_target = _EMEA_COUNTRY_THRESHOLDS[country]
        else:
            region = regions.get(metro)
            p50_target, p95_target = _EMEA_REGION_THRESHOLDS.get(region, _DEFAULT_THRESHOLDS) if region else _DEFAULT_THRESHOLDS
    else:
        p50_target, p95_target = _DEFAULT_THRESHOLDS
    return 2 * traffic_gbps * (max(p50 - p50_target, 0) ** 2) + 2 * traffic_gbps * (max(p95 - p95_target, 0))

def compute_perf_penalty_for_metro(
    metro: str,
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    try_hitrates: dict[str, float],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    conn: Convolution,
    traffic_from: dict[str, float],
) -> dict[str, float]:
    new_perf_penalty: dict[str, float] = {}
    for candidate_metro in neighborhood_to.get(metro, []):
        if candidate_metro not in performance_models:
            continue
        p50, p95 = compute_performance_for_metro(
            candidate_metro,
            neighborhood_to,
            performance_models,
            try_hitrates,
            traffic_lookup_by_airport,
            conn,
        )
        traffic_gbps = traffic_from.get(candidate_metro, 0.0) / 1000.0
        new_perf_penalty[candidate_metro] = penalty_function(p50, p95, traffic_gbps, metro=candidate_metro)
    return new_perf_penalty


def compute_gradient_for_metro(
    metro: str,
    gradient_step: float,
    descriptors: dict[str, FootprintDescriptor],
    disk_provisioned: dict[str, float],
    try_hitrates: dict[str, float],
    cost_by_metro: dict[str, float],
    perf_penalty_by_metro: dict[str, float],
    incoming_traffic: dict[str, float],
    metro_tiers: dict[str, int],
    airport_to_metro: dict[str, str],
    mch_metros: list[str],
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    conn: Convolution,
    cost_model: CaribouCostCalculator,
    traffic_from: dict[str, float],
) -> tuple[str, dict[str, float]]:
    if metro not in descriptors:
        return metro, {
            "cost_gradient": 0.0,
            "perf_gradient": 0.0,
            "overall_gradient": 0.0,
        }

    descriptor = descriptors[metro]
    current_disk = disk_provisioned.get(metro, 0.0)
    current_hitrate = min(max(try_hitrates.get(metro, descriptor.hitrate_for_cache(current_disk)), 0.0), 100.0)
    current_cost = cost_by_metro.get(metro, 0.0)

    replication_factor = replication_factor_for_metro(metro, metro_tiers, airport_to_metro)
    current_incoming_traffic = incoming_traffic.get(metro, 0.0)

    max_cache_space = max(point.cache_space for point in descriptor._points_sorted_by_cache)
    new_disk = min(current_disk + gradient_step, max_cache_space)
    effective_step = new_disk - current_disk
    if effective_step <= 0:
        return metro, {
            "cost_gradient": 0.0,
            "perf_gradient": 0.0,
            "overall_gradient": 0.0,
        }

    new_hitrate = min(max(descriptor.hitrate_for_cache(new_disk), 0.0), 100.0)
    increased_try_hitrates = dict(try_hitrates)
    increased_try_hitrates[metro] = new_hitrate

    increased_cost = compute_replicated_total_cost_model_b(
        cost_model=cost_model,
        total_disk_required_tb=new_disk,
        hitrate_fraction=new_hitrate / 100.0,
        incoming_traffic_mbps=current_incoming_traffic,
        replication_factor=replication_factor,
        is_mch_in_metro=metro in mch_metros,
    )
    cost_differential = increased_cost - current_cost
    increased_perf_penalty = compute_perf_penalty_for_metro(
        metro,
        neighborhood_to,
        performance_models,
        increased_try_hitrates,
        traffic_lookup_by_airport,
        conn,
        traffic_from,
    )
    perf_differential = sum(
        increased_perf_penalty.get(candidate_metro, 0.0) - perf_penalty_by_metro.get(candidate_metro, 0.0)
        for candidate_metro in increased_perf_penalty
    )

    cost_gradient = cost_differential 
    perf_gradient = perf_differential 
    gradient = cost_gradient + perf_gradient
    return metro, {
        "cost_gradient": cost_gradient,
        "perf_gradient": perf_gradient,
        "overall_gradient": gradient,
    }


def load_gradients_threaded(
    metros: list[str],
    gradient_step: float,
    descriptors: dict[str, FootprintDescriptor],
    disk_provisioned: dict[str, float],
    try_hitrates: dict[str, float],
    cost_by_metro: dict[str, float],
    perf_penalty_by_metro: dict[str, float],
    incoming_traffic: dict[str, float],
    metro_tiers: dict[str, int],
    airport_to_metro: dict[str, str],
    mch_metros: list[str],
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    traffic_from: dict[str, float],
) -> dict[str, dict[str, float]]:
    if not metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    cost_model = CaribouCostCalculator(geo=_GEO, traffic_class=_BUCKET)
    gradients: dict[str, dict[str, float]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_metro = {
            executor.submit(
                compute_gradient_for_metro,
                metro,
                gradient_step,
                descriptors,
                disk_provisioned,
                try_hitrates,
                cost_by_metro,
                perf_penalty_by_metro,
                incoming_traffic,
                metro_tiers,
                airport_to_metro,
                mch_metros,
                neighborhood_to,
                performance_models,
                traffic_lookup_by_airport,
                Convolution(),
                cost_model,
                traffic_from,
            ): metro
            for metro in metros
            if metro in descriptors and metro in performance_models
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            built_metro, gradient_components = future.result()
            gradients[built_metro] = gradient_components
            print(
                f"Gradient for metro {metro}: cost={gradient_components['cost_gradient']}, "
                f"perf={gradient_components['perf_gradient']}, "
                f"overall={gradient_components['overall_gradient']}"
            )

    return gradients


def evaluate_state(
    metros: list[str],
    disk_provisioned: dict[str, float],
    descriptors: dict[str, FootprintDescriptor],
    incoming_traffic: dict[str, float],
    metro_tiers: dict[str, int],
    airport_to_metro: dict[str, str],
    mch_metros: list[str],
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    traffic_from: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    cost_model = CaribouCostCalculator(geo=_GEO, traffic_class=_BUCKET)
    try_hitrates: dict[str, float] = {}
    cost_by_metro: dict[str, float] = {}
    perf_penalty: dict[str, float] = {}
    performance_stats: dict[str, tuple[float, float]] = {}

    for metro in metros:
        if metro not in descriptors:
            continue
        descriptor = descriptors[metro]
        disk = disk_provisioned.get(metro, 0.0)
        try_hitrates[metro] = min(max(descriptor.hitrate_for_cache(disk), 0.0), 100.0)

        replication_factor = replication_factor_for_metro(metro, metro_tiers, airport_to_metro)
        cost_by_metro[metro] = compute_replicated_total_cost_model_b(
            cost_model=cost_model,
            total_disk_required_tb=disk,
            hitrate_fraction=try_hitrates[metro] / 100.0,
            incoming_traffic_mbps=incoming_traffic.get(metro, 0.0),
            replication_factor=replication_factor,
            is_mch_in_metro=metro in mch_metros,
        )

    conn = Convolution()
    for metro in metros:
        if metro not in performance_models:
            continue
        p50, p95 = compute_performance_for_metro(
            metro,
            neighborhood_to,
            performance_models,
            try_hitrates,
            traffic_lookup_by_airport,
            conn,
        )
        performance_stats[metro] = (p50, p95)
        perf_penalty[metro] = penalty_function(p50, p95, traffic_from.get(metro, 0.0) / 1000.0, metro=metro)

    return try_hitrates, cost_by_metro, perf_penalty, performance_stats

def plot_neighborhood_graph(neighborhood: dict[str, list[str]], airport_info: dict[str, dict], traffic_lookup: dict[tuple[str, str], float]):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    # Add nodes with positions based on latitude and longitude
    for metro in neighborhood:
        if metro not in airport_info:
            continue
        lat = airport_info[metro]['latitude']
        lon = airport_info[metro]['longitude']
        G.add_node(metro, pos=(lon, lat))

    # Add edges with weights based on traffic
    for metro, neighbors in neighborhood.items():
        for neighbor in neighbors:
            traffic = traffic_lookup.get((metro, neighbor), 0.0)
            if traffic > 30000:  # Only show edges with > 30 Gbps
                G.add_edge(metro, neighbor, weight=traffic)

    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(16, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Draw edges with width proportional to traffic
    if weights:
        max_weight = max(weights.values())
        edge_widths = [5 * (w / max_weight) for w in weights.values()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                               edge_color='gray', arrows=True, arrowsize=10)
        
        # Draw edge labels showing traffic in Gbps
        edge_labels = {edge: f"{weight/1000:.1f}G" for edge, weight in weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(f"Traffic Graph: End-User Metros → Serving Metros ({_BUCKET}, edges > 10 Gbps)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = _BASE_DIR / "traffic_graph.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Traffic graph saved to {output_path}")
    plt.show()

def log_iteration_state(
    iteration: int,
    metros: list[str],
    disk_provisioned: dict[str, float],
    hitrates: dict[str, float],
    cost_by_metro: dict[str, float],
    perf_penalty: dict[str, float],
    performance_stats: dict[str, tuple[float, float]],
    gradients_by_metro: dict[str, dict[str, float]],
    total_cost: float,
    total_penalty: float,
    combined_total: float,
    per_metro_log_path: Path,
    summary_log_path: Path,
) -> None:
    with per_metro_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"iteration={iteration}\n")
        for metro in metros:
            if metro not in disk_provisioned:
                continue
            p50, p95 = performance_stats.get(metro, (0.0, 0.0))
            gradient_components = gradients_by_metro.get(
                metro,
                {"cost_gradient": 0.0, "perf_gradient": 0.0, "overall_gradient": 0.0},
            )
            handle.write(
                f"{metro}, disk_mb={disk_provisioned.get(metro, 0.0):.2f}, "
                f"hitrate={hitrates.get(metro, 0.0):.2f}, "
                f"cost={cost_by_metro.get(metro, 0.0):.2f}, "
                f"perf_penalty={perf_penalty.get(metro, 0.0):.2f}, "
                f"p50={p50:.4f}, p95={p95:.4f}, "
                f"cost_gradient={gradient_components['cost_gradient']:.4f}, "
                f"perf_gradient={gradient_components['perf_gradient']:.4f}, "
                f"overall_gradient={gradient_components['overall_gradient']:.4f}\n"
            )
        handle.write("\n")

    with summary_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"iteration={iteration}, total_cost={total_cost:.2f}, "
            f"total_perf_penalty={total_penalty:.2f}, objective={combined_total:.2f}\n"
        )


if __name__ == "__main__":
    args = get_args()
    _GEO = args.geo
    _BUCKET = args.bucket
    _TRAFFIC_THRESHOLD = args.traffic_threshold
    _FDS2_DIR = _BASE_DIR / f"FDS_{_BUCKET}"

    airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(_BASE_DIR / "PERF" / "metro_areas.csv", _GEO)
    metro_tiers = get_metro_tiers()
    parent_assignment = assign_parent_metros(airport_info)

    traffic_lookup = _load_traffic_lookup()

    # asn_metro is the end user metro and bw_metro is where the traffic is coming from
    # For each metro in all metros, build a neighborhood list of metros. Store it in a dictionary.
    # A metro is in the neighborhood if this metro sends more than 2Gbps of traffic to the neighbor.
    neighborhood_to = defaultdict(list)
    neighborhood_from = defaultdict(list)
    for (asn_metro, bw_metro), traffic in traffic_lookup.items():
        if traffic > _TRAFFIC_THRESHOLD and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
            neighborhood_to[metro_to_airport[asn_metro]].append(metro_to_airport[bw_metro])
            neighborhood_from[metro_to_airport[bw_metro]].append(metro_to_airport[asn_metro])

    ## Update the traffic lookup to be keyed by airport codes instead of metro names
    traffic_lookup_by_airport = {}
    for (asn_metro, bw_metro), traffic in traffic_lookup.items():
        if asn_metro in metro_to_airport and bw_metro in metro_to_airport:
            if traffic > _TRAFFIC_THRESHOLD:
                print(f"ASN Metro: {asn_metro}, BW Metro: {bw_metro}, Traffic: {traffic} Mbps")
            traffic_lookup_by_airport[(metro_to_airport[asn_metro], metro_to_airport[bw_metro])] = traffic

    metros = list(airport_info.keys())
    INCOMING_TRAFFIC = defaultdict(float)
    TRAFFIC_FROM = defaultdict(float)

    for metro in metros:
        for from_metro in neighborhood_from.get(metro, []):
            traffic = traffic_lookup_by_airport.get((from_metro, metro), 0.0)
            INCOMING_TRAFFIC[metro] += traffic
            TRAFFIC_FROM[from_metro] += traffic

    total_traffic = sum(INCOMING_TRAFFIC[metro] for metro in metros)

    print("Incoming traffic for each metro (in Mbps):")
    for metro in metros:
        if INCOMING_TRAFFIC[metro] > 0:
            print(f"{metro}: {INCOMING_TRAFFIC[metro]} Mbps")

    print("Traffic served from each metro (in Mbps):")
    for metro in metros:
        if TRAFFIC_FROM[metro] > 0:
            print(f"{metro}: {TRAFFIC_FROM[metro]} Mbps")

    if total_traffic > 0:
        print("Traffic fraction for each metro:")
        for metro in metros:
            if INCOMING_TRAFFIC[metro] > 0:
                print(f"{metro}: {INCOMING_TRAFFIC[metro] / total_traffic}")
    
    MCH_PERFORMANCE_MODELS = load_mch_performance_models_threaded(MCH_METROS, airport_to_metro)
    print(f"Loaded {len(MCH_PERFORMANCE_MODELS)} MCH performance models using up to {max(1, os.cpu_count() or 1)} threads")

    fds_metros = []
    for fd_path in _FDS2_DIR.glob("*.txt"):
        metro = fd_path.stem.upper()
        if metro in airport_info:
            fds_metros.append((metro, airport_info[metro]))
    print(f"Metros available in FDS2: {[metro for metro, _ in fds_metros]}")

    FDS_BY_METRO = load_smoothed_fds_for_all_metros_threaded(
        ALL_METROS,
        airport_info,
        metro_tiers,
        fds_metros,
    )
    print(f"Loaded {len(FDS_BY_METRO)} smoothed footprint descriptors using up to {max(1, os.cpu_count() or 1)} threads")

    PERFORMANCE_MODELS = load_performance_models_threaded(
        ALL_METROS,
        FDS_BY_METRO,
        airport_to_metro,
        parent_assignment,
        neighborhood_from,
        MCH_PERFORMANCE_MODELS,
    )
    print(f"Loaded {len(PERFORMANCE_MODELS)} metro performance models using up to {max(1, os.cpu_count() or 1)} threads")

    COST_OPTIMAL_POINTS = load_cost_optimal_points_threaded(
        metros,
        FDS_BY_METRO,
        INCOMING_TRAFFIC,
        metro_tiers,
        airport_to_metro,
        MCH_METROS,
    )
    print(f"Computed {len(COST_OPTIMAL_POINTS)} cost-optimal points using up to {max(1, os.cpu_count() or 1)} threads")
    # print("\nCost-optimal points for each metro:")
    # for metro, result in COST_OPTIMAL_POINTS.items():
    #     disk = result["disk"]
    #     hitrate = result["hitrate"]
    #     cost = result["cost"]
    #     print(f"{metro}: Optimal Disk = {disk:.2f} MB, Optimal Hitrate = {hitrate}%, Total Cost = ${cost:.2f}, Incoming Traffic = {INCOMING_TRAFFIC[metro]:.2f} Mbps, Traffic From = {TRAFFIC_FROM[metro]:.2f} Mbps, Traffic Fraction = {(INCOMING_TRAFFIC[metro] / total_traffic * 100) if total_traffic > 0 else 0:.2f}%")

    # DISK_PROVISIONED = {metro: float(COST_OPTIMAL_POINTS[metro]["disk"]) for metro in COST_OPTIMAL_POINTS}
    # HITRATES = {metro: float(COST_OPTIMAL_POINTS[metro]["hitrate"]) for metro in COST_OPTIMAL_POINTS}
    # TRY_HITRATES = dict(HITRATES)
    # COST = {metro: float(COST_OPTIMAL_POINTS[metro]["cost"]) for metro in COST_OPTIMAL_POINTS}

    # print("\nMinimum-disk starting point for each metro:")
    # DISK_PROVISIONED = {}
    # for metro, descriptor in FDS_BY_METRO.items():
    #     if not descriptor._points_sorted_by_cache:
    #         continue
    #     min_disk = min(point.cache_space for point in descriptor._points_sorted_by_cache)
    #     DISK_PROVISIONED[metro] = float(min_disk)
    #     print(
    #         f"{metro}: Minimum Disk = {min_disk:.2f} MB, "
    #         f"Incoming Traffic = {INCOMING_TRAFFIC[metro]:.2f} Mbps, "
    #         f"Traffic From = {TRAFFIC_FROM[metro]:.2f} Mbps, "
    #         f"Traffic Fraction = {(INCOMING_TRAFFIC[metro] / total_traffic * 100) if total_traffic > 0 else 0:.2f}%"
    #     )

    print("\nCost-optimal starting point for each metro:")
    DISK_PROVISIONED = {metro: float(COST_OPTIMAL_POINTS[metro]["disk"]) for metro in COST_OPTIMAL_POINTS}
    HITRATES = {metro: float(COST_OPTIMAL_POINTS[metro]["hitrate"]) for metro in COST_OPTIMAL_POINTS}
    TRY_HITRATES = dict(HITRATES)
    COST = {metro: float(COST_OPTIMAL_POINTS[metro]["cost"]) for metro in COST_OPTIMAL_POINTS}
    for metro, result in COST_OPTIMAL_POINTS.items():
        print(
            f"{metro}: Disk = {result['disk']:.2f} MB, Hitrate = {result['hitrate']:.2f}%, "
            f"Cost = ${result['cost']:.2f}, Incoming Traffic = {INCOMING_TRAFFIC[metro]:.2f} Mbps"
        )

    PERF_PENALTY = defaultdict(float)

    conn = Convolution()
    cost_optimal_rows = []
    for metro in metros:
        if metro not in PERFORMANCE_MODELS:
            continue
        p50, p95 = compute_performance_for_metro(
            metro,
            neighborhood_to,
            PERFORMANCE_MODELS,
            TRY_HITRATES,
            traffic_lookup_by_airport,
            conn,
        )
        PERF_PENALTY[metro] = penalty_function(p50, p95, TRAFFIC_FROM[metro] / 1000.0, metro=metro)
        print(f"Initial metro {metro}: P50={p50} ms, P95={p95} ms, penalty={PERF_PENALTY[metro]:.2f}")
        cost_optimal_rows.append({
            "metro": metro,
            "disk_mb": DISK_PROVISIONED.get(metro, 0.0),
            "hitrate": TRY_HITRATES.get(metro, 0.0),
            "cost": COST.get(metro, 0.0),
            "p50_ms": p50,
            "p95_ms": p95,
            "perf_penalty": PERF_PENALTY[metro],
        })

    cost_optimal_log_path = _BASE_DIR / f"cost_optimal_starting_points_{_GEO}_{_BUCKET}.csv"
    with open(cost_optimal_log_path, "w") as f:
        f.write("metro,disk_mb,hitrate,cost,p50_ms,p95_ms,perf_penalty\n")
        for row in cost_optimal_rows:
            f.write(
                f"{row['metro']},{row['disk_mb']:.2f},{row['hitrate']:.2f},"
                f"{row['cost']:.2f},{row['p50_ms']:.4f},{row['p95_ms']:.4f},{row['perf_penalty']:.2f}\n"
            )
    print(f"\nCost-optimal starting points written to {cost_optimal_log_path}")


    gradient_step = 5 * 1000 * 1000  # 5 TB in MB — used only for gradient probing, not for the update step
    TB_IN_MB = 1_000_000  # 1 TB expressed in MB
    PER_METRO_STEP_TB = 10  # TB per metro used to size the budget
    active_metros = [metro for metro in metros if metro in FDS_BY_METRO and metro in PERFORMANCE_MODELS]
    per_metro_log_path = _BASE_DIR / "gradient_descent_metro_log.txt"
    summary_log_path = _BASE_DIR / "gradient_descent_summary_log.txt"

    per_metro_log_path.write_text("", encoding="utf-8")
    summary_log_path.write_text("", encoding="utf-8")

    '''
    # Log iteration 0: the cost-optimal starting point (before any gradient steps).
    print("Evaluating iteration 0 (cost-optimal starting point) for logging...")
    _, _, _, perf_stats_0 = evaluate_state(
        active_metros,
        DISK_PROVISIONED,
        FDS_BY_METRO,
        INCOMING_TRAFFIC,
        metro_tiers,
        airport_to_metro,
        MCH_METROS,
        neighborhood_to,
        PERFORMANCE_MODELS,
        traffic_lookup_by_airport,
        TRAFFIC_FROM,
    )
    zero_gradients: dict[str, dict[str, float]] = {
        m: {"cost_gradient": 0.0, "perf_gradient": 0.0, "overall_gradient": 0.0}
        for m in active_metros
    }
    total_cost_0 = sum(COST.get(m, 0.0) for m in active_metros)
    total_penalty_0 = sum(PERF_PENALTY.get(m, 0.0) for m in active_metros)
    log_iteration_state(
        0,
        active_metros,
        DISK_PROVISIONED,
        TRY_HITRATES,
        COST,
        PERF_PENALTY,
        perf_stats_0,
        zero_gradients,
        total_cost_0,
        total_penalty_0,
        total_cost_0 + total_penalty_0,
        per_metro_log_path,
        summary_log_path,
    )
    print(f"Logged iteration 0: total_cost={total_cost_0:.2f}, total_penalty={total_penalty_0:.2f}")
    '''
    iteration = 0
    while True:
        iteration += 1
        gradients = load_gradients_threaded(
            active_metros,
            gradient_step,
            FDS_BY_METRO,
            DISK_PROVISIONED,
            TRY_HITRATES,
            COST,
            PERF_PENALTY,
            INCOMING_TRAFFIC,
            metro_tiers,
            airport_to_metro,
            MCH_METROS,
            neighborhood_to,
            PERFORMANCE_MODELS,
            traffic_lookup_by_airport,
            TRAFFIC_FROM,
        )

        # --- Proportional budget distribution ---
        # 1. Collect signed gradients and their absolute values
        grad_values = {m: gradients.get(m, {}).get("overall_gradient", 0.0) for m in active_metros}
        abs_grads = {m: abs(g) for m, g in grad_values.items()}
        total_abs_grad = sum(abs_grads.values())

        # 2. Compute budget for this iteration:
        #    budget = num metros with negative gradient * PER_METRO_STEP_TB TB
        #    (first iteration: all metros count; subsequent iterations: only negative-gradient metros)
        num_negative = sum(1 for g in grad_values.values() if g < 0)
        budget_mb = max(num_negative, 1) * PER_METRO_STEP_TB * TB_IN_MB

        # 3. Distribute budget proportionally, round each delta up to the nearest TB
        updated_disk_provisioned = dict(DISK_PROVISIONED)
        for metro in active_metros:
            g = grad_values[metro]
            if g == 0.0 or total_abs_grad == 0.0:
                continue
            descriptor = FDS_BY_METRO[metro]
            current_disk = DISK_PROVISIONED.get(metro, 0.0)
            min_cache_space = min(point.cache_space for point in descriptor._points_sorted_by_cache)
            max_cache_space = max(point.cache_space for point in descriptor._points_sorted_by_cache)

            weight = abs_grads[metro] / total_abs_grad
            raw_delta_mb = weight * budget_mb
            # Round up to nearest TB
            delta_mb = math.ceil(raw_delta_mb / TB_IN_MB) * TB_IN_MB

            if g < 0:
                # Negative gradient: increasing disk improves objective → add disk
                new_disk = min(current_disk + delta_mb, max_cache_space)
            else:
                # Positive gradient: increasing disk hurts objective → remove disk
                new_disk = max(current_disk - delta_mb, min_cache_space)

            updated_disk_provisioned[metro] = new_disk

        DISK_PROVISIONED = updated_disk_provisioned
        TRY_HITRATES, COST, PERF_PENALTY, performance_stats = evaluate_state(
            active_metros,
            DISK_PROVISIONED,
            FDS_BY_METRO,
            INCOMING_TRAFFIC,
            metro_tiers,
            airport_to_metro,
            MCH_METROS,
            neighborhood_to,
            PERFORMANCE_MODELS,
            traffic_lookup_by_airport,
            TRAFFIC_FROM,
        )
        HITRATES = dict(TRY_HITRATES)

        total_cost = sum(COST.get(metro, 0.0) for metro in active_metros)
        total_penalty = sum(PERF_PENALTY.get(metro, 0.0) for metro in active_metros)
        combined_total = total_cost + total_penalty

        log_iteration_state(
            iteration,
            active_metros,
            DISK_PROVISIONED,
            HITRATES,
            COST,
            PERF_PENALTY,
            performance_stats,
            gradients,
            total_cost,
            total_penalty,
            combined_total,
            per_metro_log_path,
            summary_log_path,
        )

        print(f"Iteration {iteration}")
        for metro in active_metros:
            p50, p95 = performance_stats.get(metro, (0.0, 0.0))
            gradient_components = gradients.get(
                metro,
                {"cost_gradient": 0.0, "perf_gradient": 0.0, "overall_gradient": 0.0},
            )
            print(
                f"{metro}: disk={DISK_PROVISIONED.get(metro, 0.0):.5f} MB, "
                f"hitrate={HITRATES.get(metro, 0.0):.5f}%, "
                f"cost={COST.get(metro, 0.0):.5f}, "
                f"perf_penalty={PERF_PENALTY.get(metro, 0.0):.5f}, "
                f"p50={p50:.5f} ms, p95={p95:.5f} ms, "
                f"cost_gradient={gradient_components['cost_gradient']:.5f}, "
                f"perf_gradient={gradient_components['perf_gradient']:.5f}, "
                f"overall_gradient={gradient_components['overall_gradient']:.5f}"
            )
        print(
            f"Overall total cost={total_cost:.5f}, total performance penalty={total_penalty:.5f}, "
            f"objective={combined_total:.5f} (cost + penalty)"
        )
