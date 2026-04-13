# Wohoo, we are going to discover the disk needs of each metro in the US! We are going to be using multithreading to do it!
# Let's begin. from __future__ import annotations

from pathlib import Path
import random
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt

from cost import CaribouCostCalculator, CaribouCostBreakdown
from fds import FootprintDescriptor
from perf_with_mch import MetroPerformanceWithMCH, MCHPerformanceModel
from probability import Convolution, gaussian_pdf, weighted_pdf_sum
from analyse import *
from itertools import product
from collections import defaultdict

import sys

INT32_MAX = 2**31 - 1

_BASE_DIR = Path(__file__).resolve().parent
_FDS_DIR = _BASE_DIR / "FDS"
_FDS2_DIR = _BASE_DIR / "FDS2"
_COST_PERF_DIR = _BASE_DIR / "COST-PERF"

FDS_EXCEPTIONS = { "LGA" : "EWR_LGA"}

MCH_METROS = ["ORD", "DFW", "LGA", "IAD", "ATL", "MIA", "SEA", "SJC", "LAX", "BOS", "DEN"]

ALL_METROS = []

def parse_metro_areas(file_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    airport_info = {}
    metro_to_airport = {}
    airport_to_metro = {}

    with file_path.open() as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 8:
                continue  # Skip malformed lines

            # Strip the quotes from each part
            parts = [part.strip().strip('"') for part in parts]

            id, metro_area, latitude, longitude, airport_code, country, state, max_distance = parts

            if country != "US":
                continue  # Skip non-US entries

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
    csv_path = _BASE_DIR / "SERVEDFROM_DATA" / "served_from.csv"
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
        while fallback_metro is not None and fallback_metro not in visited:
            visited.add(fallback_metro)
            fd_text = _read_fd_text_for_metro(fallback_metro)
            if fd_text is not None:
                source_metro = fallback_metro
                break
            fallback_metro = get_nearest_metro(airport_info, fallback_metro, fds_metros)

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
    effective_incoming_traffic_mbps = incoming_traffic_mbps / replication_factor if replication_factor > 0 else incoming_traffic_mbps

    for hitrate in range(1, 101):
        disk_required = descriptor.nearest_cache_for_hitrate(hitrate)
        breakdown = cost_model.compute_monthly_cost(
            total_disk_required_tb=disk_required,
            hitrate_fraction=float(hitrate) / 100.0,
            total_traffic_mbps=effective_incoming_traffic_mbps,
            free_disk_tb=0.0,
            is_mch_in_metro=is_mch_metro,
        )

        if best_breakdown is None or breakdown.total_cost < min_cost:
            min_cost = breakdown.total_cost
            best_hitrate = hitrate
            best_disk = disk_required
            best_breakdown = breakdown

    return metro, {
        "disk": best_disk,
        "hitrate": best_hitrate,
        "cost": min_cost,
        "breakdown": best_breakdown,
    }


def compute_same_metro_ttfb_for_hitrate(
    metro: str,
    hitrate: float,
    performance_models: dict[str, MetroPerformanceWithMCH],
    conn: Convolution,
) -> tuple[float, float]:
    pdf = performance_models[metro].get_ttfb_pdf(
        from_metro=metro,
        hitrate=hitrate,
        conn=conn,
    )
    combined_pdf = pdf.to_microsecond_pdf()
    p50 = combined_pdf.millisecond_at_percentile(50) / 100
    p95 = combined_pdf.millisecond_at_percentile(95) / 100
    return p50, p95


def disk_sweep_values(descriptor: FootprintDescriptor, step_fraction: float = 0.01) -> list[float]:
    min_disk_mb = min(point.cache_space for point in descriptor._points_sorted_by_cache)
    max_disk_mb = max(point.cache_space for point in descriptor._points_sorted_by_cache)
    step_mb = 1000000
    values: list[float] = []
    current_disk_mb = 0
    while current_disk_mb <= max_disk_mb:
        values.append(current_disk_mb)
        next_disk_mb = min(current_disk_mb + step_mb, max_disk_mb)
        if next_disk_mb == current_disk_mb:
            break
        current_disk_mb = next_disk_mb

    if values and values[-1] < max_disk_mb:
        values.append(max_disk_mb)

    return values


def compute_cost_performance_curve_for_metro(
    metro: str,
    descriptor: FootprintDescriptor,
    in_metro_traffic_mbps: float,
    replication_factor: int,
    is_mch_metro: bool,
    performance_models: dict[str, MetroPerformanceWithMCH],
    conn: Convolution,
    cost_model: CaribouCostCalculator,
) -> tuple[str, list[dict[str, float]]]:
    """Compute cost/performance tradeoff points using same-metro traffic only."""

    curve_points: list[dict[str, float]] = []
    effective_in_metro_traffic_mbps = in_metro_traffic_mbps / replication_factor if replication_factor > 0 else in_metro_traffic_mbps

    if effective_in_metro_traffic_mbps <= 0:
        return metro, curve_points

    for disk_required in disk_sweep_values(descriptor):
        hitrate = descriptor.hitrate_for_cache(disk_required)
        try:
            p50, p95 = compute_same_metro_ttfb_for_hitrate(
                metro,
                float(hitrate),
                performance_models,
                conn,
            )
        except KeyError as exc:
            #print(f"Skipping metro {metro} at disk {disk_required:.2f} MB due to missing performance input: {exc}")
            continue

        breakdown = cost_model.compute_monthly_cost(
            total_disk_required_tb=disk_required,
            hitrate_fraction=float(hitrate) / 100.0,
            total_traffic_mbps=effective_in_metro_traffic_mbps,
            free_disk_tb=0.0,
            is_mch_in_metro=is_mch_metro,
        )

        curve_points.append(
            {
                "hitrate": float(hitrate),
                "disk": float(disk_required),
                "cost": float(breakdown.total_cost),
                "p50": float(p50),
                "p95": float(p95),
            }
        )

    return metro, curve_points


def load_cost_performance_curves_threaded(
    metros: list[str],
    descriptors: dict[str, FootprintDescriptor],
    in_metro_traffic: dict[str, float],
    metro_tiers: dict[str, int],
    airport_to_metro: dict[str, str],
    mch_metros: list[str],
    performance_models: dict[str, MetroPerformanceWithMCH],
) -> dict[str, list[dict[str, float]]]:
    """Compute same-metro cost/performance tradeoff curves in parallel."""

    if not metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    results: dict[str, list[dict[str, float]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_metro = {
            executor.submit(
                compute_cost_performance_curve_for_metro,
                metro,
                descriptors[metro],
                in_metro_traffic.get(metro, 0.0),
                replication_factor_for_metro(metro, metro_tiers, airport_to_metro),
                metro in mch_metros,
                performance_models,
                Convolution(),
                CaribouCostCalculator(),
            ): metro
            for metro in metros
            if metro in descriptors and metro in performance_models
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            built_metro, curve_points = future.result()
            results[built_metro] = curve_points
            print(f"Computed {len(curve_points)} cost-performance points for {metro}")

    return results


def plot_cost_performance_curve(metro: str, curve_points: list[dict[str, float]], output_dir: Path) -> None:
    """Save cost-vs-P50 and cost-vs-P95 plots for one metro."""

    if not curve_points:
        print(f"Skipping plots for {metro}: no valid cost-performance points were available")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cost_values = [point["cost"] for point in curve_points]
    p50_values = [point["p50"] for point in curve_points]
    p95_values = [point["p95"] for point in curve_points]

    plt.figure()
    plt.plot(p50_values, cost_values, marker="o")
    plt.title(f"Cost vs P50 TTFB for {metro}")
    plt.xlabel("P50 TTFB (ms)")
    plt.ylabel("Monthly cost (USD)")
    plt.grid(True)
    plt.tight_layout()
    p50_output_path = output_dir / f"cost_vs_p50_{metro}.png"
    plt.savefig(p50_output_path)
    plt.close()
    print(f"Saved plot to {p50_output_path}")

    plt.figure()
    plt.plot(p95_values, cost_values, marker="s")
    plt.title(f"Cost vs P95 TTFB for {metro}")
    plt.xlabel("P95 TTFB (ms)")
    plt.ylabel("Monthly cost (USD)")
    plt.grid(True)
    plt.tight_layout()
    p95_output_path = output_dir / f"cost_vs_p95_{metro}.png"
    plt.savefig(p95_output_path)
    plt.close()
    print(f"Saved plot to {p95_output_path}")


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
    cost_model = CaribouCostCalculator()
    results: dict[str, dict[str, float | int | CaribouCostBreakdown]] = {}

    def _replication_factor_for_metro(metro: str) -> int:
        metro_name = airport_to_metro.get(metro)
        tier = metro_tiers.get(metro_name, 2)
        if tier == 0:
            return 7
        if tier == 1:
            return 5
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
        return 7
    if tier == 1:
        return 5
    return 2


def compute_performance_for_metro(
    metro: str,
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    try_hitrates: dict[str, float],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    conn: Convolution,
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
        ttfb_pdfs.append(pdf.to_microsecond_pdf())
        weights.append(traffic_lookup_by_airport.get((metro, to_metro), 0.0))

    if not ttfb_pdfs or not any(weight > 0 for weight in weights):
        return 0.0, 0.0

    combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
    p50 = combined_pdf.millisecond_at_percentile(50) / 100
    p95 = combined_pdf.millisecond_at_percentile(95) / 100
    return p50, p95


def penalty_function(p50: float, p95: float) -> float:
    return 50.0 * (max(p50 - 23, 0) ** 2) + 100.0 * (max(p95 - 105, 0))

def compute_perf_penalty_for_metro(
    metro: str,
    neighborhood_to: dict[str, list[str]],
    performance_models: dict[str, MetroPerformanceWithMCH],
    try_hitrates: dict[str, float],
    traffic_lookup_by_airport: dict[tuple[str, str], float],
    conn: Convolution,
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
        new_perf_penalty[candidate_metro] = penalty_function(p50, p95)
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
) -> tuple[str, float]:
    if metro not in descriptors:
        return metro, 0.0

    descriptor = descriptors[metro]
    current_disk = disk_provisioned.get(metro, 0.0)
    current_hitrate = try_hitrates.get(metro, descriptor.hitrate_for_cache(current_disk))
    current_cost = cost_by_metro.get(metro, 0.0)

    replication_factor = replication_factor_for_metro(metro, metro_tiers, airport_to_metro)
    effective_traffic = incoming_traffic.get(metro, 0.0) / replication_factor if replication_factor > 0 else incoming_traffic.get(metro, 0.0)

    max_cache_space = max(point.cache_space for point in descriptor._points_sorted_by_cache)
    new_disk = min(current_disk + gradient_step, max_cache_space)
    new_hitrate = descriptor.hitrate_for_cache(new_disk)
    increased_try_hitrates = dict(try_hitrates)
    increased_try_hitrates[metro] = new_hitrate

    increased_breakdown = cost_model.compute_monthly_cost(
        total_disk_required_tb=new_disk,
        hitrate_fraction=new_hitrate / 100.0,
        total_traffic_mbps=effective_traffic,
        free_disk_tb=0.0,
        is_mch_in_metro=metro in mch_metros,
    )
    cost_differential = increased_breakdown.total_cost - current_cost
    increased_perf_penalty = compute_perf_penalty_for_metro(
        metro,
        neighborhood_to,
        performance_models,
        increased_try_hitrates,
        traffic_lookup_by_airport,
        conn,
    )
    perf_differential = sum(
        increased_perf_penalty.get(candidate_metro, 0.0) - perf_penalty_by_metro.get(candidate_metro, 0.0)
        for candidate_metro in increased_perf_penalty
    )

    previous_disk = max(current_disk - gradient_step, 0.0)
    previous_hitrate = descriptor.hitrate_for_cache(previous_disk)
    decreased_try_hitrates = dict(try_hitrates)
    decreased_try_hitrates[metro] = previous_hitrate
    previous_breakdown = cost_model.compute_monthly_cost(
        total_disk_required_tb=previous_disk,
        hitrate_fraction=previous_hitrate / 100.0,
        total_traffic_mbps=effective_traffic,
        free_disk_tb=0.0,
        is_mch_in_metro=metro in mch_metros,
    )
    cost_differential_prev = previous_breakdown.total_cost - current_cost
    decreased_perf_penalty = compute_perf_penalty_for_metro(
        metro,
        neighborhood_to,
        performance_models,
        decreased_try_hitrates,
        traffic_lookup_by_airport,
        conn,
    )
    perf_differential_prev = sum(
        decreased_perf_penalty.get(candidate_metro, 0.0) - perf_penalty_by_metro.get(candidate_metro, 0.0)
        for candidate_metro in decreased_perf_penalty
    )

    gradient = ((cost_differential - cost_differential_prev) + (perf_differential - perf_differential_prev)) / 2.0
    return metro, gradient


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
) -> dict[str, float]:
    if not metros:
        return {}

    max_workers = max(1, os.cpu_count() or 1)
    cost_model = CaribouCostCalculator()
    gradients: dict[str, float] = {}

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
            ): metro
            for metro in metros
            if metro in descriptors and metro in performance_models
        }

        for future in as_completed(future_to_metro):
            metro = future_to_metro[future]
            built_metro, gradient = future.result()
            gradients[built_metro] = gradient
            print(f"Gradient for metro {metro}: {gradient}")

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
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    cost_model = CaribouCostCalculator()
    try_hitrates: dict[str, float] = {}
    cost_by_metro: dict[str, float] = {}
    perf_penalty: dict[str, float] = {}
    performance_stats: dict[str, tuple[float, float]] = {}

    for metro in metros:
        if metro not in descriptors:
            continue
        descriptor = descriptors[metro]
        disk = disk_provisioned.get(metro, 0.0)
        try_hitrates[metro] = descriptor.hitrate_for_cache(disk)

        replication_factor = replication_factor_for_metro(metro, metro_tiers, airport_to_metro)
        effective_traffic = incoming_traffic.get(metro, 0.0) / replication_factor if replication_factor > 0 else incoming_traffic.get(metro, 0.0)
        breakdown = cost_model.compute_monthly_cost(
            total_disk_required_tb=disk,
            hitrate_fraction=try_hitrates[metro] / 100.0,
            total_traffic_mbps=effective_traffic,
            free_disk_tb=0.0,
            is_mch_in_metro=metro in mch_metros,
        )
        cost_by_metro[metro] = breakdown.total_cost

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
        perf_penalty[metro] = penalty_function(p50, p95)

    return try_hitrates, cost_by_metro, perf_penalty, performance_stats


def log_iteration_state(
    iteration: int,
    metros: list[str],
    disk_provisioned: dict[str, float],
    hitrates: dict[str, float],
    cost_by_metro: dict[str, float],
    perf_penalty: dict[str, float],
    performance_stats: dict[str, tuple[float, float]],
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
            handle.write(
                f"{metro}, disk_mb={disk_provisioned.get(metro, 0.0):.2f}, "
                f"hitrate={hitrates.get(metro, 0.0):.2f}, "
                f"cost={cost_by_metro.get(metro, 0.0):.2f}, "
                f"perf_penalty={perf_penalty.get(metro, 0.0):.2f}, "
                f"p50={p50:.2f}, p95={p95:.2f}\n"
            )
        handle.write("\n")

    with summary_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"iteration={iteration}, total_cost={total_cost:.2f}, "
            f"total_perf_penalty={total_penalty:.2f}, objective={combined_total:.2f}\n"
        )


if __name__ == "__main__":
    airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(_BASE_DIR / "PERF" / "metro_areas.csv")
    metro_tiers = get_metro_tiers()
    parent_assignment = assign_parent_metros(airport_info)

    traffic_lookup = _load_traffic_lookup()

    # asn_metro is the end user metro and bw_metro is where the traffic is coming from
    # For each metro in all metros, build a neighborhood list of metros. Store it in a dictionary.
    # A metro is in the neighborhood if this metro sends more than 2Gbps of traffic to the neighbor.
    neighborhood_to = defaultdict(list)
    neighborhood_from = defaultdict(list)
    for (asn_metro, bw_metro), traffic in traffic_lookup.items():
        if traffic > 10000 and asn_metro in metro_to_airport and bw_metro in metro_to_airport:  # Traffic is in Mbps, so 10000 Mbps = 10 Gbps
            neighborhood_to[metro_to_airport[asn_metro]].append(metro_to_airport[bw_metro])
            neighborhood_from[metro_to_airport[bw_metro]].append(metro_to_airport[asn_metro])

    ## Update the traffic lookup to be keyed by airport codes instead of metro names
    traffic_lookup_by_airport = {}
    for (asn_metro, bw_metro), traffic in traffic_lookup.items():
        if asn_metro in metro_to_airport and bw_metro in metro_to_airport:
            traffic_lookup_by_airport[(metro_to_airport[asn_metro], metro_to_airport[bw_metro])] = traffic

    metros = list(airport_info.keys())
    INCOMING_TRAFFIC = defaultdict(float)
    IN_METRO_TRAFFIC = defaultdict(float)
    TRAFFIC_FROM = defaultdict(float)

    for metro in metros:
        for from_metro in neighborhood_from.get(metro, []):
            traffic = traffic_lookup_by_airport.get((from_metro, metro), 0.0)
            INCOMING_TRAFFIC[metro] += traffic
            TRAFFIC_FROM[from_metro] += traffic

    for metro in metros:
        IN_METRO_TRAFFIC[metro] = traffic_lookup_by_airport.get((metro, metro), 0.0)

    total_traffic = sum(INCOMING_TRAFFIC[metro] for metro in metros)

    print("Incoming traffic for each metro (in Mbps):")
    for metro in metros:
        if INCOMING_TRAFFIC[metro] > 0:
            print(f"{metro}: {INCOMING_TRAFFIC[metro]} Mbps")

    print("Traffic served from each metro (in Mbps):")
    for metro in metros:
        if TRAFFIC_FROM[metro] > 0:
            print(f"{metro}: {TRAFFIC_FROM[metro]} Mbps")

    print("In-metro traffic for each metro (in Mbps):")
    for metro in metros:
        if IN_METRO_TRAFFIC[metro] > 0:
            print(f"{metro}: {IN_METRO_TRAFFIC[metro]} Mbps")

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

    COST_PERF_CURVES = load_cost_performance_curves_threaded(
        metros,
        FDS_BY_METRO,
        IN_METRO_TRAFFIC,
        metro_tiers,
        airport_to_metro,
        MCH_METROS,
        PERFORMANCE_MODELS,
    )
    print(f"Computed {len(COST_PERF_CURVES)} cost-performance curves using up to {max(1, os.cpu_count() or 1)} threads")

    for metro, curve_points in COST_PERF_CURVES.items():
        print(f"\nCost-performance sweep for {metro}:")
        for point in curve_points:
            print(
                f"{metro}: Hitrate = {point['hitrate']:.0f}%, Disk = {point['disk']:.2f} MB, "
                f"Cost = ${point['cost']:.2f}, P50 = {point['p50']:.2f} ms, P95 = {point['p95']:.2f} ms"
            )
        plot_cost_performance_curve(metro, curve_points, _COST_PERF_DIR)

    print(f"Saved cost-performance plots to {_COST_PERF_DIR}")


    '''
    gradient_step = 500000 # Fixed step size in MB
    active_metros = [metro for metro in metros if metro in FDS_BY_METRO and metro in PERFORMANCE_MODELS]
    per_metro_log_path = _BASE_DIR / "gradient_descent_metro_log.txt"
    summary_log_path = _BASE_DIR / "gradient_descent_summary_log.txt"

    per_metro_log_path.write_text("", encoding="utf-8")
    summary_log_path.write_text("", encoding="utf-8")

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
        )

        updated_disk_provisioned = dict(DISK_PROVISIONED)
        for metro in active_metros:
            gradient = gradients.get(metro, 0.0)
            descriptor = FDS_BY_METRO[metro]
            current_disk = DISK_PROVISIONED.get(metro, 0.0)
            max_cache_space = max(point.cache_space for point in descriptor._points_sorted_by_cache)
            if gradient < 0:
                new_disk = min(current_disk + gradient_step, max_cache_space)
            else:
                new_disk = max(current_disk - gradient_step, 0.0)
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
            total_cost,
            total_penalty,
            combined_total,
            per_metro_log_path,
            summary_log_path,
        )

        print(f"Iteration {iteration}")
        for metro in active_metros:
            p50, p95 = performance_stats.get(metro, (0.0, 0.0))
            print(
                f"{metro}: disk={DISK_PROVISIONED.get(metro, 0.0):.2f} MB, "
                f"hitrate={HITRATES.get(metro, 0.0):.2f}%, "
                f"cost={COST.get(metro, 0.0):.2f}, "
                f"perf_penalty={PERF_PENALTY.get(metro, 0.0):.2f}, "
                f"p50={p50:.2f} ms, p95={p95:.2f} ms"
            )
        print(
            f"Overall total cost={total_cost:.2f}, total performance penalty={total_penalty:.2f}, "
            f"objective={combined_total:.2f}"
        )

    '''
