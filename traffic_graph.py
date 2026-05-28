# I have a table in SERVEDFROM_DATA/metros_servedfrom_copy.db called traffic_summary with columns asn_metro|bucket|bw_metro|quarter|total_traffic
# Here asn_metro is the metro where the traffic originates, for bucket always use Disney_Videos, bw_metro is the metro where the traffic is served from, quarter is 1Q26 and total_traffic is the total traffic in Gbps. 
# Wite a function that given a end user metro and a serving metro, returns the total traffic in Gbps for the Disney_Videos bucket for 1Q26. Use sqlite3 to query the database.
from __future__ import annotations

from pathlib import Path
import random

from fds import FootprintDescriptor
from cost import CaribouCostCalculator
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
    parser.add_argument("--traffic-threshold", type=float, default=20000.0, help="Minimum traffic (in Mbps) required to plot an edge.")
    return parser.parse_args(args_list)

try:
    _args = get_args()
    _GEO = _args.geo
    _BUCKET = _args.bucket
    _TRAFFIC_THRESHOLD = _args.traffic_threshold
except SystemExit:
    _GEO = "NA"
    _BUCKET = "AkamaiHD"
    _TRAFFIC_THRESHOLD = 20000.0


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
        try:
            normalized_header = _parse_csv_line(header_line)
            asn_idx = normalized_header.index("asn_metro")
            bw_idx = normalized_header.index("bw_metro")
            traffic_idx = normalized_header.index("traffic_mbps")
        except ValueError:
            # Fallback if names are slightly different
            asn_idx, bw_idx, traffic_idx = 0, 1, 2
        for line in csv_file:
            if not line.strip():
                continue
            normalized_row = _parse_csv_line(line)
            lookup[(normalized_row[asn_idx], normalized_row[bw_idx])] = float(normalized_row[traffic_idx])
    return lookup

def parse_metro_areas(file_path: Path, target_geo: str = "NA") -> tuple[dict[str, dict], dict[str, str], dict[str, str]]:
    airport_info = {}
    metro_to_airport = {}
    airport_to_metro = {}

    geo_mapping = load_geo_mapping()

    with file_path.open() as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 8:
                continue

            parts = [part.strip().strip('"') for part in parts]
            id, metro_area, latitude, longitude, airport_code, country, state, max_distance = parts

            metro_geo = geo_mapping.get(metro_area)
            if metro_geo != target_geo:
                continue

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

def plot_neighborhood_graph(neighborhood: dict[str, list[str]], airport_info: dict[str, dict], traffic_lookup: dict[tuple[str, str], float]):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    for metro in neighborhood:
        if metro not in airport_info:
            continue
        if metro == "HNL" or metro == "ANC":  # Skip Honolulu and Anchorage as they are outliers far from the continental US
            continue
        lat = airport_info[metro]['latitude']
        lon = airport_info[metro]['longitude']
        G.add_node(metro, pos=(lon, lat))

    for metro, neighbors in neighborhood.items():
        if metro == "HNL" or metro == "ANC":
            continue
        for neighbor in neighbors:
            if neighbor == "HNL" or neighbor == "ANC":
                continue
            traffic = traffic_lookup.get((metro, neighbor), 0.0)
            if traffic > _TRAFFIC_THRESHOLD:  # Only show edges with > _TRAFFIC_THRESHOLD Mbps
                if neighbor not in G and neighbor in airport_info:
                    G.add_node(neighbor, pos=(airport_info[neighbor]['longitude'], airport_info[neighbor]['latitude']))
                G.add_edge(metro, neighbor, weight=traffic)

    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_edge_attributes(G, 'weight')

    # Apply a manual jitter to separate metros that are geographically very close (e.g., East Coast)
    import random
    random.seed(42)
    jittered_pos = {}
    for node, (lon, lat) in pos.items():
        # A small random offset (approx 0.7 degrees) to spread overlapping nodes
        jittered_pos[node] = (lon + random.uniform(-0.7, 0.7), lat + random.uniform(-0.7, 0.7))

    plt.figure(figsize=(30, 36))
    
    nx.draw_networkx_nodes(G, jittered_pos, node_size=800, node_color='lightblue', alpha=0.8)
    
    # Calculate total generated traffic per metro (in Gbps), rounded to nearest integer
    metro_total_traffic = {}
    for (asn, bw), t in traffic_lookup.items():
        metro_total_traffic[asn] = metro_total_traffic.get(asn, 0.0) + t
        
    node_labels = {}
    for node in G.nodes():
        traffic_gbps = round(metro_total_traffic.get(node, 0.0) / 1000.0)
        node_labels[node] = f"{node}\n{traffic_gbps}G"
    
    # Draw labels directly inside the node points
    nx.draw_networkx_labels(G, jittered_pos, labels=node_labels, font_size=8, font_weight='bold')
    
    if weights:
        max_weight = max(weights.values())
        edge_widths = [5 * (w / max_weight) for w in weights.values()]
        nx.draw_networkx_edges(G, jittered_pos, width=edge_widths, alpha=0.6, 
                               edge_color='gray', arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.1')
        
        edge_labels = {edge: f"{weight/1000:.1f}G" for edge, weight in weights.items()}
        nx.draw_networkx_edge_labels(G, jittered_pos, edge_labels=edge_labels, font_size=7)

    plt.title(f"Traffic Graph: End-User ({_GEO}) → Serving Metros ({_BUCKET}, edges > {_TRAFFIC_THRESHOLD/1000} Gbps)", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = _BASE_DIR / f"traffic_graph_{_GEO}_{_BUCKET}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Traffic graph saved to {output_path}")


def plot_cross_metro_serving_graph(airport_info: dict[str, dict], traffic_lookup: dict[tuple[str, str], float]):
    """Plot only edges where a metro's traffic is being served by a *different* metro."""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    for (asn, bw), traffic in traffic_lookup.items():
        if asn == bw:
            continue  # Skip self-serving
        if traffic <= _TRAFFIC_THRESHOLD:
            continue
        if asn not in airport_info or bw not in airport_info:
            continue
        if asn in ("HNL", "ANC") or bw in ("HNL", "ANC"):
            continue

        for metro in (asn, bw):
            if metro not in G:
                lon = airport_info[metro]['longitude']
                lat = airport_info[metro]['latitude']
                G.add_node(metro, pos=(lon, lat))

        G.add_edge(asn, bw, weight=traffic)

    pos = nx.get_node_attributes(G, 'pos')

    import random
    random.seed(42)
    jittered_pos = {node: (lon + random.uniform(-0.7, 0.7), lat + random.uniform(-0.7, 0.7))
                    for node, (lon, lat) in pos.items()}

    weights = nx.get_edge_attributes(G, 'weight')

    # Color source metros (asn) orange, serving metros (bw) green
    asn_nodes = {e[0] for e in G.edges()}
    bw_nodes = {e[1] for e in G.edges()}
    both_nodes = asn_nodes & bw_nodes
    only_asn = asn_nodes - bw_nodes
    only_bw = bw_nodes - asn_nodes

    plt.figure(figsize=(30, 18))
    nx.draw_networkx_nodes(G, jittered_pos, nodelist=list(only_asn), node_size=800, node_color='orange', alpha=0.9)
    nx.draw_networkx_nodes(G, jittered_pos, nodelist=list(only_bw), node_size=800, node_color='lightgreen', alpha=0.9)
    nx.draw_networkx_nodes(G, jittered_pos, nodelist=list(both_nodes), node_size=800, node_color='skyblue', alpha=0.9)

    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, jittered_pos, labels=node_labels, font_size=8, font_weight='bold')

    if weights:
        max_weight = max(weights.values())
        edge_widths = [5 * (w / max_weight) for w in weights.values()]
        nx.draw_networkx_edges(G, jittered_pos, width=edge_widths, alpha=0.6,
                               edge_color='crimson', arrows=True, arrowsize=10,
                               connectionstyle='arc3,rad=0.1')
        edge_labels = {edge: f"{w/1000:.1f}G" for edge, w in weights.items()}
        nx.draw_networkx_edge_labels(G, jittered_pos, edge_labels=edge_labels, font_size=7)

    plt.title(
        f"Cross-Metro Serving ({_GEO}, {_BUCKET}): orange=end-user only, green=serving only, blue=both",
        fontsize=14,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = _BASE_DIR / f"cross_metro_serving_{_GEO}_{_BUCKET}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Cross-metro serving graph saved to {output_path}")

def plot_traffic_graph():
    airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(_BASE_DIR / "PERF" / "metro_areas.csv", _GEO)
    traffic_lookup = _load_traffic_lookup()

    neighborhood_to = defaultdict(list)
    traffic_lookup_by_airport = {}

    for (asn_metro, bw_metro), traffic in traffic_lookup.items():
        if traffic > _TRAFFIC_THRESHOLD and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
            neighborhood_to[metro_to_airport[asn_metro]].append(metro_to_airport[bw_metro])

    for (asn_metro, bw_metro), traffic in traffic_lookup.items():
        if asn_metro in metro_to_airport and bw_metro in metro_to_airport:
            traffic_lookup_by_airport[(metro_to_airport[asn_metro], metro_to_airport[bw_metro])] = traffic

    print(f"Generating traffic routing graph for {_GEO}, bucket {_BUCKET}, threshold {_TRAFFIC_THRESHOLD} Mbps...")
    plot_neighborhood_graph(neighborhood_to, airport_info, traffic_lookup_by_airport)
    plot_cross_metro_serving_graph(airport_info, traffic_lookup_by_airport)

if __name__ == "__main__":
    plot_traffic_graph()