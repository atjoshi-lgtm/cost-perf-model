from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from analyse import *
from cost import CaribouCostCalculator
from fds import FootprintDescriptor, load_footprint_descriptors
from perf_with_mch import MCHPerformanceModel, MetroPerformanceWithMCH
from probability import Convolution, weighted_pdf_sum
from solve_for_US import parse_metro_areas, assign_parent_metros

BASE_DIR = Path('.')
_FDS_DIR = BASE_DIR / "FDS"

ALL_METROS = ["LAX", "LAS", "PHX"]
MCH_PARENT_METROS = ["LAX"]

MCH_PARENT_ASSIGNMENT = {
	"LAX": "LAX",
	"LAS": "LAX",
	"PHX": "LAX",
}

METRO_NAMES = {
	"LAX": "Los_Angeles",
	"LAS": "Las_Vegas",
	"PHX": "Phoenix",
}

NEIGHBOR_METROS = {
	"LAX": ["LAX", "LAS", "PHX"],
	"LAS": ["LAS", "LAX"],
	"PHX": ["PHX", "LAX"],
}

FDS: dict[str, FootprintDescriptor] = {}
MAX_CACHE_SPACE: dict[str, float] = {}
DISK_STEP: dict[str, float] = {}
for metro in ALL_METROS:
	FDS[metro] = FootprintDescriptor.from_file(_FDS_DIR / f"{metro.lower()}.txt")
	MAX_CACHE_SPACE[metro] = max(point.cache_space for point in FDS[metro]._points_sorted_by_cache)
	DISK_STEP[metro] = MAX_CACHE_SPACE[metro] * 0.01
	print(
		f"Metro: {metro}, Max Possible Hitrate: {FDS[metro].find_max_possible_hitrate().hitrate}%, "
		f"STEP: {DISK_STEP[metro]} MB"
	)

MCH_PERFORMANCE_MODELS: dict[str, MCHPerformanceModel] = {}
for metro in MCH_PARENT_METROS:
	mch_performance_model = MCHPerformanceModel(name=metro)
	tat_pdf = get_parent_tat_pdf(METRO_NAMES[metro])
	mch_performance_model.set_mch_tat(tat_pdf)
	MCH_PERFORMANCE_MODELS[metro] = mch_performance_model

PERFORMANCE_MODELS: dict[str, MetroPerformanceWithMCH] = {}
TRAFFIC = defaultdict(lambda: defaultdict(float))


def get_traffic(asn_metro: str, bw_metro: str) -> float:
	db_path = BASE_DIR / "SERVEDFROM_DATA" / "metros_servedfrom_copy.db"
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	query = """
	SELECT total_traffic
	FROM traffic_summary
	WHERE asn_metro = ? AND bw_metro = ? AND bucket = 'Disney_Videos' AND quarter = '1Q26'
	"""

	cursor.execute(query, (asn_metro, bw_metro))
	result = cursor.fetchone()
	conn.close()

	if result is not None:
		return result[0] * 1000.0
	return 0.0


for metro in ALL_METROS:
	metro_performance = MetroPerformanceWithMCH(
		name=metro,
		parent_model=MCH_PERFORMANCE_MODELS[MCH_PARENT_ASSIGNMENT[metro]],
		descriptor=FDS[metro],
	)

	edge_tat_hit = get_edge_tat_pdf(METRO_NAMES[metro])
	metro_performance.set_edge_tat_hit(edge_tat_hit)

	parent_name = METRO_NAMES[MCH_PARENT_ASSIGNMENT[metro]]
	rtt_cache_miss = get_midgress_rtt_pdf(parent_name, METRO_NAMES[metro])
	metro_performance.set_mch_rtt(rtt_cache_miss)

	for neighbor in NEIGHBOR_METROS[metro]:
		TRAFFIC[metro][neighbor] = get_traffic(asn_metro=METRO_NAMES[neighbor], bw_metro=METRO_NAMES[metro])
		edge_rtt_pdf = get_rtt_pdf(METRO_NAMES[metro], client_metro_ids[METRO_NAMES[neighbor]])
		metro_performance.set_edge_rtt(neighbor, edge_rtt_pdf)

	PERFORMANCE_MODELS[metro] = metro_performance

conn = Convolution()
cost_model = CaribouCostCalculator()
INCOMING_TRAFFIC = defaultdict(float)
for metro in ALL_METROS:
	INCOMING_TRAFFIC[metro] = sum(TRAFFIC[metro][neighbor] for neighbor in NEIGHBOR_METROS[metro])
	print(f"Incoming traffic for {metro}: {INCOMING_TRAFFIC[metro]} Mbps")


def compute_weighted_ttfb(metro: str, disk_mb: float) -> tuple[float, float]:
	hitrate = FDS[metro].hitrate_for_cache(disk_mb)
	ttfb_pdfs = []
	weights = []

	for to_metro in NEIGHBOR_METROS[metro]:
		pdf = PERFORMANCE_MODELS[to_metro].get_ttfb_pdf(
			from_metro=metro,
			hitrate=hitrate,
			conn=conn,
		)
		ttfb_pdfs.append(pdf.to_microsecond_pdf())
		weights.append(TRAFFIC[to_metro][metro])

	combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
	p50 = combined_pdf.millisecond_at_percentile(50) / 100
	p95 = combined_pdf.millisecond_at_percentile(95) / 100
	return p50, p95


def disk_sweep_values(metro: str) -> list[float]:
	min_disk_mb = min(point.cache_space for point in FDS[metro]._points_sorted_by_cache)
	max_disk_mb = MAX_CACHE_SPACE[metro]
	step_mb = DISK_STEP[metro]

	values: list[float] = []
	current_disk_mb = min_disk_mb
	while current_disk_mb <= max_disk_mb:
		values.append(current_disk_mb)
		next_disk_mb = min(current_disk_mb + step_mb, max_disk_mb)
		if next_disk_mb == current_disk_mb:
			break
		current_disk_mb = next_disk_mb

	if values[-1] < max_disk_mb:
		values.append(max_disk_mb)

	return values


def plot_cost_performance_curves(metro: str) -> None:
	disk_values = disk_sweep_values(metro)
	plotted_disk_values: list[float] = []
	cost_values: list[float] = []
	p50_values: list[float] = []
	p95_values: list[float] = []

	for disk_mb in disk_values:
		hitrate = FDS[metro].hitrate_for_cache(disk_mb)
		try:
			p50, p95 = compute_weighted_ttfb(metro, disk_mb)
		except ValueError as exc:
			print(f"Skipping metro {metro} at disk {disk_mb:.2f} MB: {exc}")
			continue

		breakdown = cost_model.compute_monthly_cost(
			total_disk_required_tb=disk_mb,
			hitrate_fraction=hitrate / 100.0,
			total_traffic_mbps=INCOMING_TRAFFIC[metro],
			free_disk_tb=0.0,
			is_mch_in_metro=metro in MCH_PARENT_METROS,
		)

		plotted_disk_values.append(disk_mb)
		cost_values.append(breakdown.total_cost)
		p50_values.append(p50)
		p95_values.append(p95)
		print(
			f"Metro: {metro}, Disk: {disk_mb:.2f} MB, Hitrate: {hitrate:.2f}%, "
			f"Cost: {breakdown.total_cost:.2f} USD, P50: {p50:.2f} ms, P95: {p95:.2f} ms"
		)

	if not plotted_disk_values:
		print(f"Skipping plot for {metro}: no valid TTFB points were available")
		return

	plt.figure()
	plt.plot(plotted_disk_values, p50_values, marker="o", label="P50 TTFB")
	plt.plot(plotted_disk_values, p95_values, marker="s", label="P95 TTFB")
	plt.title(f"TTFB vs Disk for {metro}")
	plt.xlabel("Disk provisioned (MB)")
	plt.ylabel("TTFB (ms)")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	output_path = BASE_DIR / f"ttfb_vs_disk_{metro}.png"
	plt.savefig(output_path)
	plt.close()
	print(f"Saved plot to {output_path}")

	plt.figure()
	plt.plot(p50_values, cost_values, marker="o")
	plt.title(f"Cost vs P50 TTFB for {metro}")
	plt.xlabel("P50 TTFB (ms)")
	plt.ylabel("Monthly cost (USD)")
	plt.grid(True)
	plt.tight_layout()
	p50_output_path = BASE_DIR / f"cost_vs_p50_{metro}.png"
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
	p95_output_path = BASE_DIR / f"cost_vs_p95_{metro}.png"
	plt.savefig(p95_output_path)
	plt.close()
	print(f"Saved plot to {p95_output_path}")


# --- Convexity Check Code ---
PLOT_DIR = BASE_DIR / 'convexity'
PLOT_DIR.mkdir(exist_ok=True)

# Load metro info
airport_info, metro_to_airport, airport_to_metro = parse_metro_areas(BASE_DIR / 'PERF' / 'metro_areas.csv')
parent_assignment = assign_parent_metros(airport_info)
FDS_BY_METRO = load_footprint_descriptors(BASE_DIR / 'FDS')

# Build traffic lookup (as in solve_for_US)
def _parse_csv_line(line):
    return [part.strip().strip('"') for part in line.strip().lstrip("\ufeff").split('","')]
traffic_lookup = {}
with (BASE_DIR / 'SERVEDFROM_DATA' / 'served_from.csv').open('r', encoding='utf-8') as f:
    header = _parse_csv_line(f.readline())
    asn_idx, bw_idx, traffic_idx = header.index('asn_metro'), header.index('bw_metro'), header.index('traffic_mbps')
    for line in f:
        if not line.strip(): continue
        row = _parse_csv_line(line)
        traffic_lookup[(row[asn_idx], row[bw_idx])] = float(row[traffic_idx])

# Build neighborhood_from: for each metro, list of source metros sending traffic to it
from collections import defaultdict
neighborhood_from = defaultdict(list)
for (asn_metro, bw_metro), traffic in traffic_lookup.items():
    if traffic > 10000 and asn_metro in metro_to_airport and bw_metro in metro_to_airport:
        neighborhood_from[metro_to_airport[bw_metro]].append(metro_to_airport[asn_metro])

# Pick 5 random metros from the available FDs
all_metros = list(FDS_BY_METRO.keys())
random.seed(42)
selected_sources = random.sample(all_metros, 5)

# For each source, get all neighbors it routes to (from neighborhood_from)
for src in selected_sources:
    neighbors = set(neighborhood_from.get(src, []))
    if not neighbors:
        continue
    for dst in neighbors:
        # For each disk size in dst's FD (1TB increments)
        fd = FDS_BY_METRO[dst]
        min_disk = min(point.cache_space for point in fd._points_sorted_by_cache)
        max_disk = max(point.cache_space for point in fd._points_sorted_by_cache)
        disk_sizes = list(range(int(min_disk), int(max_disk)+1, 1000))  # MB, 1TB=1000GB=1000*1MB
        p50s, p95s, disks = [], [], []
        for disk in disk_sizes:
            # Find closest FD point >= disk
            fd_point = next((p for p in fd._points_sorted_by_cache if p.cache_space >= disk), fd._points_sorted_by_cache[-1])
            hitrate = fd_point.hitrate
            # Build perf model for dst
            parent_code = parent_assignment[dst][0]
            parent_name = airport_to_metro[parent_code]
            metro_name = airport_to_metro[dst]
            parent_model = MCHPerformanceModel(parent_name)
            parent_model.set_mch_tat(get_parent_tat_pdf(parent_name))
            perf_model = MetroPerformanceWithMCH(metro_name, parent_model, fd)
            # Set edge RTTs
            for from_metro in neighborhood_from.get(dst, []):
                src_name = airport_to_metro[from_metro]
                if src_name in client_metro_ids:
                    perf_model.set_edge_rtt(from_metro, get_rtt_pdf(metro_name, client_metro_ids[src_name]))
            perf_model.set_edge_tat_hit(get_edge_tat_pdf(metro_name, cache_hit_type=1))
            perf_model.set_edge_tat_miss(get_edge_tat_pdf(metro_name, cache_hit_type=0))
            perf_model.set_mch_rtt(get_midgress_rtt_pdf(parent_name, metro_name))
            # Compute TTFB for src->dst
            if src in client_metro_ids and dst in perf_model.edge_rtt:
                conn = Convolution()
                ttfb_pdf = perf_model.get_ttfb_pdf(src, hitrate, conn)
                try:
                    p50 = ttfb_pdf.millisecond_at_percentile(50)
                    p95 = ttfb_pdf.millisecond_at_percentile(95)
                except Exception:
                    p50 = p95 = float('nan')
                p50s.append(p50)
                p95s.append(p95)
                disks.append(disk)
        # Plot p50
        if disks:
            plt.figure()
            plt.plot(disks, p50s, marker='o')
            plt.xlabel('Disk size in destination metro (MB)')
            plt.ylabel('p50 TTFB (ms)')
            plt.title(f'p50 TTFB: {src} → {dst}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f'p50_{src}_to_{dst}.png')
            plt.close()
            # Plot p95
            plt.figure()
            plt.plot(disks, p95s, marker='o', color='orange')
            plt.xlabel('Disk size in destination metro (MB)')
            plt.ylabel('p95 TTFB (ms)')
            plt.title(f'p95 TTFB: {src} → {dst}')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f'p95_{src}_to_{dst}.png')
            plt.close()
