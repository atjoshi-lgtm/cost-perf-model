from __future__ import annotations

from pathlib import Path
import csv
import random

from fds import FootprintDescriptor
from cost import CaribouCostCalculator
from perf_with_mch import MetroPerformanceWithMCH, MCHPerformanceModel
from probability import Convolution, gaussian_pdf, weighted_pdf_sum
from analyse import *
from itertools import product
from collections import defaultdict
import sys

INT32_MAX = 2**31 - 1

_BASE_DIR = Path(__file__).resolve().parent
_FDS_DIR = _BASE_DIR / "FDS"

ALL_METROS = ["AUS", "DFW", "IAH", "SAT", "OKC"]

MCH_PARENT_METROS = ["DFW"]

MCH_PARENT_ASSIGNMENT = {
	"AUS": "DFW",
	"DFW": "DFW",
	"IAH": "DFW",
    "SAT": "DFW",
    "OKC": "DFW"
	}

REPLICATION ={
	"AUS": 5,
	"DFW": 7,
	"IAH": 7,
	"SAT": 1,
	"OKC": 1
}

METRO_NAMES = {
    "AUS": "Austin",
    "DFW": "Dallas",
	"IAH": "Houston",
	"SAT": "San_Antonio",
	"OKC": "Oklahoma_City"
	}

NEIGHBOR_METROS = {
	"DFW": ["DFW", "IAH", "OKC", "SAT", "AUS"],
	"IAH": ["IAH", "DFW"],
	"SAT": ["SAT", "DFW"],
	"OKC": ["OKC", "DFW"],
    "AUS": ["AUS", "DFW"]
	}

FDS = {}
DISK_PROVISIONED = {}
COST = {}
PERF_PENALTY = {}  
HITRATES = {} 
TRY_HITRATES = {}
MAX_POSSIBLE_HITRATE = {}
MAX_CACHE_SPACE = {}
DISK_STEP = {}
for metro in METRO_NAMES:
	FDS[metro] = FootprintDescriptor.from_file(_FDS_DIR / f"{metro.lower()}.txt")
	DISK_PROVISIONED[metro] = 0.0
	COST[metro] = 0.0
	PERF_PENALTY[metro] = 0.0
	MAX_POSSIBLE_HITRATE[metro] = FDS[metro].find_max_possible_hitrate().hitrate
	MAX_CACHE_SPACE[metro] = max(point.cache_space for point in FDS[metro]._points_sorted_by_cache)
	DISK_STEP[metro] = MAX_CACHE_SPACE[metro] * 0.005
	print(f"Metro: {metro}, Max Possible Hitrate: {MAX_POSSIBLE_HITRATE[metro]}, STEP: {DISK_STEP[metro]} TB")
	HITRATES[metro] = 0.0
	TRY_HITRATES[metro] = 0.0

MCH_PERFORMANCE_MODELS = {}
for metro in MCH_PARENT_METROS:
	mch_performance_model = MCHPerformanceModel(name=metro)
	tat_pdf = get_parent_tat_pdf(METRO_NAMES[metro])
	tat_pdf.plot(title=f"MCH TAT PDF for metro {metro}")
	plt.savefig(f"mch_tat_pdf_{metro}.png")
	plt.clf()
	mch_performance_model.set_mch_tat(tat_pdf)
	MCH_PERFORMANCE_MODELS[metro] = mch_performance_model

# Read traffic from SERVEDFROM_DATA/served_from.csv with columns:
# asn_metro,bw_metro,traffic_mbps

TRAFFIC_LOOKUP = {}


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


TRAFFIC_LOOKUP = _load_traffic_lookup()

def get_traffic(asn_metro: str, bw_metro: str) -> float:
	return TRAFFIC_LOOKUP.get((asn_metro, bw_metro), 0.0)


PERFORMANCE_MODELS = {}
TRAFFIC = defaultdict(lambda: defaultdict(float))

for metro in METRO_NAMES:

	metro_performance = MetroPerformanceWithMCH(name=metro, parent_model=MCH_PERFORMANCE_MODELS[MCH_PARENT_ASSIGNMENT[metro]], descriptor=FDS[metro])

	edge_tat_hit = get_edge_tat_pdf(METRO_NAMES[metro])
	print(f"Edge TAT Hit PDF for metro {metro}:")
	edge_tat_hit.plot(title=f"Edge TAT Hit PDF for metro {metro}")
	#plt.show()
	plt.savefig(f"edge_tat_hit_pdf_{metro}.png")
	plt.clf()
	metro_performance.set_edge_tat_hit(edge_tat_hit)

	parent_name = MCH_PARENT_ASSIGNMENT[metro]
	parent_name = METRO_NAMES[parent_name]

	rtt_cache_miss = get_midgress_rtt_pdf(parent_name, METRO_NAMES[metro])
	print(f"RTT PDF for cache miss from  metro {metro} to parent {parent_name}:")
	rtt_cache_miss.plot(title=f"RTT PDF for cache miss from  metro {metro} to parent {parent_name}")
	#plt.show()
	plt.savefig(f"rtt_cache_miss_pdf_{metro}.png")
	plt.clf()
	metro_performance.set_mch_rtt(rtt_cache_miss)

	for neighbor in NEIGHBOR_METROS[metro]:
		TRAFFIC[metro][neighbor] = get_traffic(asn_metro=METRO_NAMES[neighbor], bw_metro=METRO_NAMES[metro])
		edge_rtt_pdf = get_rtt_pdf(METRO_NAMES[metro], client_metro_ids[METRO_NAMES[neighbor]])
		print(f"RTT PDF for edge {neighbor} to metro {metro}:")
		edge_rtt_pdf.plot(title=f"RTT PDF for {neighbor} to metro {metro}")
		#plt.show()
		plt.savefig(f"rtt_pdf_{neighbor}_to_{metro}.png")
		plt.clf()
		metro_performance.set_edge_rtt(neighbor, edge_rtt_pdf)

	PERFORMANCE_MODELS[metro] = metro_performance

cost_model = CaribouCostCalculator()
conn = Convolution()

metros = NEIGHBOR_METROS.keys()
INCOMING_TRAFFIC = defaultdict(float)
TRAFFIC_FROM = defaultdict(float)

for metro in metros:
	for neighbor in NEIGHBOR_METROS[metro]:
		INCOMING_TRAFFIC[metro] += TRAFFIC[metro][neighbor]
		TRAFFIC_FROM[neighbor] += TRAFFIC[metro][neighbor]

total_traffic = sum(INCOMING_TRAFFIC[metro] for metro in metros)

# Print the traffic fraction
for metro in metros:
	print(f"Traffic fraction for {metro}: {float(INCOMING_TRAFFIC[metro]) / total_traffic if total_traffic > 0 else 0.0}")

print("Incoming traffic for each metro (in Mbps):")
for metro in metros:
	print(f"{metro}: {INCOMING_TRAFFIC[metro]} Mbps")

def compute_performance(metro):
	p50, p95 = 0.0, 0.0
	ttfb_pdfs = []
	weights = []
	for to_metro in NEIGHBOR_METROS[metro]:
		hitrate = TRY_HITRATES[to_metro]
		#print(f"Computing TTFB PDF for {metro} to {to_metro} with hitrate {hitrate}%")
		pdf = PERFORMANCE_MODELS[to_metro].get_ttfb_pdf(
			from_metro=metro,
			hitrate=hitrate,
			conn=conn,
		)
		#pdf.plot()
		pdf_microsecond = pdf.to_microsecond_pdf()

		ttfb_pdfs.append(pdf_microsecond)
		weights.append(TRAFFIC[to_metro][metro])

	combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
	p50 = combined_pdf.millisecond_at_percentile(50)/100
	p95 = combined_pdf.millisecond_at_percentile(95)/100
	#combined_pdf.plot()
	return p50, p95

def penalty_function(p50, p95):
	return 50.0*((max(p50 - 23, 0))**2) + 50.0 * (max(p95 - 105, 0))

def compute_perf_penalty(metro):
	new_perf_penalty = {}
	for m in NEIGHBOR_METROS[metro]:
		p50, p95 = compute_performance(m)
		new_perf_penalty[m] =  penalty_function(p50, p95)
	return new_perf_penalty

def compute_objective():
	total_cost = 0.0
	penalty = 0.0
	for metro in metros:
		total_cost += COST[metro]
		penalty += PERF_PENALTY[metro]
	return total_cost + penalty

def compute_gradient(metro):

	#print(f"Computing gradient for metro: {metro}")
	cost_differential = 0.0
	perf_differential = 0.0

	current_disk = DISK_PROVISIONED[metro]
	step = DISK_STEP[metro]
	new_disk = min(current_disk + step, MAX_CACHE_SPACE[metro])
	current_hitrate = TRY_HITRATES[metro]	
	new_hitrate = FDS[metro].hitrate_for_cache(new_disk)
	TRY_HITRATES[metro] = new_hitrate
	#print(f"Current hitrate: {hitrate}%, New hitrate: {new_hitrate}%")
	#print(f"Current disk: {current_disk} TB, New disk: {new_disk} TB")
	breakdown = cost_model.compute_monthly_cost(
		total_disk_required_tb=new_disk,
		hitrate_fraction=new_hitrate / 100.0,
		total_traffic_mbps=INCOMING_TRAFFIC[metro]/REPLICATION[metro],
		free_disk_tb=0.0,
	)

	cost_differential += breakdown.total_cost - COST[metro]
	localized_perf_penalty = compute_perf_penalty(metro)
	#print(f"Localized performance penalty after increasing hitrate: {localized_perf_penalty}")
	for m in localized_perf_penalty:
		perf_differential += localized_perf_penalty[m] - PERF_PENALTY[m]

	print(f"Localized performance penalty and cost_differential for metro {metro} after increasing disk: {perf_differential}, Cost differential: {cost_differential}")
	print(f"Current hitrate: {current_hitrate}%, New hitrate: {new_hitrate}%, Current disk: {current_disk} TB, New disk: {new_disk} TB")

	# decrement disk by one step to compute the other side of the gradient
	previous_disk = max(current_disk - step, 0.0)
	previous_hitrate = FDS[metro].hitrate_for_cache(previous_disk)
	TRY_HITRATES[metro] = previous_hitrate
	#print(f"Previous hitrate: {previous_hitrate}%, Previous disk: {previous_disk} TB")
	previous_breakdown = cost_model.compute_monthly_cost(
		total_disk_required_tb=previous_disk,
		hitrate_fraction=previous_hitrate / 100.0,
		total_traffic_mbps=INCOMING_TRAFFIC[metro]/REPLICATION[metro],
		free_disk_tb=0.0,
	)

	cost_differential_1 = previous_breakdown.total_cost - COST[metro]
	#print(f"Cost differential: {cost_differential}")
	localized_perf_penalty_1 = compute_perf_penalty(metro)
	#print(f"Localized performance penalty after decreasing hitrate: {localized_perf_penalty}")
	perf_differential_1 = 0.0
	for m in localized_perf_penalty_1:
		perf_differential_1 += localized_perf_penalty_1[m] - PERF_PENALTY[m]
	#print("PERF penalty ", PERF_PENALTY)
	print(f"Localized performance penalty and cost_differential for metro {metro} after decreasing disk: {perf_differential_1}, Cost differential: {cost_differential_1}")
	print(f"Previous hitrate: {previous_hitrate}%, Previous disk: {previous_disk} TB")


	cost_differential = (cost_differential - cost_differential_1) 
	perf_differential = (perf_differential - perf_differential_1)
	TRY_HITRATES[metro] = FDS[metro].hitrate_for_cache(current_disk)
	gradient = (cost_differential + perf_differential) / 2.0
	return gradient

for metro in metros:
	min_cost = INT32_MAX
	disk_tb = 0.0
	min_hitrate = 50
	min_breakdown = None

	for hitrate in range(1, 101, 1):

		breakdown = cost_model.compute_monthly_cost(
			total_disk_required_tb=FDS[metro].nearest_cache_for_hitrate(hitrate),
			hitrate_fraction=float(hitrate) / 100.0,
			total_traffic_mbps=INCOMING_TRAFFIC[metro]/REPLICATION[metro],
			free_disk_tb=DISK_PROVISIONED[metro],
		)

		if hitrate == 50 or breakdown.total_cost < min_cost:
			min_cost = breakdown.total_cost
			min_hitrate = hitrate
			min_breakdown = breakdown
			disk_tb = FDS[metro].nearest_cache_for_hitrate(hitrate)

	DISK_PROVISIONED[metro] += disk_tb
	COST[metro] = min_cost
	HITRATES[metro] = min_hitrate
	TRY_HITRATES[metro] = min_hitrate
	
	#print(f"Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {min_hitrate}%, Cost: {min_cost:.2f} USD")

for metro in metros:
	p50, p95 = compute_performance(metro)  
	print(f"Metro: {metro}, P50: {p50} ms, P95: {p95} ms")
	PERF_PENALTY[metro] = float(TRAFFIC_FROM[metro])/total_traffic * penalty_function(p50, p95)  
	print(f"Disk provisioned for metro {metro}: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")
	#print(f"Metro: {metro}, Initial Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")

objective_value = compute_objective()
print(f"Initial Objective Value: {objective_value:.2f} USD")

complete = False

i = 0
while True:
	for metro in metros:
		i += 1
		gradient = compute_gradient(metro)
		print(f"Gradient for metro {metro}: {gradient}")
		step = DISK_STEP[metro]
		if gradient < 0:
			print("Gradient step: Increasing disk for metro:", metro)
			new_disk = min(DISK_PROVISIONED[metro] + step, MAX_CACHE_SPACE[metro])
		else:
			print("Gradient step: Decreasing disk for metro:", metro)
			new_disk = max(DISK_PROVISIONED[metro] - step, 0.0)

		if new_disk != DISK_PROVISIONED[metro]:
			new_hitrate = FDS[metro].hitrate_for_cache(new_disk)
			breakdown = cost_model.compute_monthly_cost(
				total_disk_required_tb=new_disk,
				hitrate_fraction=new_hitrate / 100.0,
				total_traffic_mbps=INCOMING_TRAFFIC[metro]/REPLICATION[metro],
				free_disk_tb=0.0,
			)

			COST[metro] = breakdown.total_cost
			DISK_PROVISIONED[metro] = new_disk
			TRY_HITRATES[metro] = new_hitrate
			tmp_perf_penalty = compute_perf_penalty(metro)
			#print(f"Updated performance penalties: {tmp_perf_penalty}")
			##Print the P50 and P95 after the hitrate change
			for m in ALL_METROS:
				p50, p95 = compute_performance(m)
				print(f"Metro: {m}, P50: {p50} ms, P95: {p95} ms")
			print("Cost after disk change:", COST[metro])
			for m in tmp_perf_penalty:
				PERF_PENALTY[m] = tmp_perf_penalty[m]
			print(f"Disk provisioned for metro {metro}: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {new_hitrate}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")
			HITRATES[metro] = new_hitrate
			
	n_objective_value = compute_objective()
	print(f"Updated Objective Value: {n_objective_value:.2f} USD, Total cost: {sum(COST[metro] for metro in metros):.2f} USD, Total performance penalty: {sum(PERF_PENALTY[metro] for metro in metros):.2f} USD")
	#if abs(objective_value - n_objective_value) < 30.0:
	#	complete = True
	#		break
	objective_value = n_objective_value

for metro in metros:
	p50, p95 = compute_performance(metro)
	print(f"Metro: {metro}, Final P50: {p50} ms, Final P95: {p95} ms")
	print(f"Final Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")