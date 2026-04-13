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

INT32_MAX = 2**31 - 1

_BASE_DIR = Path(__file__).resolve().parent
_FDS_DIR = _BASE_DIR / "FDS"

ALL_METROS = ["LAX", "LAS", "PHX"]

MCH_PARENT_METROS = ["LAX"]

MCH_PARENT_ASSIGNMENT = {
	"LAX": "LAX",
	"LAS": "LAX",
	"PHX": "LAX"
	}

METRO_NAMES = {
	"LAX": "Los_Angeles",
	"LAS": "Las_Vegas",
	"PHX": "Phoenix"
	}

NEIGHBOR_METROS = {
	"LAX": ["LAX", "LAS", "PHX"],
	"LAS": ["LAS", "LAX"],
	"PHX": ["PHX", "LAX"]
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
    DISK_STEP[metro] = MAX_CACHE_SPACE[metro] * 0.01
    print(f"Metro: {metro}, Max Possible Hitrate: {MAX_POSSIBLE_HITRATE[metro]}%")
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

# I have a table in SERVEDFROM_DATA/metros_servedfrom_copy.db called traffic_summary with columns asn_metro|bucket|bw_metro|quarter|total_traffic
# Here asn_metro is the metro where the traffic originates, for bucket always use Disney_Videos, bw_metro is the metro where the traffic is served from, quarter is 1Q26 and total_traffic is the total traffic in Gbps. 
# Wite a function that given a end user metro and a serving metro, returns the total traffic in Gbps for the Disney_Videos bucket for 1Q26. Use sqlite3 to query the database.

def get_traffic(asn_metro: str, bw_metro: str) -> float:
	import sqlite3

	db_path = _BASE_DIR / "SERVEDFROM_DATA" / "metros_servedfrom_copy.db"
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
		return result[0] * 1000  # total_traffic in Mbps
	else:
		return 0.0  # Return 0 if no data is found for the given metros


PERFORMANCE_MODELS = {}
TRAFFIC = defaultdict(lambda: defaultdict(float))

for metro in METRO_NAMES:

	metro_performance = MetroPerformanceWithMCH(name=metro, parent_model=MCH_PERFORMANCE_MODELS[MCH_PARENT_ASSIGNMENT[metro]], descriptor=FDS[metro])

	edge_tat_hit = get_edge_tat_pdf(METRO_NAMES[metro])
	#print(f"Edge TAT Hit PDF for metro {metro}:")
	edge_tat_hit.plot(title=f"Edge TAT Hit PDF for metro {metro}")
	#plt.show()
	#plt.savefig(f"edge_tat_hit_pdf_{metro}.png")
	#plt.clf()
	metro_performance.set_edge_tat_hit(edge_tat_hit)

	parent_name = MCH_PARENT_ASSIGNMENT[metro]
	parent_name = METRO_NAMES[parent_name]

	rtt_cache_miss = get_midgress_rtt_pdf(parent_name, METRO_NAMES[metro])
	#print(f"RTT PDF for cache miss from  metro {metro} to parent {parent_name}:")
	rtt_cache_miss.plot(title=f"RTT PDF for cache miss from  metro {metro} to parent {parent_name}")
	#plt.show()
	#plt.savefig(f"rtt_cache_miss_pdf_{metro}.png")
	#plt.clf()
	metro_performance.set_mch_rtt(rtt_cache_miss)

	for neighbor in NEIGHBOR_METROS[metro]:
		TRAFFIC[metro][neighbor] = get_traffic(asn_metro=METRO_NAMES[neighbor], bw_metro=METRO_NAMES[metro])
		edge_rtt_pdf = get_rtt_pdf(METRO_NAMES[metro], client_metro_ids[METRO_NAMES[neighbor]])
		#print(f"RTT PDF for edge {neighbor} to metro {metro}:")
		#edge_rtt_pdf.plot(title=f"RTT PDF for {neighbor} to metro {metro}")
		#plt.show()
		#plt.savefig(f"rtt_pdf_{neighbor}_to_{metro}.png")
		#plt.clf()
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
	return 50.0*((max(p50 - 23, 0))) + 50.0 * (max(p95 - 105, 0))

def compute_perf_penalty(metro):
	new_perf_penalty = {}
	for m in NEIGHBOR_METROS[metro]:
		p50, p95 = compute_performance(m)
		new_perf_penalty[m] = (float(TRAFFIC_FROM[m])/total_traffic) * penalty_function(p50, p95)
	return new_perf_penalty

def compute_objective():
	total_cost = 0.0
	penalty = 0.0
	for metro in metros:
		total_cost += COST[metro]
		penalty += PERF_PENALTY[metro]
	return total_cost + penalty

def compute_gradient(metro):
    cost_differential = 0.0
    perf_differential = 0.0
    current_disk = DISK_PROVISIONED[metro]
    step = DISK_STEP[metro]

    # compute gradient at +step disk
    new_disk = min(current_disk + step, MAX_CACHE_SPACE[metro])
    new_hitrate = FDS[metro].hitrate_for_cache(new_disk)
    TRY_HITRATES[metro] = new_hitrate
    breakdown = cost_model.compute_monthly_cost(
        total_disk_required_tb=new_disk,
        hitrate_fraction=new_hitrate / 100.0,
        total_traffic_mbps=INCOMING_TRAFFIC[metro],
        free_disk_tb=0.0,
    )
    cost_differential += breakdown.total_cost - COST[metro]
    localized_perf_penalty = compute_perf_penalty(metro)
    for m in localized_perf_penalty:
        perf_differential += localized_perf_penalty[m] - PERF_PENALTY[m]

    # compute gradient at -step disk
    previous_disk = max(current_disk - step, 0.0)
    previous_hitrate = FDS[metro].hitrate_for_cache(previous_disk)
    TRY_HITRATES[metro] = previous_hitrate
    previous_breakdown = cost_model.compute_monthly_cost(
        total_disk_required_tb=previous_disk,
        hitrate_fraction=previous_hitrate / 100.0,
        total_traffic_mbps=INCOMING_TRAFFIC[metro],
        free_disk_tb=0.0,
    )
    cost_differential -= previous_breakdown.total_cost - COST[metro]
    localized_perf_penalty = compute_perf_penalty(metro)
    for m in localized_perf_penalty:
        perf_differential -= localized_perf_penalty[m] - PERF_PENALTY[m]

    TRY_HITRATES[metro] = FDS[metro].hitrate_for_cache(current_disk)
    gradient = (cost_differential + perf_differential) / 2.0
    return gradient

# for metro in metros:
# 	min_cost = INT32_MAX
# 	disk_tb = 0.0
# 	min_hitrate = 50
# 	min_breakdown = None
#
# 	for hitrate in range(1, 101, 1):
#
# 		breakdown = cost_model.compute_monthly_cost(
# 			total_disk_required_tb=FDS[metro].nearest_cache_for_hitrate(hitrate),
# 			hitrate_fraction=float(hitrate) / 100.0,
# 			total_traffic_mbps=INCOMING_TRAFFIC[metro],
# 			free_disk_tb=DISK_PROVISIONED[metro],
# 		)
#
# 		if hitrate == 50 or breakdown.total_cost < min_cost:
# 			min_cost = breakdown.total_cost
# 			min_hitrate = hitrate
# 			min_breakdown = breakdown
# 			disk_tb = FDS[metro].nearest_cache_for_hitrate(hitrate)
#
# 	DISK_PROVISIONED[metro] += disk_tb
# 	COST[metro] = min_cost
# 	HITRATES[metro] = min_hitrate
# 	TRY_HITRATES[metro] = min_hitrate
# 	
# 	#print(f"Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {min_hitrate}%, Cost: {min_cost:.2f} USD")

for metro in metros:
	min_disk_tb = min(point.cache_space for point in FDS[metro]._points_sorted_by_cache)
	min_hitrate = FDS[metro].hitrate_for_cache(min_disk_tb)
	breakdown = cost_model.compute_monthly_cost(
		total_disk_required_tb=min_disk_tb,
		hitrate_fraction=min_hitrate / 100.0,
		total_traffic_mbps=INCOMING_TRAFFIC[metro],
		free_disk_tb=DISK_PROVISIONED[metro],
	)

	DISK_PROVISIONED[metro] = min_disk_tb
	COST[metro] = breakdown.total_cost
	HITRATES[metro] = min_hitrate
	TRY_HITRATES[metro] = min_hitrate

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
                total_traffic_mbps=INCOMING_TRAFFIC[metro],
                free_disk_tb=0.0,
            )

            COST[metro] = breakdown.total_cost
            DISK_PROVISIONED[metro] = new_disk
            TRY_HITRATES[metro] = new_hitrate
            tmp_perf_penalty = compute_perf_penalty(metro)
            for m in ALL_METROS:
                p50, p95 = compute_performance(m)
                print(f"Metro: {m}, P50: {p50} ms, P95: {p95} ms")
            print("Cost after disk change:", COST[metro])
            print(f"Disk provisioned for metro {metro}: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {TRY_HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")
            for m in tmp_perf_penalty:
                PERF_PENALTY[m] = tmp_perf_penalty[m]
            HITRATES[metro] = new_hitrate

    n_objective_value = compute_objective()
    print(f"Updated Objective Value: {n_objective_value:.2f} USD, Total cost: {sum(COST[metro] for metro in metros):.2f} USD, Total performance penalty: {sum(PERF_PENALTY[metro] for metro in metros):.2f} USD")
    #if abs(objective_value - n_objective_value) < 30.0:
    #   complete = True
    #   break
    objective_value = n_objective_value

for metro in metros:
	p50, p95 = compute_performance(metro)
	print(f"Metro: {metro}, Final P50: {p50} ms, Final P95: {p95} ms")
	print(f"Final Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")