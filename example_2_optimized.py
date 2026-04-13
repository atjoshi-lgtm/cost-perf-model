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

ALL_METROS = ["SEA", "YVR", "PDX", "SLC", "SJC"]

MCH_PARENT_METROS = ["SJC", "SEA"]

MCH_PARENT_ASSIGNMENT = {
    "SEA": "SEA",
    "YVR": "SEA",
    "PDX": "SEA",
    "SLC": "SEA",
    "SJC": "SJC",
    }

METRO_NAMES = {
    "SEA": "Seattle",
    "YVR": "Vancouver",
    "PDX": "Portland",
    "SLC": "Salt_Lake_City",
    "SJC": "San_Jose",
    }

NEIGHBOR_METROS = {
    "SEA": ["SEA", "YVR", "PDX", "SLC", "SJC"],
    "YVR": ["YVR", "SEA"],
    "PDX": ["PDX", "SEA", "SJC"],
    "SLC": ["SLC", "SJC", "SEA"],
    "SJC": ["SJC", "SEA", "PDX", "SLC"]
}

FDS = {}
DISK_PROVISIONED = {}
COST = {}
PERF_PENALTY = {}  
HITRATES = {} 
TRY_HITRATES = {}
MAX_POSSIBLE_HITRATE = {}
for metro in METRO_NAMES:
	FDS[metro] = FootprintDescriptor.from_file(_FDS_DIR / f"{metro.lower()}.txt")
	DISK_PROVISIONED[metro] = 0.0
	COST[metro] = 0.0
	PERF_PENALTY[metro] = 0.0
	MAX_POSSIBLE_HITRATE[metro] = FDS[metro].find_max_possible_hitrate().hitrate
	print(f"Metro: {metro}, Max Possible Hitrate: {MAX_POSSIBLE_HITRATE[metro]}%")
	HITRATES[metro] = 0.0
	TRY_HITRATES[metro] = 0.0

MCH_PERFORMANCE_MODELS = {}
for metro in MCH_PARENT_METROS:
	mch_performance_model = MCHPerformanceModel(name=metro)
	tat_pdf = get_parent_tat_pdf(METRO_NAMES[metro])
	tat_pdf.plot(title=f"MCH TAT PDF for metro {metro}")
	plt.savefig(f"exp2/mch_tat_pdf_{metro}.png")
	plt.clf()
	mch_performance_model.set_mch_tat(tat_pdf)
	MCH_PERFORMANCE_MODELS[metro] = mch_performance_model

PERFORMANCE_MODELS = {}
TRAFFIC = {}

for metro in METRO_NAMES:

    metro_performance = MetroPerformanceWithMCH(name=metro, parent_model=MCH_PERFORMANCE_MODELS[MCH_PARENT_ASSIGNMENT[metro]], descriptor=FDS[metro])

    edge_tat_hit = get_edge_tat_pdf(METRO_NAMES[metro])
    print(f"Edge TAT Hit PDF for metro {metro}:")
    edge_tat_hit.plot(title=f"Edge TAT Hit PDF for metro {metro}")
    #plt.show()
    plt.savefig(f"exp2/edge_tat_hit_pdf_{metro}.png")
    plt.clf()
    metro_performance.set_edge_tat_hit(edge_tat_hit)

    parent_name = MCH_PARENT_ASSIGNMENT[metro]
    parent_name = METRO_NAMES[parent_name]

    rtt_cache_miss = get_midgress_rtt_pdf(parent_name, METRO_NAMES[metro])
    print(f"RTT PDF for cache miss from  metro {metro} to parent {parent_name}:")
    rtt_cache_miss.plot(title=f"RTT PDF for cache miss from  metro {metro} to parent {parent_name}")
    #plt.show()
    plt.savefig(f"exp2/rtt_cache_miss_pdf_{metro}.png")
    plt.clf()
    metro_performance.set_mch_rtt(rtt_cache_miss)
    if metro == "LAX":
        traffic = random.uniform(200000,400000)
    else:
        traffic = random.uniform(100000,200000)
	
    TRAFFIC[metro] = traffic

    for neighbor in NEIGHBOR_METROS[metro]:
        edge_rtt_pdf = get_rtt_pdf(METRO_NAMES[metro], client_metro_ids[METRO_NAMES[neighbor]])
        print(f"RTT PDF for edge {neighbor} to metro {metro}:")
        edge_rtt_pdf.plot(title=f"RTT PDF for {neighbor} to metro {metro}")
        #plt.show()
        plt.savefig(f"exp2/rtt_pdf_{neighbor}_to_{metro}.png")
        plt.clf()
        metro_performance.set_edge_rtt(neighbor, edge_rtt_pdf)
		
    PERFORMANCE_MODELS[metro] = metro_performance

cost_model = CaribouCostCalculator()
conn = Convolution()

metros = NEIGHBOR_METROS.keys()
INCOMING_TRAFFIC = defaultdict(dict)

for metro in metros:
	for neighbor in NEIGHBOR_METROS[metro]:
		if neighbor != metro:
			INCOMING_TRAFFIC[neighbor] = 0.4 * TRAFFIC[metro] / (len(NEIGHBOR_METROS[metro]) - 1)
		else:
			INCOMING_TRAFFIC[neighbor] = 0.6 * TRAFFIC[metro]

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
		if to_metro != metro:
			weights.append(float(0.4)/ (len(NEIGHBOR_METROS[metro]) - 1))
		else:
			weights.append(0.6)

	combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
	p50 = combined_pdf.millisecond_at_percentile(50)/1000
	p95 = combined_pdf.millisecond_at_percentile(95)/1000
	#combined_pdf.plot()
	return p50, p95

def penalty_function(p50, p95):
	return 100.0*((max(p50 - 28, 0))**2) + 100.0 * (max(p95 - 110, 0))

def compute_perf_penalty(metro):
	new_perf_penalty = {}
	for m in NEIGHBOR_METROS[metro]:
		p50, p95 = compute_performance(m)
		new_perf_penalty[m] = penalty_function(p50, p95)
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
	hitrate = FDS[metro].hitrate_for_cache(current_disk)
	new_hitrate = min(hitrate + 1, 100)
	TRY_HITRATES[metro] = new_hitrate
	new_disk = FDS[metro].nearest_cache_for_hitrate(new_hitrate)
	#print(f"Current hitrate: {hitrate}%, New hitrate: {new_hitrate}%")
	#print(f"Current disk: {current_disk} TB, New disk: {new_disk} TB")
	breakdown = cost_model.compute_monthly_cost(
		total_disk_required_tb=new_disk,
		hitrate_fraction=new_hitrate / 100.0,
		total_traffic_mbps=INCOMING_TRAFFIC[metro],
		free_disk_tb=0.0,
	)

	cost_differential += breakdown.total_cost - COST[metro]
	localized_perf_penalty = compute_perf_penalty(metro)
	#print(f"Localized performance penalty after increasing hitrate: {localized_perf_penalty}")
	for m in localized_perf_penalty:
		perf_differential += localized_perf_penalty[m] - PERF_PENALTY[m]

	# decrement the hitrate by 1 to compute the other side of the gradient
	previous_hitrate = max(hitrate - 1, 50)
	TRY_HITRATES[metro] = previous_hitrate
	previous_disk = FDS[metro].nearest_cache_for_hitrate(previous_hitrate)
	#print(f"Previous hitrate: {previous_hitrate}%, Previous disk: {previous_disk} TB")
	previous_breakdown = cost_model.compute_monthly_cost(
		total_disk_required_tb=previous_disk,
		hitrate_fraction=previous_hitrate / 100.0,
		total_traffic_mbps=INCOMING_TRAFFIC[metro],
		free_disk_tb=0.0,
	)

	cost_differential -= previous_breakdown.total_cost - COST[metro]
	#print(f"Cost differential: {cost_differential}")
	localized_perf_penalty = compute_perf_penalty(metro)
	#print(f"Localized performance penalty after decreasing hitrate: {localized_perf_penalty}")
	for m in localized_perf_penalty:
		perf_differential -= localized_perf_penalty[m] - PERF_PENALTY[m]
	#print("PERF penalty ", PERF_PENALTY)

	TRY_HITRATES[metro] = hitrate
	gradient = (cost_differential + perf_differential) / 2.0
	return gradient

for metro in metros:
	min_cost = INT32_MAX
	disk_tb = 0.0
	min_hitrate = 50
	min_breakdown = None

	for hitrate in range(50, 101, 1):

		breakdown = cost_model.compute_monthly_cost(
			total_disk_required_tb=FDS[metro].nearest_cache_for_hitrate(hitrate),
			hitrate_fraction=float(hitrate) / 100.0,
			total_traffic_mbps=INCOMING_TRAFFIC[metro],
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
	PERF_PENALTY[metro] = penalty_function(p50, p95)  
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
		if gradient < 0:
			# increase hitrate
			print("Gradient step: Increasing hitrate for metro:", metro)
			new_hitrate = min(TRY_HITRATES[metro] + 1, MAX_POSSIBLE_HITRATE[metro])
			if new_hitrate == TRY_HITRATES[metro]:
				print(f"Metro {metro} has reached maximum possible hitrate. Cannot increase further.")
				continue
		else:
			# decrease hitrate
			print("Gradient step: Decreasing hitrate for metro:", metro)
			new_hitrate = max(TRY_HITRATES[metro] - 1, 50)

		if new_hitrate != TRY_HITRATES[metro]:
			new_disk = FDS[metro].nearest_cache_for_hitrate(new_hitrate)
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
			#print(f"Updated performance penalties: {tmp_perf_penalty}")
			##Print the P50 and P95 after the hitrate change
			for m in ALL_METROS:
				p50, p95 = compute_performance(m)
				print(f"Metro: {m}, P50: {p50} ms, P95: {p95} ms")
			print("Cost after hitrate change:", COST[metro])
			print(f"Disk provisioned for metro {metro}: {round(DISK_PROVISIONED[metro]/1024/1024)} TB")
			#print(f"Updated performance penalties: {tmp_perf_penalty}")
			for m in tmp_perf_penalty:
				PERF_PENALTY[m] = tmp_perf_penalty[m]
			HITRATES[metro] = new_hitrate
			
	n_objective_value = compute_objective()
	print(f"Updated Objective Value: {n_objective_value:.2f} USD")
	if abs(objective_value - n_objective_value) < 100.0:
		complete = True
		break
	objective_value = n_objective_value

for metro in metros:
	p50, p95 = compute_performance(metro)
	print(f"Metro: {metro}, Final P50: {p50} ms, Final P95: {p95} ms")
	print(f"Final Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")