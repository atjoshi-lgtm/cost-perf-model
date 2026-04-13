from __future__ import annotations

from pathlib import Path
import random

from fds import FootprintDescriptor
from cost import CaribouCostCalculator
from perf import MetroPerformance
from probability import Convolution, gaussian_pdf, weighted_pdf_sum
from analyse import *
from itertools import product
from collections import defaultdict
import sys

INT32_MAX = 2**31 - 1

_BASE_DIR = Path(__file__).resolve().parent
_FDS_DIR = _BASE_DIR / "FDS"

ALL_METROS = ["SEA", "YVR", "SJC", "LAX", "MEX", "DFW"]

METRO_NAMES = {
	"SEA": "Seattle",
	"YVR": "Vancouver",
	"SJC": "San_Jose",
	"LAX": "Los_Angeles",
	"MEX": "Mexico_City",
	"DFW": "Dallas",
	"HYD": "Hyderabad",
	"BLR": "Bangalore",
	"BOG": "Bogota",
	"PIT": "Pittsburgh",
	"SLC": "Salt_Lake_City",
	"PHX": "Phoenix",
	"BOM": "Mumbai",
	"MAA": "Chennai",
	"DEL": "Delhi",
	}

FDS = {}
DISK_PROVISIONED = {}
COST = {}
PERF_PENALTY = {}  
HITRATES = {} 
TRY_HITRATES = {}
metro1 = "SEA"
for metro in METRO_NAMES:
	FDS[metro] = FootprintDescriptor.from_file(_FDS_DIR / f"{metro1.lower()}.txt")
	DISK_PROVISIONED[metro] = 0.0
	COST[metro] = 0.0
	PERF_PENALTY[metro] = 0.0
	HITRATES[metro] = 0.0
	TRY_HITRATES[metro] = 0.0

NEIGHBOR_METROS = {
	"SEA": ["SEA", "YVR", "SJC"],
	"YVR": ["YVR", "SEA", "SJC"],
	"SJC": ["SJC", "SEA", "LAX", "YVR"],
	"LAX": ["LAX", "SJC", "MEX", "DFW"],
	"MEX": ["MEX", "LAX", "DFW"],
	"DFW": ["DFW", "LAX", "MEX"],
	"BOM": ["BOM", "HYD", "BLR", "MAA"],
	"HYD": ["HYD", "BOM", "BLR", "MAA"],
	"BLR": ["BLR", "DEL"],
	"MAA": ["MAA", "BOM", "HYD", "BLR"]
	}

PERFORMANCE_MODELS = {}
TRAFFIC = {}

for metro in NEIGHBOR_METROS:

	metro_performance = MetroPerformance(name=metro, descriptor=FDS["SEA"])
	edge_tat_hit = get_edge_tat_pdf(METRO_NAMES[metro])
	#edge_tat_hit = edge_tat_hit.to_microsecond_pdf()
	metro_performance.set_edge_tat_hit(edge_tat_hit)
	
	edge_tat_miss = get_edge_tat_pdf(METRO_NAMES[metro], cache_hit_type=0)  
	#edge_tat_miss = edge_tat_miss.to_microsecond_pdf()
	metro_performance.set_edge_tat_miss(edge_tat_miss)
	
	traffic = random.uniform(200000,400000)
	TRAFFIC[metro] = traffic
	
	rtt_pdfs = []
	rtt_pdf_weights = []
	for neighbor in NEIGHBOR_METROS[metro]:
		edge_rtt_pdf = get_rtt_pdf(METRO_NAMES[metro], client_metro_ids[METRO_NAMES[neighbor]])
		rtt_pdfs.append(edge_rtt_pdf)
		if neighbor != metro:
			rtt_pdf_weights.append(float(0.3)/ (len(NEIGHBOR_METROS[metro]) - 1))
		else:
			rtt_pdf_weights.append(0.7)
		#edge_rtt_pdf = edge_rtt_pdf.to_microsecond_pdf()
	combined_rtt_pdf = weighted_pdf_sum(rtt_pdfs, rtt_pdf_weights)
	metro_performance.set_edge_rtt(metro, combined_rtt_pdf)
	
	PERFORMANCE_MODELS[metro] = metro_performance

cost_model = CaribouCostCalculator()
conn = Convolution()

metros = NEIGHBOR_METROS.keys()
INCOMING_TRAFFIC = defaultdict(dict)

for metro in metros:
	for neighbor in NEIGHBOR_METROS[metro]:
		if neighbor != metro:
			INCOMING_TRAFFIC[neighbor] = TRAFFIC[metro] / (len(NEIGHBOR_METROS[metro]) - 1)
		else:
			INCOMING_TRAFFIC[neighbor] = 0.8 * TRAFFIC[metro]

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
		#pdf_microsecond = pdf.to_microsecond_pdf()

		ttfb_pdfs.append(pdf)
		if to_metro != metro:
			weights.append(float(0.2)/ (len(NEIGHBOR_METROS[metro]) - 1))
		else:
			weights.append(0.8)

	combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
	p50 = combined_pdf.millisecond_at_percentile(50)/1000
	p95 = combined_pdf.millisecond_at_percentile(95)/1000
	#combined_pdf.plot()
	return p50, p95

## Ignore the traffic from neighboring metros. Just compute P50, P95 using the performance model for the metro itself. This will give us a more localized view of the performance and how it changes with hitrate adjustments.
def compute_performance_localized(metro):

	p50, p95 = 0.0, 0.0
	ttfb_pdfs = []
	weights = []
	for to_metro in NEIGHBOR_METROS[metro]:
		hitrate = TRY_HITRATES[to_metro]
		if to_metro == metro:
			pdf = PERFORMANCE_MODELS[to_metro].get_ttfb_pdf(
				from_metro=metro,
				hitrate=hitrate,
				conn=conn,
			)
			pdf_microsecond = pdf.to_microsecond_pdf()
			ttfb_pdfs.append(pdf_microsecond)
			weights.append(1.0)

	combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
	p50 = combined_pdf.millisecond_at_percentile(50)/1000
	p95 = combined_pdf.millisecond_at_percentile(95)/1000
	return p50, p95

def penalty_function(p50, p95):
	return 100.0*((max(p50 - 36, 0))**2) + 100.0 * (max(p95 - 110, 0))

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
	new_hitrate = min(hitrate + 0.1, 100)
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
	previous_hitrate = max(hitrate - 0.1, 50)
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

'''
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
	#print(f"Metro: {metro}, P50: {p50} ms, P95: {p95} ms")
	PERF_PENALTY[metro] = penalty_function(p50, p95)  
	#print(f"Metro: {metro}, Initial Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")
'''

# I want you to write a function that does the following. For each metro, it iterates through the possible hitrates from 50 to 100 in increments of 0.1 and 
# computes the P50 and P95 latencies using the performance model. It then plots both as a function of the hitrate. The function should be called plot_performance_vs_hitrate and should take the metro name as an argument.
def plot_performance_vs_hitrate(metro):
	hitrates = []
	p50s = []
	p95s = []
	for hitrate in range(25, 1001, 25):
		TRY_HITRATES[metro] = hitrate / 10.0
		p50, p95 = compute_performance_localized(metro)
		hitrates.append(hitrate / 10.0)
		p50s.append(p50)
		p95s.append(p95)

	import matplotlib.pyplot as plt

	plt.figure(figsize=(10, 6))
	plt.plot(hitrates, p50s, label='P50 Latency (ms)')
	plt.plot(hitrates, p95s, label='P95 Latency (ms)')
	plt.xlabel('Hitrate (%)')
	plt.ylabel('Latency (ms)')
	plt.title(f'Performance vs Hitrate for Metro: {metro}')
	plt.legend()
	plt.grid()
	plt.savefig(f"{metro}_performance_vs_hitrate.png")
	plt.clf()
	

for metro in ["BLR"]:
	plot_performance_vs_hitrate(metro)

sys.exit()

objective_value = compute_objective()
print(f"Initial Objective Value: {objective_value:.2f} USD")

complete = False

i = 0
while True:
	gradient_sum = 0.0
	for metro in metros:
		i += 1
		gradient = compute_gradient(metro)
		gradient_sum += gradient
		print(f"Gradient for metro {metro}: {gradient}")
		if gradient < 0:
			# increase hitrate
			print("Gradient step: Increasing hitrate for metro:", metro)
			new_hitrate = min(TRY_HITRATES[metro] + 0.5, 100)
		else:
			# decrease hitrate
			print("Gradient step: Decreasing hitrate for metro:", metro)
			new_hitrate = max(TRY_HITRATES[metro] - 0.5, 50)

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
			print(f"Updated performance penalties: {tmp_perf_penalty}")
			p50, p95 = compute_performance(metro)
			print(f"Metro: {metro}, Final P50: {p50} ms, Final P95: {p95} ms, disk provisioned: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, hitrate: {HITRATES[metro]}%")
			for neighbor in NEIGHBOR_METROS[metro]:
				neighbor_p50, neighbor_p95 = compute_performance(neighbor)
				print(f"Neighbor Metro: {neighbor}, P50: {neighbor_p50} ms, P95: {neighbor_p95} ms")
			print("Cost after hitrate change:", COST[metro])
			for m in tmp_perf_penalty:
				PERF_PENALTY[m] = tmp_perf_penalty[m]
			HITRATES[metro] = new_hitrate
			
	n_objective_value = compute_objective()
	print(f"Updated Objective Value: {n_objective_value:.2f} USD, Gradient Sum: {gradient_sum:.2f}")

	if gradient_sum <= 100:
		print("Gradient sum is small, stopping optimization.")
		complete = True
		break

	#if abs(objective_value - n_objective_value)/objective_value < 0.005:
	#		complete = True
	#	break
	objective_value = n_objective_value

for metro in metros:
	p50, p95 = compute_performance(metro)
	print(f"Metro: {metro}, Final P50: {p50} ms, Final P95: {p95} ms")
	print(f"Final Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")