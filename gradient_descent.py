from __future__ import annotations

from pathlib import Path
import random

from fds import FootprintDescriptor
from cost import CaribouCostCalculator
from perf import MetroPerformance
from probability import Convolution, gaussian_pdf, weighted_pdf_sum
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
	"DFW": "Dallas"
	}

FDS = {}
DISK_PROVISIONED = {}
COST = {}
PERF_PENALTY = {}  
HITRATES = {} 
TRY_HITRATES = {}
for metro in ALL_METROS:
	FDS[metro] = FootprintDescriptor.from_file(_FDS_DIR / f"{metro.lower()}.txt")
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
	}

PERFORMANCE_MODELS = {}
TRAFFIC = {}

for metro in NEIGHBOR_METROS:

	metro_performance = MetroPerformance(name=metro, descriptor=FDS[metro])
	
	mean = random.uniform(2, 6)
	stddev = random.uniform(3, 6)

	mean = 3
	stddev = 2

	edge_tat_hit = gaussian_pdf(
		mean=mean,
		stddev=stddev,
		lower_ms=0,
		upper_ms=200,
	)
	metro_performance.set_edge_tat_hit(edge_tat_hit)
	
	mean = random.uniform(15,25)
	stddev = random.uniform(5, 20)

	mean = 20
	stddev = 10
	edge_tat_miss = gaussian_pdf(
		mean=mean,
		stddev=stddev,
		lower_ms=0,
		upper_ms=200,
	)    
	metro_performance.set_edge_tat_miss(edge_tat_miss)
	
	traffic = random.uniform(200000,400000)
	TRAFFIC[metro] = traffic
	
	for neighbor in NEIGHBOR_METROS[metro]:
		# Sample a random mean between 20 and 40 ms and stddev between 20 and 30ms
		if neighbor != metro:
			mean = random.uniform(25, 40)
			stddev = random.uniform(20, 30)
			mean = 30
			stddev = 20
		else:
			mean = random.uniform(15, 25)
			stddev = random.uniform(5, 15)
			mean = 20
			stddev = 10

		edge_rtt_pdf = gaussian_pdf(
			mean=mean,
			stddev=stddev,
			lower_ms=0,
			upper_ms=200,
		)
		metro_performance.set_edge_rtt(neighbor, edge_rtt_pdf)
	
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
		pdf_microsecond = pdf.to_microsecond_pdf()

		ttfb_pdfs.append(pdf_microsecond)
		if to_metro != metro:
			weights.append(float(0.2)/ (len(NEIGHBOR_METROS[metro]) - 1))
		else:
			weights.append(0.8)

	combined_pdf = weighted_pdf_sum(ttfb_pdfs, weights)
	p50 = combined_pdf.millisecond_at_percentile(50)/1000
	p95 = combined_pdf.millisecond_at_percentile(95)/1000
	#combined_pdf.plot()
	return p50, p95

def penalty_function(p50, p95):
	return 100.0*((max(p50 - 23, 0))**2) + 100.0 * (max(p95 - 50, 0))

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
	#print(f"Metro: {metro}, P50: {p50} ms, P95: {p95} ms")
	PERF_PENALTY[metro] = penalty_function(p50, p95)  
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
			print("Cost after hitrate change:", COST[metro])
			for m in tmp_perf_penalty:
				PERF_PENALTY[m] = tmp_perf_penalty[m]
			HITRATES[metro] = new_hitrate
			
	n_objective_value = compute_objective()
	print(f"Updated Objective Value: {n_objective_value:.2f} USD")

	if abs(objective_value - n_objective_value) < 30.0:
		complete = True
		break
	objective_value = n_objective_value

for metro in metros:
	p50, p95 = compute_performance(metro)
	print(f"Metro: {metro}, Final P50: {p50} ms, Final P95: {p95} ms")
	print(f"Final Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {HITRATES[metro]}%, Cost: {COST[metro]:.2f} USD, Performance Penalty: {PERF_PENALTY[metro]:.2f} USD")