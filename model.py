"""Application-level model objects for footprint descriptors."""

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
FDS = {}
DISK_PROVISIONED = {}
for metro in ALL_METROS:
	FDS[metro] = FootprintDescriptor.from_file(_FDS_DIR / f"{metro.lower()}.txt")
	DISK_PROVISIONED[metro] = 0.0

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
	edge_tat_hit = gaussian_pdf(
		mean=mean,
		stddev=stddev,
		lower_ms=0,
		upper_ms=200,
	)
	metro_performance.set_edge_tat_hit(edge_tat_hit)
	
	mean = random.uniform(15,25)
	stddev = random.uniform(5, 20)
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
		else:
			mean = random.uniform(15, 25)
			stddev = random.uniform(5, 15)

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
# shuffle the metros
metros = list(metros)
random.shuffle(metros)

total_cost = 0.0

BW_COST_ACCOUNTED = defaultdict(float)
## Solve for each metro first without considering the neighbors, considering only self-loop traffic
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
			total_traffic_mbps=TRAFFIC[metro] * 0.8,
			free_disk_tb=DISK_PROVISIONED[metro],
		)

		if hitrate == 50 or breakdown.total_cost < min_cost:
			min_cost = breakdown.total_cost
			min_hitrate = hitrate
			min_breakdown = breakdown
			disk_tb = FDS[metro].nearest_cache_for_hitrate(hitrate)

	total_cost += min_cost
	DISK_PROVISIONED[metro] += disk_tb
	BW_COST_ACCOUNTED[metro] = min_breakdown.midgress_cost

	print(f"Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB, Hitrate: {min_hitrate}%, Cost: {min_cost:.2f} USD")


print(f"After self-loop optimization, total cost: {total_cost:.2f} USD")
print("BW cost accounted per metro:")
for metro in BW_COST_ACCOUNTED:
	print(f"Metro: {metro}, BW Cost Accounted: {BW_COST_ACCOUNTED[metro]:.2f} USD")
'''
for metro in metros:
	TTFB = {}
	COST = {}
	total_traffic = TRAFFIC[metro]

	for neighbor in NEIGHBOR_METROS[metro]:
		TTFB[neighbor] = {}
		COST[neighbor] = {}
		
		free_bw = 0.0

		if neighbor == metro:
			traffic = total_traffic * 0.8
			free_bw = BW_COST_ACCOUNTED[metro]
		else:
			traffic = total_traffic * (0.2 / (len(NEIGHBOR_METROS[metro]) - 1))

		for hitrate in range(50, 101, 5):

			COST[neighbor][hitrate] = cost_model.compute_monthly_cost(
				total_disk_required_tb=FDS[neighbor].nearest_cache_for_hitrate(hitrate),
				hitrate_fraction=float(hitrate) / 100.0,
				total_traffic_mbps=traffic,
				free_disk_tb=DISK_PROVISIONED[neighbor],
				free_bw=free_bw
			)

			TTFB[neighbor][hitrate] = PERFORMANCE_MODELS[neighbor].get_ttfb_pdf(
				from_metro=metro,
				hitrate=hitrate,
				conn=conn,
			)

	neighbors = NEIGHBOR_METROS[metro]
	min_cost = INT32_MAX
	min_ttfb = None

	for combo in product(range(50, 101, 5), repeat=len(neighbors)):

		neighbor_hitrates = dict(zip(neighbors, combo))
		cost_combo = 0
		weights = []
		ttfbs = []

		for neighbor in neighbors:
			cost_combo += COST[neighbor][neighbor_hitrates[neighbor]].total_cost
			if neighbor == metro:
				weight = 0.8
			else:
				weight = 0.2 / (len(NEIGHBOR_METROS[metro]) - 1)

			weights.append(weight)
			ttfbs.append(TTFB[neighbor][neighbor_hitrates[neighbor]])

		# Compute weighted average TTFB
		combined_ttfb = weighted_pdf_sum(ttfbs, weights)
		p50 = combined_ttfb.millisecond_at_percentile(50)
		p95 = combined_ttfb.millisecond_at_percentile(95)

		if cost_combo < min_cost:
			min_cost = cost_combo
			min_hitrates = neighbor_hitrates
			min_ttfb = combined_ttfb

	total_cost += min_cost

	print(f"Metro: {metro}, Min Cost: {min_cost:.2f} USD, Hitrates: {min_hitrates}, TTFB P50: {min_ttfb.millisecond_at_percentile(50)} ms, P95: {min_ttfb.millisecond_at_percentile(95)} ms")
	for neighbor in min_hitrates:
		extra_disk = max(0.0, FDS[neighbor].nearest_cache_for_hitrate(min_hitrates[neighbor]) - DISK_PROVISIONED[neighbor])
		DISK_PROVISIONED[neighbor] += extra_disk

	for metro in DISK_PROVISIONED:
		print(f"Metro: {metro}, Provisioned Disk: {round(DISK_PROVISIONED[metro]/1024/1024)} TB")

	# Write TTFB[neighbor][hitrate] to a file
	output_file = _BASE_DIR / f"ttfb_{metro.lower()}.txt"
	with open(output_file, "w") as f:
		f.write("NeighborMetro,HitratePercentage,P50ms,P95ms\n")
		for neighbor in TTFB:
			for hitrate in TTFB[neighbor]:
				p50 = TTFB[neighbor][hitrate].millisecond_at_percentile(50)
				p95 = TTFB[neighbor][hitrate].millisecond_at_percentile(95)
				f.write(f"{neighbor},{hitrate},{p50},{p95}\n")

	# Write COST[neighbor][hitrate] to a file
	output_file = _BASE_DIR / f"cost_{metro.lower()}.txt"
	with open(output_file, "w") as f:
		f.write("NeighborMetro,HitratePercentage,TotalCostUSD,DepreciationCostUSD,ColocationCostUSD,MidgressCostUSD,MachinesRequired,ProvidedDiskTB,MissTrafficMbps\n")
		for neighbor in COST:
			for hitrate in COST[neighbor]:
				breakdown = COST[neighbor][hitrate]
				f.write(f"{neighbor},{hitrate},{breakdown.total_cost},{breakdown.depreciation_cost},{breakdown.colocation_cost},{breakdown.midgress_cost},{breakdown.machines_required},{breakdown.provided_disk_tb},{breakdown.miss_traffic_mbps}\n")

print(f"After complete optimization, total cost: {total_cost:.2f} USD")






