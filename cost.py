"""Cost modeling utilities for the Caribou storage machines."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional

from fds import FootprintDescriptor


# -------------------------
# Caribou hardware constants
# -------------------------

CARIBOU_MACHINE_NAME = "Caribou"
CARIBOU_MACHINE_COST_USD = 9_600.0 * 1.6
CARIBOU_DISK_COST_RATIO = 0.40
CARIBOU_DEPRECIATION_MONTHS = 60.0
CARIBOU_DISK_CAPACITY_TB = 35.0

CARIBOU_POWER_RATING_WATTS = 585.0
CARIBOU_AVERAGE_POWER_USAGE_WATTS = 429.0

KW_INFRA_COST_PER_KW_MONTH = 80.0
MIDGRESS_COST_PER_Mbps_MONTH_NONMCH = 0.07  # 7 cents per Mbps-month
MIDGRESS_COST_PER_Mbps_MONTH_MCH = 0.01  # 1 cent per Mbps-month for MCH traffic

# The energy provider charges on a metered basis per kWh.  The exact rate can
# vary between deployments, so the calculator exposes it as a configurable
# parameter.  We keep a conservative default that can be overridden.
DEFAULT_METERED_POWER_RATE_PER_KWH = 0.12

# Average number of hours in a month.  This value can be tuned if a specific
# calendar month is required.
DEFAULT_HOURS_PER_MONTH = 24 * 30.4375  # ≈ 730.5 hours


@dataclass(frozen=True)
class CaribouCostBreakdown:
	"""Detailed accounting for a set of Caribou machines."""

	total_cost: float
	depreciation_cost: float
	colocation_cost: float
	midgress_cost: float
	parent_service_cost: float
	machines_required: int
	parent_machines_required: float
	provided_disk_tb: float
	hitrate_percentage: float
	miss_traffic_mbps: float


class CaribouCostCalculator:
	"""Compute monthly costs for Caribou storage capacity."""

	def __init__(
		self,
		*,
		metered_power_rate_per_kwh: float = DEFAULT_METERED_POWER_RATE_PER_KWH,
		hours_per_month: float = DEFAULT_HOURS_PER_MONTH,
	) -> None:
		self.metered_power_rate_per_kwh = metered_power_rate_per_kwh
		self.hours_per_month = hours_per_month

		disk_capex = CARIBOU_MACHINE_COST_USD * CARIBOU_DISK_COST_RATIO
		self._monthly_depreciation_per_machine = disk_capex / CARIBOU_DEPRECIATION_MONTHS
		parent_capex = CARIBOU_MACHINE_COST_USD * (1 - CARIBOU_DISK_COST_RATIO)
		self._monthly_parent_service_cost_per_machine = parent_capex / CARIBOU_DEPRECIATION_MONTHS

		self._monthly_kw_infra_cost_per_machine = (
			CARIBOU_POWER_RATING_WATTS / 1000.0
		) * KW_INFRA_COST_PER_KW_MONTH

		self._monthly_metered_power_cost_per_machine = (
			(CARIBOU_AVERAGE_POWER_USAGE_WATTS / 1000.0)
			* self.hours_per_month
			* self.metered_power_rate_per_kwh
		)

	def compute_monthly_cost(
		self,
		total_disk_required_tb: float,
		hitrate_fraction: float,
		total_traffic_mbps: float,
		free_disk_tb: Optional[float] = 0.0,
		free_bw: Optional[float] = 0.0,
		is_mch_in_metro: bool = False,
	) -> CaribouCostBreakdown:
		"""Return the monthly cost for the requested Caribou disk footprint.

		Parameters
		----------
		total_disk_required_tb:
			The desired aggregate Caribou disk capacity in terabytes.
		descriptor:
			Footprint descriptor that provides hit-rate statistics for the cache
			space corresponding to ``total_disk_required_tb``.  The value is assumed
			to use the same units the descriptor expects.
		total_traffic_mbps:
			Total traffic entering the Caribou tier, measured in Mbps.
		"""

		Effcap = 24966 # This is how much traffic in mbps a caribou server can handle. 

		if total_disk_required_tb < 0:
			raise ValueError("Disk requirement cannot be negative")
		if total_traffic_mbps < 0:
			raise ValueError("Traffic cannot be negative")
		if free_disk_tb < 0:
			raise ValueError("Free disk capacity cannot be negative")

		effective_paid_disk_tb = max(total_disk_required_tb - free_disk_tb, 0.0)/1024/1024

		machines_required = 0
		if total_disk_required_tb > 0:
			machines_required = effective_paid_disk_tb / CARIBOU_DISK_CAPACITY_TB

		monthly_depreciation_cost = machines_required * self._monthly_depreciation_per_machine
		monthly_colocation_cost = machines_required * (
			self._monthly_kw_infra_cost_per_machine + self._monthly_metered_power_cost_per_machine
		)

		miss_traffic_mbps = (1.0 - hitrate_fraction) * total_traffic_mbps
		parent_machines_required = float(miss_traffic_mbps) / Effcap if Effcap > 0 else 0.0
		parent_service_cost = parent_machines_required * self._monthly_parent_service_cost_per_machine
		if is_mch_in_metro:
			midgress_cost = (miss_traffic_mbps * MIDGRESS_COST_PER_Mbps_MONTH_MCH) - free_bw
		else:
			midgress_cost = (miss_traffic_mbps * MIDGRESS_COST_PER_Mbps_MONTH_NONMCH) - free_bw

		total_cost = monthly_depreciation_cost + monthly_colocation_cost + midgress_cost + parent_service_cost
		provided_disk_tb = machines_required * CARIBOU_DISK_CAPACITY_TB

		return CaribouCostBreakdown(
			total_cost=total_cost,
			depreciation_cost=monthly_depreciation_cost,
			colocation_cost=monthly_colocation_cost,
			midgress_cost=midgress_cost,
			parent_service_cost=parent_service_cost,
			machines_required=machines_required,
			parent_machines_required=parent_machines_required,
			provided_disk_tb=provided_disk_tb,
			hitrate_percentage=hitrate_fraction * 100,
			miss_traffic_mbps=miss_traffic_mbps,
		)
