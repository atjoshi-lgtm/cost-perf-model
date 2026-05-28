"""Utilities for working with footprint descriptor (FDS) files.

This module exposes :class:`FootprintDescriptor`, a helper that understands the
structure of the footprint data files found in the ``FDS`` directory. Each file
contains five whitespace-separated columns; the class cares about the first
column (cache space) and the third column (hit-rate percentage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class FootprintPoint:
	"""A single cache space / hit-rate observation."""

	cache_space: int
	hitrate: float


class FootprintDescriptor:
	"""Footprint descriptor lookups for cache space and hit-rate values.

	Parameters
	----------
	points:
		Iterable of :class:`FootprintPoint` entries describing the descriptor.

	Notes
	-----
	The constructor is rarely used directly; prefer :meth:`from_file` which
	knows how to parse the on-disk FDS files.
	"""

	def __init__(self, points: Iterable[FootprintPoint]) -> None:
		unique_by_cache: MutableMapping[int, FootprintPoint] = {}

		for point in points:
			unique_by_cache.setdefault(point.cache_space, point)

		if not unique_by_cache:
			raise ValueError("FootprintDescriptor requires at least one data point")

		# Preserve natural order by cache space for exact lookups and by hitrate
		# for nearest-neighbour searches.
		self._points_by_cache: Mapping[int, FootprintPoint] = dict(sorted(
			unique_by_cache.items(), key=lambda item: item[0]
		))

		points_by_hitrate: Dict[float, FootprintPoint] = {}
		for point in self._points_by_cache.values():
			existing = points_by_hitrate.get(point.hitrate)
			if existing is None or point.cache_space < existing.cache_space:
				points_by_hitrate[point.hitrate] = point

		self._points_sorted_by_cache: List[FootprintPoint] = list(self._points_by_cache.values())
		self._cache_spaces: List[int] = [point.cache_space for point in self._points_sorted_by_cache]

		self._points_sorted_by_hitrate: List[FootprintPoint] = sorted(
			points_by_hitrate.values(), key=lambda point: point.hitrate
		)

		self._hitrates: List[float] = [point.hitrate for point in self._points_sorted_by_hitrate]

	@staticmethod
	def _parse_lines(lines: Iterable[str], *, source: str) -> List[FootprintPoint]:
		points: List[FootprintPoint] = []
		iterator = iter(lines)

		# Skip the first line (header/meta data as per the FDS convention).
		next(iterator, None)

		for line_number, raw_line in enumerate(iterator, start=2):
			stripped = raw_line.strip()
			if not stripped or stripped.startswith("#"):
				continue

			columns = stripped.split()
			if len(columns) < 3:
				raise ValueError(
					f"Expected at least 3 columns on line {line_number} of {source}, "
					f"found {len(columns)}"
				)

			try:
				cache_space = int(float(columns[0]))
				hitrate = float(columns[4])
			except ValueError as exc:  # pragma: no cover - defensive coding
				raise ValueError(
					f"Unable to parse cache space and hitrate on line {line_number} of {source}"
				) from exc

			points.append(FootprintPoint(cache_space=cache_space, hitrate=hitrate))

		return points

	@classmethod
	def from_file(cls, path: str | Path, *, encoding: str = "utf-8") -> "FootprintDescriptor":
		"""Create a descriptor by parsing an FDS file.

		Parameters
		----------
		path:
			File system path to the descriptor file.
		encoding:
			Text encoding to use when reading the file. Defaults to UTF-8.

		Returns
		-------
		FootprintDescriptor
			A populated descriptor instance.
		"""

		file_path = Path(path)
		if not file_path.exists():
			raise FileNotFoundError(file_path)

		with file_path.open(encoding=encoding) as handle:
			points = cls._parse_lines(handle, source=str(file_path))

		return cls(points)

	@classmethod
	def from_text(cls, text: str) -> "FootprintDescriptor":
		"""Create a descriptor by parsing raw FDS text.

		The input should follow the same line-oriented format as :meth:`from_file`.
		The first line is treated as a header or metadata line and ignored.
		"""

		points = cls._parse_lines(text.splitlines(), source="<text>")
		return cls(points)

	def hitrate_for_cache(self, cache_space: int) -> float:
		"""Return the hit-rate corresponding to ``cache_space``.

		If an exact cache entry is unavailable, the hit-rate of the nearest cache
		space is returned. For equidistant cache spaces, the smaller cache space is
		preferred.
		"""

		point = self._points_by_cache.get(cache_space)
		if point is None:
			point = self.nearest_point_for_cache(cache_space)
		return point.hitrate

	def nearest_point_for_cache(self, cache_space: int) -> FootprintPoint:
		"""Return the :class:`FootprintPoint` closest to ``cache_space``."""

		if not self._points_sorted_by_cache:
			raise ValueError("Descriptor contains no cache space data")

		from bisect import bisect_left

		index = bisect_left(self._cache_spaces, cache_space)

		if index <= 0:
			return self._points_sorted_by_cache[0]
		if index >= len(self._cache_spaces):
			return self._points_sorted_by_cache[-1]

		next_point = self._points_sorted_by_cache[index]
		prev_point = self._points_sorted_by_cache[index - 1]

		next_delta = abs(next_point.cache_space - cache_space)
		prev_delta = abs(cache_space - prev_point.cache_space)

		if prev_delta < next_delta:
			return prev_point
		if next_delta < prev_delta:
			return next_point

		# Tie-breaker: prefer the smaller cache space.
		return prev_point

	def nearest_cache_for_hitrate(self, hitrate: float) -> int:
		"""Return the cache space with the nearest available hit-rate value.

		The method searches for the recorded hit-rate with the smallest absolute
		difference from the requested ``hitrate`` value. In the event of a tie,
		the cache space associated with the lower hit-rate is returned; if both
		hit-rates are identical, the smaller cache space wins.
		"""

		point = self.nearest_point_for_hitrate(hitrate)
		return point.cache_space

	def nearest_point_for_hitrate(self, hitrate: float) -> FootprintPoint:
		"""Return the :class:`FootprintPoint` closest to ``hitrate``."""

		if not self._points_sorted_by_hitrate:
			# The constructor forbids this state, but the guard keeps type-checkers happy.
			raise ValueError("Descriptor contains no hit-rate data")

		# Binary search across the monotonic hit-rate list.
		from bisect import bisect_left

		index = bisect_left(self._hitrates, hitrate)

		if index <= 0:
			return self._points_sorted_by_hitrate[0]
		if index >= len(self._hitrates):
			return self._points_sorted_by_hitrate[-1]

		next_point = self._points_sorted_by_hitrate[index]
		prev_point = self._points_sorted_by_hitrate[index - 1]

		next_delta = abs(next_point.hitrate - hitrate)
		prev_delta = abs(hitrate - prev_point.hitrate)

		if prev_delta < next_delta:
			return prev_point
		if next_delta < prev_delta:
			return next_point

		# Tie-breaker: prefer the point with the lower hit-rate (prev_point),
		# and fall back to the smaller cache space if hit-rates are identical.
		if prev_point.hitrate == next_point.hitrate:
			return prev_point if prev_point.cache_space <= next_point.cache_space else next_point

		return prev_point

	def __len__(self) -> int:  # pragma: no cover - easily inferred but handy in REPL
		return len(self._points_by_cache)

	def __repr__(self) -> str:  # pragma: no cover - representational helper
		cls_name = self.__class__.__name__
		return f"{cls_name}(points={len(self)})"

	def find_max_possible_hitrate(self) -> FootprintPoint:
		"""Return the :class:`FootprintPoint` with the maximum hit-rate."""

		if not self._points_sorted_by_hitrate:
			raise ValueError("Descriptor contains no hit-rate data")

		return self._points_sorted_by_hitrate[-1]


	def plot(self, *, ax: "plt.Axes | None" = None, show: bool = False) -> "plt.Axes":
		"""Plot hit-rate (y-axis) against cache space (x-axis).

		Parameters
		----------
		ax:
			Optional Matplotlib axes to render onto. If omitted, a new figure
			and axes are created.
		show:
			If ``True``, explicitly call ``plt.show()`` after creating the plot.

		Returns
		-------
		matplotlib.axes.Axes
			The axes containing the plot.

		Raises
		------
		RuntimeError
			If Matplotlib is not installed.
		"""
		try:
			import matplotlib.pyplot as plt  # type: ignore
		except ImportError as exc:  # pragma: no cover - optional dependency
			raise RuntimeError("Matplotlib is required for plotting footprint descriptors") from exc

		if ax is None:
			_, ax = plt.subplots()

		cache_spaces = self._cache_spaces
		hitrates = [point.hitrate for point in self._points_sorted_by_cache]

		ax.plot(cache_spaces, hitrates, marker="o", linestyle="-", label="Hit-rate")
		ax.set_xlabel("Cache space")
		ax.set_ylabel("Hit-rate (%)")
		ax.set_title("Footprint Descriptor")
		ax.grid(True, linestyle="--", alpha=0.3)

		if show:
			plt.show()

		return ax

	def smooth_by_cache_bucket(self, bucket_size_mb: int = 10 * 1024) -> "FootprintDescriptor":
		"""Return a rebucketed/smoothed descriptor on fixed cache-size buckets.

		The cache space in this repository is stored in MB. By default this method
		rebuckets the descriptor into 10 GB buckets (10 * 1024 MB).

		Rules
		-----
		1. If multiple original points fall into the same bucket, use the lowest
		   hitrate observed in that bucket.
		2. If consecutive populated buckets are separated by one or more missing
		   buckets, fill the missing buckets by linearly interpolating the hitrate
		   between the two endpoints.
		3. If consecutive bucketed points end up with the same hitrate, treat that
		   as a plateau and spread the increase from the immediately previous
		   hitrate evenly across the plateau buckets.

		Parameters
		----------
		bucket_size_mb:
			Bucket size in MB. Defaults to 10 GB expressed in MB.
		"""

		if bucket_size_mb <= 0:
			raise ValueError("bucket_size_mb must be positive")

		bucket_to_hitrate: Dict[int, float] = {}
		for point in self._points_sorted_by_cache:
			bucket_cache_space = (point.cache_space // bucket_size_mb) * bucket_size_mb
			existing = bucket_to_hitrate.get(bucket_cache_space)
			if existing is None or point.hitrate < existing:
				bucket_to_hitrate[bucket_cache_space] = point.hitrate

		if not bucket_to_hitrate:
			raise ValueError("Descriptor contains no cache space data")

		sorted_bucket_points = sorted(bucket_to_hitrate.items(), key=lambda item: item[0])
		smoothed_points: List[FootprintPoint] = []

		for index, (cache_space, hitrate) in enumerate(sorted_bucket_points):
			smoothed_points.append(FootprintPoint(cache_space=cache_space, hitrate=hitrate))

			if index == len(sorted_bucket_points) - 1:
				continue

			next_cache_space, next_hitrate = sorted_bucket_points[index + 1]
			gap_buckets = ((next_cache_space - cache_space) // bucket_size_mb) - 1
			if gap_buckets <= 0:
				continue

			for gap_index in range(1, gap_buckets + 1):
				fraction = gap_index / (gap_buckets + 1)
				interpolated_hitrate = hitrate + (next_hitrate - hitrate) * fraction
				interpolated_cache_space = cache_space + gap_index * bucket_size_mb
				smoothed_points.append(
					FootprintPoint(
						cache_space=interpolated_cache_space,
						hitrate=interpolated_hitrate,
					)
				)

		smoothed_points.sort(key=lambda point: point.cache_space)

		plateau_smoothed_points: List[FootprintPoint] = list(smoothed_points)
		index = 0
		while index < len(plateau_smoothed_points):
			run_end = index
			current_hitrate = plateau_smoothed_points[index].hitrate

			while (
				run_end + 1 < len(plateau_smoothed_points)
				and plateau_smoothed_points[run_end + 1].hitrate == current_hitrate
			):
				run_end += 1

			if run_end > index and index > 0:
				previous_hitrate = plateau_smoothed_points[index - 1].hitrate
				delta = current_hitrate - previous_hitrate
				run_length = run_end - index + 1

				for offset in range(run_length):
					point = plateau_smoothed_points[index + offset]
					adjusted_hitrate = previous_hitrate + delta * ((offset + 1) / run_length)
					plateau_smoothed_points[index + offset] = FootprintPoint(
						cache_space=point.cache_space,
						hitrate=adjusted_hitrate,
					)

			index = run_end + 1

		return FootprintDescriptor(plateau_smoothed_points)