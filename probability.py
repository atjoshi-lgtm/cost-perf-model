"""Probability distribution helpers for latency bucket data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import erf, sqrt
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
	import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	np = None  # type: ignore

try:  # pragma: no cover - optional dependency
	import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
	import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
	plt = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - hints only
	from pandas import DataFrame, Series  # type: ignore
else:  # pragma: no cover - used for typing fallback when pandas missing
	DataFrame = Any
	Series = Any


BUCKET_PATTERN = re.compile(
	r"(?:[A-Za-z0-9]+_)?(?P<lower>\d+)_ (?P<upper>\d+)_ms".replace(" ", "")
)


@dataclass(frozen=True)
class PdfBucket:
	"""Represents a latency bucket covering ``[lower_ms, upper_ms)``."""

	lower_ms: int
	upper_ms: int
	count: float

	@property
	def width_ms(self) -> int:
		return max(0, self.upper_ms - self.lower_ms)

	def to_millisecond_counts(self) -> Iterator[Tuple[int, float]]:
		"""Evenly distribute the count across each millisecond in the bucket."""

		width = self.width_ms
		if width <= 0 or self.count <= 0:
			return iter(())

		per_ms = self.count / width
		return ((ms, per_ms) for ms in range(self.lower_ms, self.upper_ms))


class ProbabilityDensityFunction:
	"""Stores and manipulates a probability distribution function."""

	def __init__(self, buckets: Iterable[PdfBucket]) -> None:
		self._buckets: List[PdfBucket] = sorted(
			(bucket for bucket in buckets if bucket.count > 0 and bucket.width_ms > 0),
			key=lambda bucket: bucket.lower_ms,
		)
		self._probability_series: Series | None = None

	def __iter__(self) -> Iterator[PdfBucket]:
		return iter(self._buckets)

	def total_count(self) -> float:
		return sum(bucket.count for bucket in self._buckets)

	def millisecond_series(self) -> Series:
		"""Return a pandas Series indexed by millisecond with distributed counts."""

		self._require_pandas()
		if self._probability_series is not None:
			return self._probability_series

		tally: Dict[int, float] = {}
		for bucket in self._buckets:
			for millisecond, count in bucket.to_millisecond_counts():
				tally[millisecond] = tally.get(millisecond, 0.0) + count

		if not tally:
			self._probability_series = pd.Series(dtype=float)
			return self._probability_series

		series = pd.Series(tally, dtype=float).sort_index()
		self._probability_series = series
		return series

	@property
	def probability_series(self) -> Series:
		"""Cached probability series built at millisecond resolution."""

		return self.millisecond_series()

	def zero_millisecond(self, millisecond: int) -> "ProbabilityDensityFunction":
		"""Return a new PDF with ``millisecond`` set to zero and renormalized."""

		self._require_pandas()
		series = self.probability_series.copy().astype(float)
		if series.empty:
			return ProbabilityDensityFunction.from_millisecond_series(series)

		if millisecond in series.index:
			series.at[millisecond] = 0.0
		else:
			# Ensure the millisecond is represented explicitly as zero so that
			# renormalisation still occurs in a predictable way.
			series.loc[millisecond] = 0.0

		series = series.clip(lower=0).sort_index()
		total = series.sum()
		if total <= 0:
			return ProbabilityDensityFunction.from_millisecond_series(pd.Series(dtype=float))

		normalised = (series / total)
		normalised = normalised[normalised > 0]
		return ProbabilityDensityFunction.from_millisecond_series(normalised)

	def with_fraction_at(self, millisecond: int, fraction: float) -> "ProbabilityDensityFunction":
		"""Scale existing mass to ``1 - fraction`` and add ``fraction`` at ``millisecond``.

		This returns a new PDF that preserves the original shape while allocating
		``fraction`` of the total probability mass to the specified millisecond.
		"""

		self._require_pandas()
		if not 0.0 <= fraction <= 1.0:
			raise ValueError("fraction must be between 0 and 1 inclusive")

		millisecond = int(millisecond)
		series = self.probability_series.astype(float)
		total = float(series.sum())

		if total <= 0.0:
			if fraction <= 0.0:
				return ProbabilityDensityFunction([])
			return ProbabilityDensityFunction.from_millisecond_series(
				pd.Series({millisecond: 1.0}, dtype=float)
			)

		if fraction >= 1.0:
			scaled = pd.Series(dtype=float)
		else:
			scale = (1.0 - fraction) / total
			scaled = (series * scale).astype(float)

		updated_value = scaled.get(millisecond, 0.0) + fraction
		scaled.loc[millisecond] = updated_value
		scaled = scaled.clip(lower=0).sort_index()

		return ProbabilityDensityFunction.from_millisecond_series(scaled)

	def millisecond_at_percentile(self, percentile: float) -> int:
		"""Return the smallest millisecond whose cumulative probability meets ``percentile``.

		The ``percentile`` should be a value between 0 and 100 inclusive. The
		return value is the millisecond index at which the cumulative distribution
		first reaches or exceeds the requested percentile. When the PDF is empty or
		has zero total mass, a ``ValueError`` is raised.
		"""

		self._require_pandas()
		if not 0.0 <= percentile <= 100.0:
			raise ValueError("percentile must be between 0 and 100 inclusive")

		series = self.probability_series.astype(float)
		if series.empty:
			raise ValueError("Cannot compute percentile for an empty PDF")

		total = float(series.sum())
		if total <= 0.0:
			raise ValueError("Cannot compute percentile for a PDF with zero total mass")

		cdf = (series.cumsum() / total).astype(float)
		fraction = percentile / 100.0

		if fraction <= 0.0:
			return int(cdf.index.min())
		if fraction >= 1.0:
			return int(cdf.index.max())

		meeting = cdf[cdf >= fraction]
		if not meeting.empty:
			return int(meeting.index[0])

		return int(cdf.index.max())

	def to_microsecond_pdf(self, step_us: int = 10) -> "ProbabilityDensityFunction":
		"""Distribute mass uniformly within each millisecond and return a microsecond PDF."""
		self._require_pandas()

		series = self.probability_series.astype(float)
		if series.empty:
			return ProbabilityDensityFunction([])

		micro_values: Dict[int, float] = {}
		steps_per_ms = 1_000 // step_us
		for millisecond, probability in series.items():
			if probability <= 0.0:
				continue
			base_microsecond = int(millisecond) * 1_000
			share = probability / steps_per_ms
			for i in range(steps_per_ms):
				offset = i * step_us
				micro_values[base_microsecond + offset] = share

		if not micro_values:
			return ProbabilityDensityFunction([])

		micro_series = pd.Series(micro_values, dtype=float).sort_index()
		total = float(micro_series.sum())
		if total <= 0.0:
			return ProbabilityDensityFunction([])

		normalised = micro_series / total
		return ProbabilityDensityFunction.from_millisecond_series(normalised)


	def plot(
		self,
		*,
		ax: Any | None = None,
		title: str | None = None,
		xlabel: str = "Latency (ms)",
		ylabel: str = "Probability",
		color: str = "C0",
		linewidth: float = 1.5,
		marker: str | None = None,
	) -> Any:
		"""Plot the PDF using matplotlib and return the axes."""

		self._require_pandas()
		self._require_matplotlib()

		series = self.probability_series
		if series.empty:
			series = pd.Series({0: 0.0}, dtype=float)

		axes = ax
		if axes is None:
			_, axes = plt.subplots()  # type: ignore[assignment]
		# Can you plot CDF instead of PDF? --- IGNORE ---
		cdf = series.cumsum() / series.sum()
		axes.plot(cdf.index, cdf.values, color=color, linewidth=linewidth, marker=marker)
		axes.set_xlabel(xlabel)
		axes.set_ylabel(ylabel)
		if title:
			axes.set_title(title)
		axes.grid(True, which="both", linestyle="--", alpha=0.3)
		#plt.show()
		#plt.clf()
		#return axes

	@classmethod
	def from_dataframe(cls, frame: DataFrame) -> "ProbabilityDensityFunction":
		"""Create a PDF from latency bucket columns in ``frame``.

		Columns matching ``x_y_ms`` or ``<prefix>_x_y_ms`` are interpreted as the
		count of requests whose latency was in ``[x, y)`` milliseconds. Counts are
		summed across rows when multiple samples are provided.
		"""

		buckets: List[PdfBucket] = []

		for column in frame.columns:
			match = BUCKET_PATTERN.fullmatch(column)
			if not match:
				continue

			lower_ms = int(match.group("lower"))
			upper_ms = int(match.group("upper"))
			count = float(frame[column].sum(skipna=True))
			if count > 0 and upper_ms > lower_ms:
				buckets.append(PdfBucket(lower_ms=lower_ms, upper_ms=upper_ms, count=count))

		pdf = cls(buckets)
		pdf.millisecond_series()  # warm cache so callers can reuse directly

		# Doesnt look like the pdf is normalized
		total = float(pdf.probability_series.sum())
		if total > 0:
			pdf._probability_series /= total

		return pdf

	@staticmethod
	def _require_pandas() -> None:
		if pd is None:  # pragma: no cover - runtime guard
			raise RuntimeError(
				"pandas is required for ProbabilityDensityFunction operations but is not installed."
			)

	@staticmethod
	def _require_numpy() -> None:
		if np is None:  # pragma: no cover - runtime guard
			raise RuntimeError(
				"numpy is required for FFT operations but is not installed."
			)

	@staticmethod
	def _require_matplotlib() -> None:
		if plt is None:  # pragma: no cover - runtime guard
			raise RuntimeError(
				"matplotlib is required for plotting but is not installed."
			)

	@classmethod
	def from_millisecond_series(cls, series: Series) -> "ProbabilityDensityFunction":
		"""Create a PDF from a millisecond-indexed series."""


		buckets = []
		if series is not None:
			series = series.dropna().astype(float)
			for millisecond, count in series.items():
				count_value = float(count)
				if count_value <= 0:
					continue
				ms = int(millisecond)
				buckets.append(PdfBucket(lower_ms=ms, upper_ms=ms + 1, count=count_value))

		pdf = cls(buckets)
		pdf._probability_series = series.sort_index()
		return pdf

	def to_dict(self) -> Mapping[int, float]:
		"""Return the PDF as a read-only mapping from millisecond to probability."""

		series = self.probability_series
		return dict(series.items())

	def normalised_clone(self) -> "ProbabilityDensityFunction":
		"""Return a normalised copy of this PDF."""

		series = self.probability_series
		if series.empty:
			return ProbabilityDensityFunction.from_millisecond_series(series)

		total = float(series.sum())
		if total <= 0:
			return ProbabilityDensityFunction([])

		normalised = series / total
		return ProbabilityDensityFunction.from_millisecond_series(normalised)


class Convolution:
	"""Convolve two probability density functions using FFT."""

	def __init__(self) -> None:
		ProbabilityDensityFunction._require_numpy()

	def convolve(self, lhs: ProbabilityDensityFunction, rhs: ProbabilityDensityFunction) -> ProbabilityDensityFunction:
		"""Return the convolution of the two PDFs as a new PDF."""

		lhs_series = lhs.probability_series.astype(float)
		rhs_series = rhs.probability_series.astype(float)

		if lhs_series.empty or rhs_series.empty:
			return ProbabilityDensityFunction.from_millisecond_series(pd.Series(dtype=float))

		lhs_min = int(lhs_series.index.min())
		lhs_max = int(lhs_series.index.max())
		rhs_min = int(rhs_series.index.min())
		rhs_max = int(rhs_series.index.max())

		lhs_length = lhs_max - lhs_min + 1
		rhs_length = rhs_max - rhs_min + 1

		lhs_array = np.zeros(lhs_length, dtype=float)
		rhs_array = np.zeros(rhs_length, dtype=float)

		lhs_positions = (lhs_series.index.to_numpy(dtype=int) - lhs_min).astype(int)
		rhs_positions = (rhs_series.index.to_numpy(dtype=int) - rhs_min).astype(int)

		lhs_array[lhs_positions] = lhs_series.to_numpy(dtype=float)
		rhs_array[rhs_positions] = rhs_series.to_numpy(dtype=float)

		result_length = lhs_length + rhs_length - 1
		fft_size = 1 << (result_length - 1).bit_length()

		fft_lhs = np.fft.rfft(lhs_array, fft_size)
		fft_rhs = np.fft.rfft(rhs_array, fft_size)
		convolved = np.fft.irfft(fft_lhs * fft_rhs, fft_size)[:result_length]

		result_start = lhs_min + rhs_min
		result_index = np.arange(result_start, result_start + result_length, dtype=int)
		result_series = pd.Series(convolved, index=result_index, dtype=float)

		return ProbabilityDensityFunction.from_millisecond_series(result_series)

def gaussian_pdf(
	mean: float,
	stddev: float,
	lower_ms: int,
	upper_ms: int,
) -> ProbabilityDensityFunction:
	"""Create a PDF approximating a truncated Gaussian distribution.

	The returned PDF allocates probability mass to integer millisecond buckets in
	``[lower_ms, upper_ms)`` by integrating the Gaussian density across each
	interval. The mass is renormalised so the resulting PDF sums to one within the
	truncated bounds.
	"""

	import pandas as pd  # type: ignore

	if stddev <= 0:
		raise ValueError("stddev must be positive")
	if upper_ms <= lower_ms:
		raise ValueError("upper_ms must be greater than lower_ms")

	lower = int(lower_ms)
	upper = int(upper_ms)
	stddev = float(stddev)
	mean = float(mean)
	sqrt_two = sqrt(2.0)

	def _cdf(x: float) -> float:
		return 0.5 * (1.0 + erf((x - mean) / (stddev * sqrt_two)))

	masses: Dict[int, float] = {}
	for millisecond in range(lower, upper):
		mass = _cdf(millisecond + 1.0) - _cdf(millisecond)
		if mass <= 0.0:
			continue
		masses[millisecond] = mass

	if not masses:
		raise ValueError("Gaussian mass within bounds is numerically zero; adjust parameters")

	total_mass = sum(masses.values())
	if total_mass <= 0.0:
		raise ValueError("Gaussian mass within bounds is numerically zero; adjust parameters")

	normalised = {ms: mass / total_mass for ms, mass in masses.items()}
	series = pd.Series(normalised, dtype=float)
	return ProbabilityDensityFunction.from_millisecond_series(series)

def weighted_pdf_sum(
    pdfs: Iterable[ProbabilityDensityFunction],
    weights: Iterable[float],
) -> ProbabilityDensityFunction:
    """Combine multiple PDFs using the supplied weights.

    The resulting PDF is the weighted sum of the individual PDFs, renormalised so
    that the total probability mass is one. Weights must be non-negative and the
    iterator lengths must match.

    Raises
    ------
    ValueError
        If the number of PDFs and weights differ, the list is empty, a weight is
        negative, or the combined mass is numerically zero.
    RuntimeError
        If pandas is not installed.
    """
    ProbabilityDensityFunction._require_pandas()

    pdf_list = list(pdfs)
    weight_list = list(weights)

    if not pdf_list:
        raise ValueError("At least one PDF is required")
    if len(pdf_list) != len(weight_list):
        raise ValueError("Number of PDFs and weights must match")

    accumulator = pd.Series(dtype=float)
    total_weight = 0.0

    for pdf, weight in zip(pdf_list, weight_list, strict=True):
        if weight < 0:
            raise ValueError("Weights must be non-negative")
        if weight == 0:
            continue

        series = pdf.probability_series.astype(float)
        if series.empty:
            continue

        weighted_series = series * float(weight)
        accumulator = accumulator.add(weighted_series, fill_value=0.0)
        total_weight += float(weight)

    if total_weight <= 0 or accumulator.empty:
        raise ValueError("Combined PDF has zero total weight or mass")

    combined = accumulator / accumulator.sum()
    return ProbabilityDensityFunction.from_millisecond_series(combined)