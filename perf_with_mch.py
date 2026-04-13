"""Metro-specific performance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from fds import FootprintDescriptor
from dataclasses import dataclass, field
from probability import ProbabilityDensityFunction, PdfBucket, Convolution


class MetroPerformanceWithMCH:
	"""Encapsulates latency PDFs and footprint descriptor for a metro."""

	name: str
	descriptor: FootprintDescriptor
	edge_rtt: Dict[str, ProbabilityDensityFunction] = field(default_factory=dict)

	def __init__(self, name: str, parent_model: MCHPerformanceModel, descriptor: FootprintDescriptor) -> None:
		self.name = name
		self.descriptor = descriptor
		self.edge_rtt = {}
		self.mch_rtt = None
		self.mch = parent_model

	def set_edge_rtt(self, from_metro: str, pdf: ProbabilityDensityFunction):
		self.edge_rtt[from_metro] = pdf

	def set_edge_tat_hit(self, pdf: ProbabilityDensityFunction):
		self.edge_tat_hit = pdf

	def set_edge_tat_miss(self, pdf: ProbabilityDensityFunction):
		self.edge_tat_miss = pdf

	def set_mch_rtt(self, pdf: ProbabilityDensityFunction):
		self.mch_rtt = pdf

	def get_ttfb_pdf(self, from_metro: str, hitrate: float, conn: Convolution) -> ProbabilityDensityFunction:

		edge_rtt_pdf = self.edge_rtt[from_metro]
		edge_tat_pdf = self.edge_tat_hit
		edge_tat_pdf_modified = edge_tat_pdf.with_fraction_at(0, 1 - hitrate/100)
		edge_pdf = conn.convolve(edge_rtt_pdf, edge_tat_pdf_modified)

		mch_rtt_pdf = self.mch_rtt
		mch_rtt_pdf_modified = mch_rtt_pdf.with_fraction_at(0, hitrate/100)
		edge_pdf = conn.convolve(edge_pdf, mch_rtt_pdf_modified)
		contribution = (hitrate)

		ttfb = self.mch.compute_ttfb_pdf(edge_pdf, conn, contribution)

		return ttfb

class MCHPerformanceModel:
	"""Encapsulates latency PDFs and footprint descriptor for a metro."""

	name: str
	mch_rtt: ProbabilityDensityFunction

	def __init__(self, name: str) -> None:
		self.name = name
		self.mch_tat = None

	def set_mch_tat(self, pdf: ProbabilityDensityFunction):
		self.mch_tat = pdf

	def get_mch_tat(self) -> ProbabilityDensityFunction:
		return self.mch_tat

	def compute_ttfb_pdf(self, pdf: ProbabilityDensityFunction, conn: Convolution, contribution: float) -> ProbabilityDensityFunction:
		mch_tat_modified = self.mch_tat.with_fraction_at(0, contribution/100)
		return conn.convolve(pdf, mch_tat_modified)

def _create_dummy_pdf() -> ProbabilityDensityFunction:
	pdf = ProbabilityDensityFunction([
		PdfBucket(lower_ms=0, upper_ms=1, count=1)
	])
	try:
		ProbabilityDensityFunction._require_pandas()
	except RuntimeError:
		return pdf
	return pdf.normalised_clone()

