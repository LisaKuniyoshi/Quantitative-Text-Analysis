"""Utilities supporting MNLR-related commands."""

from __future__ import annotations

from .coding import codes_per_doc, invert_code_map, rows_from_tokens
from .model import (
    Design,
    build_design,
    fit_multinomial_cluster_robust,
    predict_probabilities,
)
from .plotting import plot_prob_by_year_with_method
from .tables import build_code_method_crosstab

__all__ = [
    "Design",
    "build_code_method_crosstab",
    "build_design",
    "codes_per_doc",
    "fit_multinomial_cluster_robust",
    "invert_code_map",
    "plot_prob_by_year_with_method",
    "predict_probabilities",
    "rows_from_tokens",
]
