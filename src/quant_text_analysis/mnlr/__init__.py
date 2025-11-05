"""Utilities supporting MNLR-related commands."""

from __future__ import annotations

from .coding import codes_per_doc, invert_code_map, rows_from_tokens
from .model import (
    fit_mnlogit,
    predict_probabilities,
)
from .plotting import plot_prob_by_year_with_method
from .tables import build_code_method_crosstab

__all__ = [
    "build_code_method_crosstab",
    "codes_per_doc",
    "invert_code_map",
    "fit_mnlogit",
    "plot_prob_by_year_with_method",
    "predict_probabilities",
    "rows_from_tokens",
]
