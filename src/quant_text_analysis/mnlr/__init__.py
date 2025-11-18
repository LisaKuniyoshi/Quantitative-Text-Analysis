"""Utilities supporting MNLR-related commands."""

from __future__ import annotations

from .coding import (
    build_observation_frame,
    codes_per_doc,
    invert_code_map,
    NO_CODE_LABEL,
    rows_from_tokens,
)
from .model import (
    fit_binary_logit,
    fit_binary_logit_for_code,
    fit_binary_logit_for_codes,
    fit_mnlogit,
    predict_probabilities,
)
from .plotting import (
    plot_method_odds_ratios,
    plot_prob_by_year_with_method,
    plot_year_odds_ratios,
)
from .tables import build_code_method_crosstab

__all__ = [
    "build_code_method_crosstab",
    "build_observation_frame",
    "codes_per_doc",
    "invert_code_map",
    "NO_CODE_LABEL",
    "fit_binary_logit",
    "fit_binary_logit_for_code",
    "fit_binary_logit_for_codes",
    "fit_mnlogit",
    "plot_method_odds_ratios",
    "plot_prob_by_year_with_method",
    "plot_year_odds_ratios",
    "predict_probabilities",
    "rows_from_tokens",
]
