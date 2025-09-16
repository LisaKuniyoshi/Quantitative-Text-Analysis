from __future__ import annotations
from .data_types import TokenPolicy, RankingParams, Columns

def default_columns() -> Columns:
    return Columns()

def default_token_policy() -> TokenPolicy:
    forced = (
        ("autism", "spectrum", "disorder"),
        ("mental", "health"),
    )

    return TokenPolicy(
        target_pos=frozenset({"NOUN", "VERB", "ADJ", "ADV"}),
        exclude_ner=frozenset({"PERSON", "GPE", "LOC"}),
        exclude_propn=True,
        exclude_aux=True,
        keep_surface_for=frozenset(),
        forced_phrases=forced,
        forced_joiner="_",
    )

def default_ranking_params() -> RankingParams:
    return RankingParams(top_n=50, min_docs=1)