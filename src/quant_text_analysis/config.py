"""Factory helpers for default configuration values."""

from __future__ import annotations

from typing import Tuple

from .data_types import Columns, TokenPolicy

CODE_MAP_CLUSTER: dict[str, tuple[str, ...]] = {
    # クラスタ1【抑圧】
    "【抑圧】": (
        "avoid",
        "space",
        "suppression",
        "conscious",
        "mask",
        "community",
        "suppress",
        "masking",
        "autistic_community",
        "neurodivergent",
        "conceal",
        "society",
        "identity",
        "discrimination",
        "blend",
        "world",
        "socially",
        "privilege",
        "majority",
    ),
    # クラスタ2【代替戦略】
    "【代替戦略】": (
        "stigma",
        "navigate",
        "complexity",
        "autistic_identity",
        "strategy",
        "coping_strategy",
        "conflict",
    ),
    # クラスタ4【手段】
    "【手段】": (
        "manage",
        "maintain",
        "social_relationship",
        "achieve",
        "opportunity",
    ),
    # クラスタ11【社会的欲求】
    "【社会的欲求】": (
        "social",
        "desire",
        "day_to_day",
        "engagement",
        "interpersonal",
        "everyday",
        "authentic",
        "connection",
        "survive",
        "social_interaction",
    ),
    # クラスタ18【ギャップ】
    "【ギャップ】": (
        "compensatory",
        "cognitive",
        "deficit",
        "underlie",
        "behavioral",
        "mechanism",
        "compensation",
        "process",
        "phenotype",
        "improve",
        "observe",
    ),
    # クラスタ19【子供の性差】
    "【子供の性差】": (
        "boy",
        "girl",
        "bias",
        "tend",
        "child",
        "detect",
        "peer",
        "enable",
        "school",
        "ability",
    ),
    # クラスタ10【疲弊】
    "【疲弊】": (
        "burnout",
        "risk_factor",
        "exhaustion",
        "depression",
        "minority_stress",
        "stress",
        "minority",
    ),
    # クラスタ5【女性の見逃し】
    "【女性の見逃し】": (
        "skill",
        "recognition",
        "delay",
        "gender",
        "male",
        "female",
    ),
    # クラスタ21【診断時期】
    "【診断時期】": (
        "diagnostic",
        "diagnosis",
        "age",
        "later",
        "late",
    ),
    # クラスタ22【正式診断】
    "【正式診断】": (
        "identification",
        "formal",
        "presence",
        "woman",
        "man",
        "attention",
        "symptom_severity",
    ),
}

CODE_MAP_GENDER: dict[str, tuple[str, ...]] = {
    # 女性
    "【女性】": (
        "female",
        "woman",
        "girl",
    ),
    # 男性
    "【男性】": (
        "male",
        "man",
        "boy",
    ),
    "【性別】": (
        "gender",
        "sex",
        "sex_at_birth",
        "sex_assigned_at_birth",
        "sex_designated_at_birth",
    ),
}

MNLR_EXCLUDE_METHODS: tuple[str, ...] = ("other",)


def default_columns() -> Columns:
    """既定の列名設定を返す。

    Returns:
        Columns: 既定の列名設定を表す `Columns` インスタンス。
    """
    return Columns()



def default_token_policy() -> TokenPolicy:
    """既定のトークン正規化ポリシーを返す。

    Returns:
        TokenPolicy: 既定のトークン正規化ポリシーを表す `TokenPolicy` インスタンス。
    """
    forced: Tuple[Tuple[str, ...], ...] = (
        ("autistic", "trait"),
        ("mental", "health"),
        ("autism", "spectrum", "disorder"),
        ("autism", "spectrum", "condition"),
        ("camouflage", "behavior"),
        ("camouflaging", "behavior"),
        ("self", "report"),
        ("autistic", "identity"),
        ("social", "anxiety"),
        ("well", "being"),
        ("impression", "management"),
        ("autistic", "characteristic"),
        ("quality", "of", "life"),
        ("generalized", "anxiety"),
        ("perceived", "stress"),
        ("internal", "consistency"),
        ("compensatory", "strategy"),
        ("social", "relationship"),
        ("typically", "develop"),
        ("typically", "developing"),
        ("self", "perceived"),
        ("thematic", "analysis"),
        ("general", "population"),
        ("first", "impression"),
        ("social", "context"),
        ("social", "skill"),
        ("social", "interaction"),
        ("social", "communication"),
        ("autism", "trait"),
        ("construct", "validity"),
        ("autistic", "community"),
        ("social", "strategy"),
        ("clinical", "practice"),
        ("systematic", "review"),
        ("symptom", "severity"),
        ("risk", "factor"),
        ("fear", "of", "negative", "evaluation"),
        ("intellectual", "disability"),
        ("social", "environment"),
        ("measurement", "invariance"),
        ("minority", "stress"),
        ("social", "motivation"),
        ("convergent", "validity"),
        ("coping", "strategy"),
        ("interaction", "effect"),
        ("social", "challenge"),
        ("eating", "disorder"),
        ("theory", "of", "mind"),
        ("open", "end"),
        ("lived", "experience"),
        ("co", "occur"),
        ("co", "occurring"),
        ("gender", "diverse"),
        ("posttraumatic", "stress"),
        ("social", "norm"),
        ("mediation", "analysis"),
        ("mix", "method"),
        ("neurodivergent", "trait"),
        ("executive", "function"),
        ("general", "education"),
        ("sensory", "processing"),
        ("sensory", "process"),
        ("interpersonal", "trauma"),
        ("internalized", "stigma"),
        ("cognitive", "flexibility"),
        ("compensatory", "behavior"),
        ("social", "camouflage"),
        ("social", "camouflaging"),
        ("ASD",),
        ("retest",),
        ("sc",),
        ("ADHD",),
        ("daily", "life"),
        ("sensory", "sensitivity"),
        ("eating", "disorder"),
        ("generalized", "anxiety", "disorder"),
        ("neurodevelopmental", "disorder"),
        ("gender", "identity"),
        ("sex", "at", "birth"),
        ("sex", "assigned", "at", "birth"),
        ("sex", "designated", "at", "birth"),
        ("semi", "structured"),
        ("non", "binary"),
        ("day", "to", "day"),
        ("cross", "sectional"),
        ("co", "produce"),
        ("co", "production"),
        ("co", "twin"),
        ("self", "esteem"),
    )

    forced_aliases: Tuple[Tuple[Tuple[str, ...], str], ...] = (
        (("self", "reported"), "self_report"),
        (("well", "be"), "well_being"),
        (("generalize", "anxiety"), "generalized_anxiety"),
        (("perceive", "stress"), "perceived_stress"),#?
        (("self", "perceive"), "self_perceived"),
        (("cope", "strategy"), "coping_strategy"),
        (("live", "experience"), "lived_experience"),
        (("perceive", "stigma"), "perceived_stigma"),#?
        (("mediation", "analyzes"), "mediation_analysis"),
        (("internalize", "stigma"), "internalized_stigma"),#?
        (("generalize", "anxiety", "disorder"), "generalized_anxiety_disorder"),
        (("eat", "disorder"), "eating_disorder"),
        (("sex", "assign", "at", "birth"), "sex_assigned_at_birth"),
        (("sex", "design", "at", "birth"), "sex_designated_at_birth"),
        (("semi", "structure"), "semi_structured"),
    )

    return TokenPolicy(
        target_pos=frozenset({"NOUN", "VERB", "ADJ", "ADV"}),
        exclude_ner=frozenset({"PERSON", "GPE", "LOC"}),
        exclude_propn=True,
        exclude_aux=True,
        keep_surface_for=frozenset(),
        forced_phrases=forced,
        forced_joiner="_",
        forced_aliases=forced_aliases,
    )
