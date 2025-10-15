from __future__ import annotations
import re
from typing import Optional
from ..data_types import TokenPolicy, Normalizer, TokenLike

def build_normalizer(policy: TokenPolicy) -> Normalizer:
    """ポリシーに基づくトークン正規化関数を構築する。

    Args:
        policy (TokenPolicy): 品詞・固有表現・表層保持などの条件を含む設定。

    Returns:
        Normalizer: spaCy トークンを受け取り正規化語を返す関数。
    """
    pat = re.compile(policy.alpha_regex).match

    def normalize(tok: TokenLike) -> Optional[str]:
        text = tok.text
        if not text or not pat(text):
            return None
        if tok.ent_type_ in policy.exclude_ner:
            return None
        pos = tok.pos_
        if policy.exclude_propn and pos == "PROPN":
            return None
        if policy.exclude_aux and pos == "AUX":
            return None
        if pos not in policy.target_pos:
            return None
        surface = text.lower()
        if surface in policy.keep_surface_for:
            return surface
        lemma = tok.lemma_.lower().strip()
        if lemma in ("-pron-", ""):
            return surface
        return lemma

    return normalize