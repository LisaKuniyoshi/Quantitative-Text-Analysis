"""Shared dataclasses and protocols describing public data structures."""

from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Callable,
    FrozenSet,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

# 列名（注入可能に）
@dataclass(frozen=True)
class Columns:
    """分析で参照する列名をまとめた設定用データクラス。

    Attributes:
        abstract (str): 要約テキスト列の名前。
        year (str): 発行年列の名前。
        manual_tags (str): 手動タグ列の名前。
    """
    abstract: str = "Abstract Note"
    year: str = "Publication Year"
    manual_tags: str = "Manual Tags"

# トークン正規化ポリシー（不変）
@dataclass(frozen=True)
class TokenPolicy:
    """トークン正規化の制約を表す設定用データクラス。

    Attributes:
        target_pos (frozenset[str]): 対象とする品詞集合。
        exclude_ner (frozenset[str]): 除外する固有表現タイプ集合。
        exclude_propn (bool): 固有名詞を除外するかどうか。
        exclude_aux (bool): 助動詞を除外するかどうか。
        keep_surface_for (frozenset[str]): 表層形を保持する品詞集合。
        alpha_regex (str): アルファベット判定に用いる正規表現。
        forced_phrases (tuple[tuple[str, ...], ...]): 強制的に多語表現とみなす語列。
    forced_joiner (str): 強制多語表現を結合する際の連結文字。
    forced_aliases (tuple[tuple[tuple[str, ...], str], ...]): 多語表現ごとの出力別名。
    """
    target_pos: FrozenSet[str]
    exclude_ner: FrozenSet[str]
    exclude_propn: bool
    exclude_aux: bool
    keep_surface_for: FrozenSet[str]
    alpha_regex: str = r"^[A-Za-z]+$"
    forced_phrases: Tuple[Tuple[str, ...], ...] = ()
    forced_joiner: str = "_"
    forced_aliases: Tuple[Tuple[Tuple[str, ...], str], ...] = ()

# ランキングの表示パラメタ
@dataclass(frozen=True)
class RankingParams:
    """ランキング出力時の閾値設定。

    Attributes:
        top_n (int): 表示する上位アイテム数。
        min_docs (int): 最小出現文書数。
    """
    top_n: int = 50
    min_docs: int = 1

# ---- Protocols（外部実装の抽象化）----
@runtime_checkable
class TokenLike(Protocol):
    """spaCy 互換トークンが満たすべきインターフェース。"""
    text: str
    ent_type_: str
    pos_: str
    lemma_: str

@runtime_checkable
class DocLike(Protocol):
    """spaCy 互換ドキュメントが満たすべきシーケンスインターフェース。"""
    def __iter__(self) -> Iterator[TokenLike]:
        ...

@runtime_checkable
class NLPBackend(Protocol):
    """spaCy 互換 NLP モデルのインターフェース定義。"""
    model_name: str
    def pipe(self, texts: Iterable[str]) -> Iterable[DocLike]:
        ...

# 正規化関数の型
Normalizer = Callable[[TokenLike], Optional[str]]
