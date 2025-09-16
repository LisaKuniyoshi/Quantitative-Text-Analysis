from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, FrozenSet, Iterable, Iterator, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

# 列名（注入可能に）
@dataclass(frozen=True)
class Columns:
    abstract: str = "Abstract Note"
    year: str = "Publication Year"
    manual_tags: str = "Manual Tags"

# トークン正規化ポリシー（不変）
@dataclass(frozen=True)
class TokenPolicy:
    target_pos: FrozenSet[str]
    exclude_ner: FrozenSet[str]
    exclude_propn: bool
    exclude_aux: bool
    keep_surface_for: FrozenSet[str]
    alpha_regex: str = r"^[A-Za-z]+$"
    forced_phrases: Tuple[Tuple[str, ...], ...] = ()
    forced_joiner: str = "_"

# ランキングの表示パラメタ
@dataclass(frozen=True)
class RankingParams:
    top_n: int = 50
    min_docs: int = 1

# 1文書の結果（不変）
@dataclass(frozen=True)
class DocResult:
    tokens: Tuple[str, ...]
    total: int

# ---- Protocols（外部実装の抽象化）----
@runtime_checkable
class TokenLike(Protocol):
    text: str
    ent_type_: str
    pos_: str
    lemma_: str

@runtime_checkable
class DocLike(Protocol):
    def __iter__(self) -> Iterator[TokenLike]:
        ...

@runtime_checkable
class NLPBackend(Protocol):
    model_name: str
    def pipe(self, texts: Iterable[str]) -> Iterable[DocLike]:
        ...

# 正規化関数の型
Normalizer = Callable[[TokenLike], Optional[str]]

# キャッシュ用ユーティリティ
def to_plain(obj: object) -> Mapping[str, object]:
    return asdict(obj)  # dataclass 前提