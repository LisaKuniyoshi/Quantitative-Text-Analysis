from typing import Iterable, Iterator

import spacy
from spacy.tokens import Doc, Token

from ..data_types import DocLike, TokenLike

class _SpacyTokenAdapter:
    __slots__ = ("text", "ent_type_", "pos_", "lemma_")
    def __init__(self, token: "Token") -> None:
        self.text = token.text
        self.ent_type_ = token.ent_type_
        self.pos_ = token.pos_
        self.lemma_ = token.lemma_

class _SpacyDocAdapter:
    __slots__ = ("_d",)
    def __init__(self, doc: "Doc") -> None:
        self._d = doc
    def __iter__(self) -> Iterator[TokenLike]:
        for tok in self._d:
            yield _SpacyTokenAdapter(tok)

class SpacyBackend:
    """spaCy モデルを用いて文書解析を行うバックエンド。"""

    def __init__(self, model: str) -> None:
        """spaCy モデルを読み込む。

        Args:
            model (str): 読み込む spaCy モデル名。
        """
        self._nlp = spacy.load(model)
        self.model_name: str = model

    def pipe(self, texts: Iterable[str]) -> Iterator[DocLike]:
        """spaCy の逐次パイプラインで文書を解析する。

        Args:
            texts (Iterable[str]): 解析対象のテキスト列。

        Returns:
            Iterator[DocLike]: spaCy 互換のドキュメントイテレータ。
        """
        for doc in self._nlp.pipe(texts):
            yield _SpacyDocAdapter(doc)
