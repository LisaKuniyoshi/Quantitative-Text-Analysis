from typing import Iterable, Iterator

import spacy
from spacy.tokens import Doc, Token

from .data_types import NLPBackend, DocLike, TokenLike

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
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self._nlp = spacy.load(model)
        self.model_name: str = model
    def pipe(self, texts: Iterable[str]) -> Iterator[DocLike]:
        for doc in self._nlp.pipe(texts):
            yield _SpacyDocAdapter(doc)
