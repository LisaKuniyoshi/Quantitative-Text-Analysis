# nlp_backend.py（抜粋）
from typing import Iterable, Iterator
import spacy
from .data_types import NLPBackend, DocLike, TokenLike

class _SpacyTokenAdapter:
    __slots__ = ("_t",)
    def __init__(self, token: "spacy.tokens.Token") -> None:
        self._t = token
    @property
    def text(self) -> str:      return self._t.text
    @property
    def ent_type_(self) -> str: return self._t.ent_type_
    @property
    def pos_(self) -> str:      return self._t.pos_
    @property
    def lemma_(self) -> str:    return self._t.lemma_

class _SpacyDocAdapter:
    __slots__ = ("_d",)
    def __init__(self, doc: "spacy.tokens.Doc") -> None:
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
