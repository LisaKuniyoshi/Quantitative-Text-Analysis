"""Per-document token analysis with caching support."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import json
import os
import pickle

from breame.spelling import get_american_spelling

from ..data_types import NLPBackend, Normalizer, TokenPolicy, TokenLike

PerDocFreq = Dict[str, float]
PerDocTokens = List[str]

# 強制抽出の際に「語間で無視する記号」：ハイフン類・細かい句読点のみ
_SKIP_PUNCTS = {"-", "‐", "‒", "–", "—", "―", "·", "•", "/", "\\"}
# 句点やコロンなどは「語間の橋渡し」に使わない（= そこで強制一致は切れる）
_BREAK_PUNCTS = {".", "…", ":", ";", "?", "!", ",", "(", ")", "[", "]", "{", "}", "“", "”", "‘", "’", "'"}

# ----------------------------
# 強制フレーズ辞書
# ----------------------------
def _build_forced_index(policy: TokenPolicy) -> Dict[Tuple[str, ...], str]:
    """強制抽出辞書（キー＝lemma列の小文字タプル、値＝結合トークン）"""
    idx: Dict[Tuple[str, ...], str] = {}
    alias_map: Dict[Tuple[str, ...], str] = {}
    for key, alias in policy.forced_aliases:
        if not key or not alias:
            continue
        lowered_key = tuple(get_american_spelling(word.lower()) for word in key)
        alias_map[lowered_key] = alias.strip().lower()
    for phrase in policy.forced_phrases:
        if not phrase:
            continue
        key = tuple(get_american_spelling(w.lower()) for w in phrase)
        alias = alias_map.get(key)
        idx[key] = alias if alias is not None else policy.forced_joiner.join(key)
    for key, alias in alias_map.items():
        idx.setdefault(key, alias)
    return idx


def _is_skip_punct(tok: TokenLike) -> bool:
    """強制抽出評価時にスキップ可能な句読点かどうか。"""
    t = tok.text
    return (tok.pos_ == "SPACE") or (t in _SKIP_PUNCTS)


def _is_break_punct(tok: TokenLike) -> bool:
    """強制抽出のマッチングを打ち切る句読点かどうか。"""
    t = tok.text
    return t in _BREAK_PUNCTS

# ----------------------------
# 主処理：文書ごとの正規化トークンと文書内相対頻度
# ----------------------------

def analyze_docs(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: Sequence[str],
    policy: TokenPolicy,
) -> List[PerDocTokens]:
    """文書群を解析し正規化トークン列を求める。

    Args:
        backend (NLPBackend): トークン化・解析を行う NLP バックエンド。
        normalizer (Normalizer): トークン正規化関数。
        texts (Sequence[str]): 解析対象の文書本文。
        policy (TokenPolicy): 強制抽出やフィルタ条件を定義するポリシー。

    Returns:
        list[list[str]]: 文書ごとの正規化済みトークン列。
    """
    forced_index = _build_forced_index(policy)
    keys_by_len: List[Tuple[Tuple[str, ...], str]] = sorted(
        forced_index.items(), key=lambda kv: len(kv[0]), reverse=True
    )

    tokenized_docs: List[PerDocTokens] = []

    for doc in backend.pipe(texts):
        raw: List[TokenLike] = [tok for tok in doc]

        # (A) lemma 列（小文字）を先に作る：ここでは品詞フィルタ等を掛けない
        lemma_lower: List[str] = []
        for t in raw:
            # spaCy の lemma は空や "-PRON-" の場合があるので表層救済
            lem = (t.lemma_ or t.text).lower().strip()
            if lem in ("-pron-", ""):
                lem = t.text.lower()
            lem = get_american_spelling(lem)
            lemma_lower.append(lem)

        doc_tokens: List[str] = []
        i = 0
        n = len(raw)

        while i < n:
            # 句点・区切りは強制一致の境界にする
            if _is_break_punct(raw[i]):
                i += 1
                continue

            matched = False

            # (B) lemma 列に対する「最長一致」強制抽出（語間のハイフン類はスキップ可）
            if keys_by_len:
                for key, joined in keys_by_len:
                    j = i
                    k = 0
                    while j < n and k < len(key):
                        # BREAK 到達で中断
                        if _is_break_punct(raw[j]):
                            break
                        # 語間のハイフン類はスキップ（self-reported 等を拾う）
                        if _is_skip_punct(raw[j]):
                            j += 1
                            continue
                        if lemma_lower[j] == key[k]:
                            j += 1
                            k += 1
                        else:
                            break
                    if k == len(key):
                        doc_tokens.append(joined)  # 強制結合はフィルタをバイパス
                        i = j
                        matched = True
                        break

            if matched:
                continue

            # (C) 強制一致しなかった位置は通常正規化（POS/NER/alpha_regex など）
            norm = normalizer(raw[i])
            if norm is not None:
                doc_tokens.append(norm)
            i += 1

        tokenized_docs.append(doc_tokens)

    return tokenized_docs


def compute_term_frequencies(per_doc_tokens: Sequence[Sequence[str]]) -> List[PerDocFreq]:
    """文書ごとの正規化トークン列から相対頻度を算出する。

    Args:
        per_doc_tokens (Sequence[Sequence[str]]): 文書単位の正規化トークン列。

    Returns:
        List[PerDocFreq]: 文書内相対頻度の辞書リスト。
    """

    per_doc_freqs: List[PerDocFreq] = []
    for tokens in per_doc_tokens:
        if not tokens:
            per_doc_freqs.append({})
            continue
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        per_doc_freqs.append({term: count / total for term, count in counts.items()})

    return per_doc_freqs


# ----------------------------
# キャッシュ付きラッパ
# ----------------------------
def _policy_fingerprint(policy: TokenPolicy) -> Dict[str, object]:
    """JSON シリアライズ可能なポリシー指紋を生成する。"""

    # frozenset を JSON 化可能な list に落とす
    return {
        "target_pos": sorted(policy.target_pos),
        "exclude_ner": sorted(policy.exclude_ner),
        "exclude_propn": policy.exclude_propn,
        "exclude_aux": policy.exclude_aux,
        "keep_surface_for": sorted(policy.keep_surface_for),
        "alpha_regex": policy.alpha_regex,
        "forced_phrases": [list(p) for p in sorted(policy.forced_phrases)],
        "forced_joiner": policy.forced_joiner,
        "forced_aliases": [
            {"key": list(key), "alias": alias}
            for key, alias in sorted(policy.forced_aliases, key=lambda item: (item[0], item[1]))
        ],
    }


def _make_key(texts: Iterable[str], policy: TokenPolicy, backend_id: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update((t or "").encode("utf-8"))
    h.update(json.dumps(_policy_fingerprint(policy), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(backend_id.encode("utf-8"))
    return h.hexdigest()


def analyze_docs_with_cache(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: List[str],
    policy: TokenPolicy,
    *,
    cache_dir: Optional[str] = None,
) -> List[PerDocTokens]:
    """文書解析結果（正規化トークン列）をキャッシュから取得または新規生成する。

    Args:
        backend (NLPBackend): トークン化・解析を行うバックエンド。
        normalizer (Normalizer): トークン正規化に使用する関数。
        texts (list[str]): 解析対象の本文。
        policy (TokenPolicy): 強制抽出や除外条件を含むポリシー。

    Keyword Args:
        cache_dir (str | None): トークン列を保存するキャッシュディレクトリ。

    Returns:
        List[PerDocTokens]: 文書ごとの正規化済みトークン列。
    """
    if cache_dir is None:
        return analyze_docs(backend, normalizer, texts, policy)

    os.makedirs(cache_dir, exist_ok=True)
    backend_id = getattr(backend, "model_name", backend.__class__.__name__)
    key = _make_key(texts, policy, str(backend_id))
    path = os.path.join(cache_dir, f"per_doc_tokens_{key}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    tokenized_docs = analyze_docs(backend, normalizer, texts, policy)
    with open(path, "wb") as f:
        pickle.dump(tokenized_docs, f)
    return tokenized_docs

