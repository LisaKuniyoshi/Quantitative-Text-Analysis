from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple
import hashlib
import json
import os
import pickle
from typing import Iterable, Optional, Tuple

from breame.spelling import get_american_spelling

from ..data_types import DocResult, NLPBackend, Normalizer, TokenPolicy, TokenLike

# 強制抽出の際に「語間で無視する記号」：ハイフン類・細かい句読点のみ
_SKIP_PUNCTS = {"-", "‐", "‒", "–", "—", "―", "·", "•"}
# 句点やコロンなどは「語間の橋渡し」に使わない（= そこで強制一致は切れる）
_BREAK_PUNCTS = {".", "…", ":", ";", "/", "\\", "?", "!", ",", "(", ")", "[", "]", "{", "}", "“", "”", "‘", "’", "'"}

# ----------------------------
# 強制フレーズ辞書
# ----------------------------

def _build_forced_index(policy: TokenPolicy) -> Dict[Tuple[str, ...], str]:
    """強制抽出辞書（キー＝lemma列の小文字タプル、値＝結合トークン）"""
    idx: Dict[Tuple[str, ...], str] = {}
    for phrase in policy.forced_phrases:
        if not phrase:
            continue
        key = tuple(w.lower() for w in phrase)
        idx[key] = policy.forced_joiner.join(key)
    return idx


def _is_skip_punct(tok: TokenLike) -> bool:
    t = tok.text
    return (tok.pos_ == "PUNCT") and (t in _SKIP_PUNCTS)


def _is_break_punct(tok: TokenLike) -> bool:
    t = tok.text
    return (tok.pos_ == "PUNCT") and (t in _BREAK_PUNCTS)

# ----------------------------
# 主処理：文書ごとの正規化トークンと文書内相対頻度
# ----------------------------

def analyze_docs(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: Sequence[str],
    policy: TokenPolicy,
) -> Tuple[List[DocResult], List[Dict[str, float]]]:
    """
    パイプライン： (A) lemma 列の生成 → (B) lemma 列に対する強制抽出 → (C) 残余トークンを通常正規化
    - 強制抽出は lemma 列を使う（小文字）。語間のハイフン類は無視できる。
    - BREAK_PUNCTS を跨いだ一致は行わない。
    - 強制一致したトークンは、POS/NER/alpha_regex のフィルタをバイパスして採用する。
    """
    forced_index = _build_forced_index(policy)
    keys_by_len: List[Tuple[Tuple[str, ...], str]] = sorted(
        forced_index.items(), key=lambda kv: len(kv[0]), reverse=True
    )

    per_doc: List[DocResult] = []
    per_doc_freqs: List[Dict[str, float]] = []

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

        toks: List[str] = []
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
                        toks.append(joined)  # 強制結合はフィルタをバイパス
                        i = j
                        matched = True
                        break

            if matched:
                continue

            # (C) 強制一致しなかった位置は通常正規化（POS/NER/alpha_regex など）
            norm = normalizer(raw[i])
            if norm is not None:
                toks.append(norm)
            i += 1

        total = len(toks)
        per_doc.append(DocResult(tokens=tuple(toks), total=total))

        if total == 0:
            per_doc_freqs.append({})
        else:
            cnt: Dict[str, int] = Counter(toks)
            per_doc_freqs.append({w: c / float(total) for w, c in cnt.items()})

    return per_doc, per_doc_freqs

# ----------------------------
# キャッシュ付きラッパ
# ----------------------------

def _policy_fingerprint(policy: TokenPolicy) -> dict[str, object]:
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
    }

def _make_key(texts: Iterable[str], policy: TokenPolicy, backend_id: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update((t or "").encode("utf-8"))
    h.update(json.dumps(_policy_fingerprint(policy), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(backend_id.encode("utf-8"))
    return h.hexdigest()


def get_or_analyze_docs(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: List[str],
    policy: TokenPolicy,
    *,
    cache_dir: Optional[str] = None,
) -> Tuple[List[DocResult], List[Dict[str, float]]]:
    if cache_dir is None:
        return analyze_docs(backend, normalizer, texts,policy)

    os.makedirs(cache_dir, exist_ok=True)
    backend_id = getattr(backend, "model_name", backend.__class__.__name__)
    key = _make_key(texts, policy, str(backend_id))
    path = os.path.join(cache_dir, f"per_doc_freqs_{key}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    result = analyze_docs(backend, normalizer, texts, policy)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result
