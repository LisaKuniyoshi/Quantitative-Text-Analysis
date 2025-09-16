from __future__ import annotations
from collections import Counter
from typing import Dict, List, Sequence, Tuple
from .data_types import DocResult, NLPBackend, Normalizer, TokenPolicy, TokenLike

# 文書単位 r(d,w) 作成（純粋関数）

def _build_forced_index(policy: TokenPolicy) -> Dict[Tuple[str, ...], str]:
    idx: Dict[Tuple[str, ...], str] = {}
    for phrase in policy.forced_phrases:
        if not phrase:
            continue
        key = tuple(w.lower() for w in phrase)
        idx[key] = policy.forced_joiner.join(key)
    return idx

def analyze_docs(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: Sequence[str],
    policy: TokenPolicy,                 # ★ 追加
) -> Tuple[List[DocResult], List[Dict[str, float]]]:
    forced_index = _build_forced_index(policy)
    max_len = max((len(k) for k in forced_index.keys()), default=0)

    per_doc: List[DocResult] = []
    per_doc_freqs: List[Dict[str, float]] = []

    for doc in backend.pipe(texts):
        # 生トークン（TokenLike）と小文字列
        raw: List[TokenLike] = [tok for tok in doc]
        lowers: List[str] = [t.text.lower() for t in raw]

        toks: List[str] = []
        i = 0
        n = len(raw)

        while i < n:
            matched = False
            if max_len > 0:
                # 最長一致（max_len→1 の順に探索）
                max_try = min(max_len, n - i)
                for L in range(max_try, 0, -1):
                    key = tuple(lowers[i : i + L])
                    if key in forced_index:
                        toks.append(forced_index[key])  # ★ 強制結合トークンを追加
                        i += L
                        matched = True
                        break
            if matched:
                continue

            # 通常正規化
            norm = normalizer(raw[i])
            if norm is not None:
                toks.append(norm)
            i += 1

        total = len(toks)
        per_doc.append(DocResult(tokens=tuple(toks), total=total))

        if total == 0:
            per_doc_freqs.append({})
        else:
            cnt: Dict[str, int] = {}
            for w in toks:
                cnt[w] = cnt.get(w, 0) + 1
            per_doc_freqs.append({w: c / float(total) for w, c in cnt.items()})

    return per_doc, per_doc_freqs