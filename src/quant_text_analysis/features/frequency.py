from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# r(d,w) のグループ集計（純粋関数）

def frequency_rankings(
    per_doc_freqs: List[Dict[str, float]],
    groups: Optional[List[Optional[str]]] = None,
    *,
    top_n: int = 50,
    min_docs: int = 1,
) -> Dict[str, pd.DataFrame]:
    n_docs = len(per_doc_freqs)
    gvec: List[str] = ["ALL"] * n_docs if groups is None else [g or "__NULL__" for g in groups]

    # グループ→文書インデックス
    group_docs: Dict[str, List[int]] = {}
    for i, g in enumerate(gvec):
        if g == "__NULL__":
            continue
        group_docs.setdefault(g, []).append(i)

    # 語彙集合
    vocab: Set[str] = set()
    for fd in per_doc_freqs:
        vocab.update(fd.keys())

    out: Dict[str, pd.DataFrame] = {}
    for g, idxs in group_docs.items():
        rows: List[Tuple[str, float, int, float]] = []
        n_g = len(idxs)
        for w in vocab:
            s = 0.0
            nz = 0
            for i in idxs:
                r = per_doc_freqs[i].get(w, 0.0)
                s += r
                if r > 0.0:
                    nz += 1
            if nz < min_docs:
                continue
            mean = s / float(n_g)
            rows.append((w, mean, nz, nz / float(n_g)))
        df = (
            pd.DataFrame(rows, columns=["word", "mean_freq", "n_docs_nonzero", "doc_rate_nonzero"])  # type: ignore[reportUnknownMemberType]
            .sort_values(["mean_freq", "n_docs_nonzero"], ascending=[False, False], kind="mergesort")  # type: ignore[reportUnknownMemberType]
            .head(top_n)
            .reset_index(drop=True)
        )
        out[g] = df
    return out