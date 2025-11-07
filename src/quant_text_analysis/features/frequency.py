from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer

EMPTY_COLUMNS = ["word", "mean_freq", "n_docs_nonzero", "doc_rate_nonzero"]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_COLUMNS)


# r(d,w) のグループ集計（純粋関数）

def frequency_rankings(
    per_doc_freqs: list[dict[str, float]],
    groups: Optional[list[list[str]]] = None,
) -> dict[str, pd.DataFrame]:
    """語相対頻度をグループ別に集計しランキングを生成する。

    Args:
        per_doc_freqs (list[dict[str, float]]): 文書ごとの語相対頻度分布。
        groups (list[list[str]] | None): 文書が属するグループラベル。None の場合は全件を単一グループ扱い。

    Returns:
        dict[str, pandas.DataFrame]: グループ ID をキーに持つランキング表。
    """
    n_docs: int = len(per_doc_freqs)

    vec: DictVectorizer = DictVectorizer(dtype=np.float64, sparse=True, sort=False)
    X: sp.csr_matrix = sp.csr_matrix(vec.fit_transform(per_doc_freqs), copy=False)
    feature_names: np.ndarray = np.asarray(vec.get_feature_names_out(), dtype=str)

    group_docs: dict[str, list[int]] = {}
    if groups is None:
        group_docs = {"ALL": [i for i in range(n_docs)]}
    else:
        for i, g in enumerate(groups):
            for gg in g:
                group_docs.setdefault(str(gg), []).append(i)

    out: dict[str, pd.DataFrame] = {}
    for g, idxs in group_docs.items():
        if not idxs:
            out[g] = _empty_df()
            continue

        sub: sp.csr_matrix = X[idxs]
        mean: np.ndarray = np.asarray(sub.mean(axis=0)).ravel()
        nz: np.ndarray = np.asarray(sub.getnnz(axis=0), dtype=np.int32)

        mask: np.ndarray = nz > 0

        if not np.any(mask):
            out[g] = _empty_df()
            continue

        words = feature_names[mask]
        mean_sel = mean[mask]
        nz_sel = nz[mask]

        df = pd.DataFrame({
            "word": words,
            "mean_freq": mean_sel,
            "n_docs_nonzero": nz_sel,
        })
        df["doc_rate_nonzero"] = df["n_docs_nonzero"] / float(len(idxs))
        df = df.sort_values(["mean_freq", "n_docs_nonzero"], ascending=[False, False], kind="mergesort")
        out[g] = df.reset_index(drop=True)

    return out