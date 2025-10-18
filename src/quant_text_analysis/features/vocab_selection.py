from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer


def build_filtered_tf_matrix(
    per_doc_freqs: List[Dict[str, float]],
    *,
    top_n: int,
    min_docs: int,
) -> Tuple[sp.csr_matrix, List[str]]:
    """文書ごとの語頻度辞書をベクトル化し、語彙をフィルタリングする。

    Args:
        per_doc_freqs: 文書ごとの語相対頻度辞書のリスト。
        top_n: 残す語の最大語彙数。
        min_docs: 語が出現すべき最小文書数。

    Returns:
        tuple[sp.csr_matrix, list[str]]: フィルタ済みの文書-語行列と語彙リスト。
    """
    n_docs = len(per_doc_freqs)
    vec = DictVectorizer(dtype=np.float64, sparse=True, sort=False)
    X_full = sp.csr_matrix(vec.fit_transform(per_doc_freqs), copy=False)
    vocab_full = list(vec.get_feature_names_out())

    if X_full.shape[1] == 0 or top_n <= 0:
        return sp.csr_matrix((n_docs, 0), dtype=np.float64), []

    doc_counts = np.asarray(X_full.getnnz(axis=0), dtype=np.int32)
    if min_docs > 1:
        mask = doc_counts >= min_docs
    else:
        mask = doc_counts > 0

    candidate_idx = np.flatnonzero(mask)
    if candidate_idx.size == 0:
        return sp.csr_matrix((n_docs, 0), dtype=np.float64), []

    mean_freq = np.asarray(X_full.mean(axis=0)).ravel()
    order = np.lexsort((-doc_counts[candidate_idx], -mean_freq[candidate_idx]))
    selected_idx = candidate_idx[order]
    if top_n < selected_idx.size:
        selected_idx = selected_idx[:top_n]

    X = X_full[:, selected_idx].tocsr()
    vocab = [vocab_full[i] for i in selected_idx]
    return X, vocab
