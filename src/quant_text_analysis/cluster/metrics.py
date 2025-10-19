"""Cluster-level metrics computed from per-document frequency data."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

PerDocFreq = Dict[str, float]

def abstract_cluster_ratio(
    per_doc_freqs: List[PerDocFreq],
    vocab: List[str],
    labels: np.ndarray
) -> np.ndarray:
    """文書ごとのクラスタ比率行列を算出する。

    Args:
        per_doc_freqs (list[dict[str, float]]): 文書内語の確率分布。
        vocab (list[str]): 語彙リスト。
        labels (numpy.ndarray): 語彙に対応するクラスタラベル。

    Returns:
        numpy.ndarray: 文書 × クラスタの比率行列。
    """
    if not vocab:
        raise ValueError("vocab must be non-empty to compute cluster ratios.")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if labels.size == 0:
        raise ValueError("labels must contain at least one element.")

    word2idx = {w: i for i, w in enumerate(vocab)}
    k = int(labels.max()) + 1
    if k <= 0:
        raise ValueError("labels must contain at least one cluster assignment.")

    n_docs = len(per_doc_freqs)
    M = np.zeros((n_docs, k), dtype=np.float64)
    for d, freq in enumerate(per_doc_freqs):
        denom = 0.0
        for w, p in freq.items():
            denom += p
            j = word2idx.get(w)
            if j is not None:
                M[d, labels[j]] += p
        if denom > 0:
            M[d] /= denom
    return M
