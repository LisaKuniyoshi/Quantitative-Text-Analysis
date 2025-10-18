# quant_text_analysis/cluster/metrics.py
from __future__ import annotations
from typing import Dict, List
import numpy as np

def abstract_cluster_ratio(
    per_doc_freqs: List[Dict[str, float]],
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
    word2idx = {w: i for i, w in enumerate(vocab)}
    k = int(labels.max()) + 1
    D = len(per_doc_freqs)
    M = np.zeros((D, k), dtype=np.float64)
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
