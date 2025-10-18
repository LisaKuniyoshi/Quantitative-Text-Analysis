# quant_text_analysis/cluster/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple, Literal
import numpy as np
from sklearn.decomposition import TruncatedSVD

from ..settings import Settings
from ..features.vocab_selection import build_filtered_tf_matrix

from .algorithms import l2_normalize_rows, spherical_kmeans

def top_terms_by_centroid(
    X_unit: np.ndarray,
    vocab: List[str],
    centroids_unit: np.ndarray,
    top_n: int
) -> Dict[int, List[Tuple[str, float]]]:
    """クラスタ重心と語ベクトルの cos 類似度から上位語を抽出する。

    Args:
        X_unit (numpy.ndarray): 語ベクトル行列（各行が単位ベクトル）。
        vocab (list[str]): 語彙リスト。
        centroids_unit (numpy.ndarray): クラスタ重心の単位ベクトル行列。
        top_n (int): 返す語数。

    Returns:
        dict[int, list[tuple[str, float]]]: クラスタ ID ごとの語と類似度。
    """
    sims = X_unit @ centroids_unit.T  # (V, k)
    out: Dict[int, List[Tuple[str, float]]] = {}
    for c in range(sims.shape[1]):
        idx = np.argsort(sims[:, c])[::-1][:top_n]
        out[c] = [(vocab[i], float(sims[i, c])) for i in idx]
    return out

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
