# quant_text_analysis/cluster/metrics.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD

from ..features.ppmi import get_or_compute_ppmi
from .algorithms import l2_normalize_rows, spherical_kmeans

def top_terms_by_centroid(
    X_unit: np.ndarray,
    vocab: List[str],
    centroids_unit: np.ndarray,
    top_n: int
) -> Dict[int, List[Tuple[str, float]]]:
    """クラスタ重心と語ベクトルの cos 類似度から上位語を抽出する。

    Parameters
    ----------
    X_unit : numpy.ndarray
        語ベクトル行列（各行が単位ベクトル）。
    vocab : List[str]
        語彙リスト。
    centroids_unit : numpy.ndarray
        クラスタ重心の単位ベクトル行列。
    top_n : int
        返す語数。

    Returns
    -------
    Dict[int, List[Tuple[str, float]]]
        クラスタ ID ごとの語と類似度のリスト。
    """
    sims = X_unit @ centroids_unit.T  # (V, k)
    out: Dict[int, List[Tuple[str, float]]] = {}
    for c in range(sims.shape[1]):
        idx = np.argsort(sims[:, c])[::-1][:top_n]
        out[c] = [(vocab[i], float(sims[i, c])) for i in idx]
    return out

def jaccard(a: List[str] | set[str], b: List[str] | set[str]) -> float:
    """2 集合間の Jaccard 係数を計算する。

    Parameters
    ----------
    a : list[str] or set[str]
        一方の語集合。
    b : list[str] or set[str]
        もう一方の語集合。

    Returns
    -------
    float
        Jaccard 係数。双方空集合の場合は 1.0。
    """
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def stability_top_terms_jaccard(
    per_doc_freqs: List[Dict[str, float]],
    *,
    k: int,
    svd_dim: int,
    rng: np.random.Generator,
    top_words_per_cluster: int = 20,
    max_iter: int = 300
) -> float:
    """クラスタ上位語集合の安定性を Jaccard 係数で測定する。

    Parameters
    ----------
    per_doc_freqs : List[Dict[str, float]]
        文書ごとの語確率分布。
    k : int
        クラスタ数。
    svd_dim : int
        SVD の潜在次元。
    rng : numpy.random.Generator
        乱数生成器。
    top_words_per_cluster : int, default 20
        各クラスタで抽出する上位語数。
    max_iter : int, default 300
        spherical k-means の最大反復回数。

    Returns
    -------
    float
        対応付けたクラスタ間の Jaccard 係数平均。計算不能時は NaN。
    """
    n_docs = len(per_doc_freqs)
    perm = rng.permutation(n_docs)
    A = sorted(perm[: n_docs // 2].tolist())
    B = sorted(perm[n_docs // 2 :].tolist())

    def build_word_space(doc_idx: List[int]) -> Tuple[List[str], np.ndarray]:
        sub = [per_doc_freqs[i] for i in doc_idx]
        out = get_or_compute_ppmi(sub)
        X_wd = out.ppmi_word_doc  # (V, D)
        svd = TruncatedSVD(n_components=min(svd_dim, max(2, min(X_wd.shape) - 1)))
        Z = svd.fit_transform(X_wd)
        return out.vocab, l2_normalize_rows(Z)

    vocabA, ZA = build_word_space(A)
    vocabB, ZB = build_word_space(B)

    resA = spherical_kmeans(ZA, k=k, n_init=10, max_iter=max_iter, rng=rng)
    resB = spherical_kmeans(ZB, k=k, n_init=10, max_iter=max_iter, rng=rng)

    topA = top_terms_by_centroid(ZA, vocabA, resA.centroids_, top_words_per_cluster)
    topB = top_terms_by_centroid(ZB, vocabB, resB.centroids_, top_words_per_cluster)

    usedB: set[int] = set()
    scores: List[float] = []
    for ca in range(k):
        Sa = [w for w, _ in topA[ca]]
        best, best_cb = -1.0, -1
        for cb in range(k):
            if cb in usedB:
                continue
            Sb = [w for w, _ in topB[cb]]
            s = jaccard(Sa, Sb)
            if s > best:
                best, best_cb = s, cb
        if best_cb >= 0:
            usedB.add(best_cb)
            scores.append(best)
    if not scores:
        return float("nan")
    return float(np.mean(scores))

def abstract_cluster_ratio(
    per_doc_freqs: List[Dict[str, float]],
    vocab: List[str],
    labels: np.ndarray
) -> np.ndarray:
    """文書ごとのクラスタ比率行列を算出する。

    Parameters
    ----------
    per_doc_freqs : List[Dict[str, float]]
        文書内語の確率分布。
    vocab : List[str]
        語彙リスト。
    labels : numpy.ndarray
        語彙に対応するクラスタラベル。

    Returns
    -------
    numpy.ndarray
        文書 × クラスタの比率行列。
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
