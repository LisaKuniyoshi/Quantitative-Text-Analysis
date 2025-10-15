# quant_text_analysis/cluster/algorithms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

_L2_NORM = "l2"

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """行方向に L2 正規化を施した行列を返す。

    Args:
        X (numpy.ndarray): 正規化対象の行列。

    Returns:
        numpy.ndarray: 各行のノルムが 1 となる行列。
    """
    if sp.issparse(X):
        X_csr = sp.csr_matrix(X)
        X_norm = sk_normalize(X_csr, norm=_L2_NORM, axis=1, copy=False)
        return X_norm.toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms

def cosine_inertia(X_unit: np.ndarray, labels: np.ndarray, centroids_unit: np.ndarray) -> float:
    """cos 類似度に基づくクラスタリングの慣性を計算する。

    Args:
        X_unit (numpy.ndarray): サンプルの単位ベクトル行列。
        labels (numpy.ndarray): 各サンプルのクラスタ割当。
        centroids_unit (numpy.ndarray): クラスタ重心の単位ベクトル行列。

    Returns:
        float: Σ(1 - cos(x_i, μ_{label_i})) の値。
    """
    return float(np.sum(1.0 - np.einsum("ij,ij->i", X_unit, centroids_unit[labels])))

@dataclass
class SKMeansResult:
    """spherical k-means の結果を保持するデータクラス。

    Attributes:
        labels_ (numpy.ndarray): サンプルごとのクラスタ割当。
        centroids_ (numpy.ndarray): 単位ベクトルで表現されたクラスタ重心。
        inertia_ (float): cos 慣性の値。
    """
    labels_: np.ndarray           # (n_samples,)
    centroids_: np.ndarray        # (k, d) unit
    inertia_: float               # Σ(1 - cos)

def _kmeanspp_cos(X_unit: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    cos 距離版 k-means++ 初期化。
    返り値は (k, d) の単位ベクトル。
    """
    n, d = X_unit.shape
    centers = np.empty((k, d), dtype=X_unit.dtype)

    # 1つ目
    i0 = int(rng.integers(0, n))
    centers[0] = X_unit[i0]

    # 残り（確率 ∝ 距離^2）
    closest = 1.0 - X_unit @ centers[0].T
    for c in range(1, k):
        probs = closest ** 2
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            j = int(rng.integers(0, n))
        else:
            j = int(rng.choice(n, p=probs / s))
        centers[c] = X_unit[j]
        dist_new = 1.0 - (X_unit @ centers[c].T)
        closest = np.minimum(closest, dist_new)
    return centers

def spherical_kmeans(
    X_unit: np.ndarray,
    k: int,
    n_init: int,
    max_iter: int,
    rng: Optional[np.random.Generator] = None
) -> SKMeansResult:
    """cos 類似度最大化を目的とした spherical k-means を実行する。

    Args:
        X_unit (numpy.ndarray): 行ごとに正規化されたサンプルベクトル。
        k (int): 生成するクラスタ数。
        n_init (int): 初期化回数。
        max_iter (int): 最大反復回数。
        rng (numpy.random.Generator | None): 乱数生成器。

    Returns:
        SKMeansResult: 最良のクラスタリング結果。
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = X_unit.shape
    best: Optional[SKMeansResult] = None

    for _ in range(n_init):
        centers = _kmeanspp_cos(X_unit, k, rng)
        labels = np.zeros(n, dtype=np.int32)

        for it in range(max_iter):
            # E-step: 割当（cos 最大）
            sims = X_unit @ centers.T  # (n, k)
            new_labels = np.argmax(sims, axis=1)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # M-step: 重心更新 → 正規化（空クラスタはランダム再配置）
            for c in range(k):
                idx = np.where(labels == c)[0]
                if len(idx) == 0:
                    centers[c] = X_unit[int(rng.integers(0, n))]
                else:
                    mean = X_unit[idx].mean(axis=0)
                    norm = np.linalg.norm(mean)
                    centers[c] = X_unit[int(rng.integers(0, n))] if norm == 0.0 else (mean / norm)

        inertia = cosine_inertia(X_unit, labels, centers)
        res = SKMeansResult(labels_=labels, centroids_=centers, inertia_=inertia)
        if best is None or res.inertia_ < best.inertia_:
            best = res

    assert best is not None
    return best
