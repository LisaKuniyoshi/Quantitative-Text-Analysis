from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from ..settings import Settings

ArrayLike = Union[np.ndarray, sp.spmatrix]


def _hash_key(meta: dict) -> str:
    s = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def _svd_embedding(
    X_wd: ArrayLike,
    svd_dim: int,
    *,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """PPMI 行列から SVD による語埋め込みを生成する。

    Args:
        X_wd (ArrayLike): 語×文書の PPMI 行列。
        svd_dim (int): 希望する埋め込み次元。
        random_state (int | None): SVD の乱数シード。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: 語埋め込み行列と説明分散比。
    """
    # SVD 次元は行列の最小次元-1を上限とする
    n_components = int(min(svd_dim, max(2, min(X_wd.shape) - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    Z = svd.fit_transform(X_wd)
    return Z, svd.explained_variance_ratio_


def get_or_svd_embedding(
    X_wd: ArrayLike,
    *,
    svd_dim: int,
    cfg: Optional[Settings] = None,
    ppmi_cache_key: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD 埋め込みをキャッシュから取得または計算して返す。

    Args:
        X_wd (ArrayLike): 語×文書の PPMI 行列。
        svd_dim (int): 生成する埋め込み次元。
        cfg (Settings | None): 設定オブジェクト。None の場合は既定値を利用。
        ppmi_cache_key (str | None): PPMI 計算時のキャッシュキー。
        random_state (int | None): SVD の乱数シード。未指定時は設定値を使用。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: 語埋め込み行列と説明分散比。
    """
    s = cfg or Settings()

    # キャッシュ不使用モード
    if s.cache_dir is None:
        return _svd_embedding(X_wd, svd_dim, random_state=s.random_seed)

    meta = {
        "ppmi_key": ppmi_cache_key,
        "svd_dim": int(svd_dim),
        "shape": (int(X_wd.shape[0]), int(X_wd.shape[1])),
    }
    key = _hash_key(meta)
    base = os.path.join(str(s.cache_dir), f"svd_{key}")
    path = base + ".npy"
    meta_path = base + ".json"

    if os.path.exists(path):
        Z = np.load(path)
        explained_ratio = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_cached = json.load(f)
            ratio_cached = meta_cached.get("explained_variance_ratio")
            if ratio_cached is not None:
                explained_ratio = np.asarray(ratio_cached, dtype=float)
        if explained_ratio is not None:
            return Z, explained_ratio
        Z, explained_ratio = _svd_embedding(X_wd, svd_dim, random_state=s.random_seed)
        np.save(path, Z)
        meta["explained_variance_ratio"] = explained_ratio.tolist()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return Z, explained_ratio

    Z, explained_ratio = _svd_embedding(X_wd, svd_dim, random_state=s.random_seed)

    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.save(path, Z)
    meta["explained_variance_ratio"] = explained_ratio.tolist()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return Z, explained_ratio
