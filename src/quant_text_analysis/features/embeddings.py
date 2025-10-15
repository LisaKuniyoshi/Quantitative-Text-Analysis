from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Union

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
) -> np.ndarray:
    """PPMI 行列から SVD による語埋め込みを生成する。

    Args:
        X_wd (ArrayLike): 語×文書の PPMI 行列。
        svd_dim (int): 希望する埋め込み次元。
        random_state (int | None): SVD の乱数シード。

    Returns:
        numpy.ndarray: 形状 (V, svd_dim') の語埋め込み行列。
    """
    # SVD 次元は行列の最小次元-1を上限とする
    n_components = int(min(svd_dim, max(2, min(X_wd.shape) - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    Z = svd.fit_transform(X_wd)
    return Z


def get_or_svd_embedding(
    X_wd: ArrayLike,
    *,
    cfg: Optional[Settings] = None,
    ppmi_cache_key: Optional[str] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """SVD 埋め込みをキャッシュから取得または計算して返す。

    Args:
        X_wd (ArrayLike): 語×文書の PPMI 行列。
        cfg (Settings | None): 設定オブジェクト。None の場合は既定値を利用。
        ppmi_cache_key (str | None): PPMI 計算時のキャッシュキー。
        random_state (int | None): SVD の乱数シード。未指定時は設定値を使用。

    Returns:
        numpy.ndarray: 語埋め込み行列。キャッシュ未使用時は新たに計算される。
    """
    s = cfg or Settings()

    # キャッシュ不使用モード
    if s.cache_dir is None:
        return _svd_embedding(X_wd, s.svd_dim, random_state=random_state or s.random_seed)

    meta = {
        "ppmi_key": ppmi_cache_key,
        "svd_dim": int(s.svd_dim),
        "shape": (int(X_wd.shape[0]), int(X_wd.shape[1])),
    }
    key = _hash_key(meta)
    base = os.path.join(str(s.cache_dir), f"svd_{key}")
    path = base + ".npy"
    meta_path = base + ".json"

    if os.path.exists(path):
        return np.load(path)

    Z = _svd_embedding(X_wd, s.svd_dim, random_state=random_state or s.random_seed)

    os.makedirs(os.path.dirname(base), exist_ok=True)
    np.save(path, Z)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return Z
