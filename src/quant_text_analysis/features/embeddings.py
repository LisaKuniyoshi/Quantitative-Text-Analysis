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
    """
    純粋関数：PPMI(語×文書)行列から SVD による語埋め込みを生成する。
    settings やキャッシュには一切触れない。
    戻り値 Z の形状は (V, svd_dim')。svd_dim' は次元制約により svd_dim から減衰し得る。
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
    """
    settings を参照し、SVD 埋め込みをキャッシュから取得する。
    キャッシュが無ければ `_svd_embedding` で生成して保存し、その結果を返す。

    キーは {ppmi_cache_key, svd_dim, shape, nnz} から算出する。
    cache_dir が未設定(None)の場合はキャッシュを使わず計算のみ行う。
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
