# quant_text_analysis/ppmi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import hashlib
import json
import math
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer

from ..settings import Settings

EPS: float = 1e-8  # 数値安定用（log のゼロ回避）

s = Settings()

@dataclass(frozen=True)
class PPMIOutputs:
    """PPMI 計算から得られる主要な成果物を保持するデータクラス。

    Attributes:
        vocab (list[str]): 語彙リスト（行順）。
        doc_ids (list[int]): 文書 ID のリスト。
        X_tf (scipy.sparse.csr_matrix): 文書×語の正規化 TF 行列。
        ppmi_word_doc (scipy.sparse.csr_matrix): 語×文書の非対称 PPMI 行列。
        ppmi_word_word (scipy.sparse.csr_matrix): 語×語の対称 PPMI 行列。
        cache_key (str | None): キャッシュ識別子。未設定時は None。
    """
    vocab: List[str]                 # 行=語の順序
    doc_ids: List[int]               # 0..D-1
    X_tf: sp.csr_matrix              # D x V（正規化TF；語彙に制限後の射影行列）
    ppmi_word_doc: sp.csr_matrix     # V x D（非対称PPMI；行=語）
    ppmi_word_word: sp.csr_matrix    # V x V（対称PPMI；行=語）
    cache_key: Optional[str] = None  # キャッシュ識別子（省略可）

# ----------------------------
# 非対称 PPMI（語×要旨）
# ----------------------------
def _ppmi_word_doc_from_fullspace(
    X: sp.csr_matrix,         # D x V（語彙へ射影済み）
    s_w: np.ndarray,          # 語の周辺（語彙制限前）
    D_docs: int,              # 総量 T
) -> sp.csr_matrix:
    X_coo = X.tocoo()
    rows_w: List[int] = []
    cols_d: List[int] = []
    vals: List[float] = []
    for d, w, r_wd in zip(X_coo.row, X_coo.col, X_coo.data):
        if r_wd <= 0.0:
            continue
        denom = s_w[w] + EPS
        pmi = math.log((r_wd * float(D_docs)) / denom + EPS)
        if pmi > 0.0:
            rows_w.append(w); cols_d.append(d); vals.append(pmi)
    V = X_coo.shape[1]
    D = X_coo.shape[0]
    return sp.csr_matrix((vals, (rows_w, cols_d)), shape=(V, D), dtype=np.float64)

# ----------------------------
# 対称 PPMI（語×語）
# ----------------------------
def _ppmi_word_word_from_fullspace(
    X: sp.csr_matrix,
    s_w: np.ndarray,
    D_docs: int,
) -> sp.csr_matrix:
    C = sp.csr_matrix(X.T @ X, dtype=np.float64)
    C.setdiag(0.0)
    C.eliminate_zeros()
    C = C.tocoo()
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    T = float(D_docs)
    for i, j, cij in zip(C.row, C.col, C.data):
        denom = (s_w[i] * s_w[j]) + EPS
        pmi = math.log((cij * T) / denom + EPS)
        if pmi > 0.0:
            rows.append(i); cols.append(j); vals.append(pmi)

    M = sp.csr_matrix((vals, (rows, cols)), shape=C.shape, dtype=np.float64)
    # 数値対称性の担保
    Msym = 0.5 * (M + M.T)
    Msym.eliminate_zeros()
    return Msym

# ----------------------------
# 上位 API（per_doc_freqs → PPMI）
# ----------------------------
def _compute_ppmi_from_perdocfreqs(
    per_doc_freqs: List[Dict[str, float]],
    top_n: int,
    min_docs: int,
) -> PPMIOutputs:
    doc_ids = list(range(len(per_doc_freqs)))
    vec = DictVectorizer(dtype=np.float64, sparse=True, sort=False)
    X_full = sp.csr_matrix(vec.fit_transform(per_doc_freqs), copy=False)
    vocab_full = list(vec.get_feature_names_out())

    D_docs = len(per_doc_freqs)
    if X_full.shape[1] == 0 or top_n <= 0:
        empty_tf = sp.csr_matrix((D_docs, 0), dtype=np.float64)
        empty_wd = sp.csr_matrix((0, D_docs), dtype=np.float64)
        empty_ww = sp.csr_matrix((0, 0), dtype=np.float64)
        return PPMIOutputs([], doc_ids, empty_tf, empty_wd, empty_ww)

    doc_counts = np.asarray(X_full.getnnz(axis=0), dtype=np.int32)
    if min_docs > 1:
        mask = doc_counts >= min_docs
    else:
        mask = doc_counts > 0

    candidate_idx = np.flatnonzero(mask)
    if candidate_idx.size == 0:
        empty_tf = sp.csr_matrix((D_docs, 0), dtype=np.float64)
        empty_wd = sp.csr_matrix((0, D_docs), dtype=np.float64)
        empty_ww = sp.csr_matrix((0, 0), dtype=np.float64)
        return PPMIOutputs([], doc_ids, empty_tf, empty_wd, empty_ww)

    mean_freq = np.asarray(X_full.mean(axis=0)).ravel()
    order = np.lexsort((-doc_counts[candidate_idx], -mean_freq[candidate_idx]))
    ordered_idx = candidate_idx[order]

    if top_n < len(ordered_idx):
        ordered_idx = ordered_idx[:top_n]

    X = X_full[:, ordered_idx].tocsr()
    vocab = [vocab_full[i] for i in ordered_idx]

    s_w = np.asarray(X.sum(axis=0)).ravel()

    # 非対称・対称の PPMI（定義は元空間、分子のみ射影後）
    ppmi_wd = _ppmi_word_doc_from_fullspace(X, s_w, D_docs)
    ppmi_ww = _ppmi_word_word_from_fullspace(X, s_w, D_docs)

    return PPMIOutputs(vocab, doc_ids, X, ppmi_wd.tocsr(), ppmi_ww.tocsr())

# ----------------------------
# キャッシュ付き入口（per-doc 前処理も含む）
# ----------------------------
def _hash_for_cache(obj: object) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def _hash_per_doc_freqs(per_doc_freqs: List[Dict[str, float]]) -> str:
    """per_doc_freqs から軽量ダイジェストを生成する。

    Args:
        per_doc_freqs (list[dict[str, float]]): 文書ごとの語相対頻度。

    Returns:
        str: 16 文字のハッシュ値。
    """
    h = hashlib.sha256()
    h.update(f"D={len(per_doc_freqs)};".encode("utf-8"))
    for freq in per_doc_freqs:
        dh = hashlib.sha256()
        for w, p in sorted(freq.items()):
            dh.update(w.encode("utf-8")); dh.update(b"=")
            dh.update(("{:.12g}".format(float(p))).encode("ascii")); dh.update(b";")
        h.update(dh.digest())
    return h.hexdigest()[:16]

def get_or_compute_ppmi(
    per_doc_freqs: List[Dict[str, float]],
) -> PPMIOutputs:
    """per-doc 頻度から PPMI を計算しキャッシュを活用して返す。

    Args:
        per_doc_freqs (list[dict[str, float]]): 文書ごとの語相対頻度分布。

    Returns:
        PPMIOutputs: 語彙・TF 行列・PPMI 行列のセット。キャッシュ利用時は `cache_key` を含む。
    """
    source_meta = {"source": "per_doc_freqs", "digest": _hash_per_doc_freqs(per_doc_freqs)}
    base_key = source_meta["digest"]

    meta_key = {
        **source_meta,
        "top_n": int(s.top_n),
        "min_docs": int(s.min_docs),
        "ppmi_log_base": "e",
        "mode": "tf_norm_fullspace_denominator",
    }
    key = _hash_for_cache(meta_key)

    if s.cache_dir is not None:
        os.makedirs(s.cache_dir, exist_ok=True)
        base = os.path.join(s.cache_dir, f"ppmi_{key}")
        meta_path  = base + "_meta.json"
        wd_path    = base + "_wd.npz"
        ww_path    = base + "_ww.npz"
        vocab_path = base + "_vocab.json"
        X_path     = base + "_tf.npz"
        if all(os.path.exists(p) for p in [meta_path, wd_path, ww_path, vocab_path, X_path]):
            vocab = json.load(open(vocab_path, "r", encoding="utf-8"))["vocab"]
            X = sp.load_npz(X_path); wd = sp.load_npz(wd_path); ww = sp.load_npz(ww_path)
            return PPMIOutputs(vocab, list(range(X.shape[0])), X, wd, ww, cache_key=key)

    out = _compute_ppmi_from_perdocfreqs(per_doc_freqs, top_n=s.top_n, min_docs=s.min_docs)

    if s.cache_dir is not None:
        base = os.path.join(s.cache_dir, f"ppmi_{key}")
        json.dump(meta_key, open(base + "_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump({"vocab": out.vocab}, open(base + "_vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        sp.save_npz(base + "_tf.npz", out.X_tf)
        sp.save_npz(base + "_wd.npz", out.ppmi_word_doc)
        sp.save_npz(base + "_ww.npz", out.ppmi_word_word)

    return PPMIOutputs(out.vocab, out.doc_ids, out.X_tf, out.ppmi_word_doc, out.ppmi_word_word, cache_key=key)
