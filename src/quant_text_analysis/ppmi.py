# quant_text_analysis/ppmi.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import json
import math
import os

import numpy as np
import scipy.sparse as sp

from .data_types import DocResult, TokenPolicy, NLPBackend, Normalizer
from .cache import analyze_with_cache
from .frequency import frequency_rankings

EPS: float = 1e-8  # 数値安定用（log のゼロ回避）

@dataclass(frozen=True)
class PPMIOutputs:
    vocab: List[str]                 # 行=語の順序
    doc_ids: List[int]               # 0..D-1
    X_tf: sp.csr_matrix              # D x V（正規化TF；語彙に制限後の射影行列）
    ppmi_word_doc: sp.csr_matrix     # V x D（非対称PPMI；行=語）
    ppmi_word_word: sp.csr_matrix    # V x V（対称PPMI；行=語）

# ----------------------------
# 語彙選定（cli.py と同じ規則）
# ----------------------------
def _select_vocab_by_rankings(
    per_doc_freqs: List[Dict[str, float]],
    *,
    top_n: int,
    min_docs: int
) -> List[str]:
    df_all = frequency_rankings(per_doc_freqs, None, top_n=top_n, min_docs=min_docs)["ALL"]
    # 想定カラム名: ["word", "mean_freq", "n_docs_nonzero", "doc_rate_nonzero"]
    return df_all["word"].astype(str).tolist()

# ----------------------------
# 射影行列の構築（X: D x V）
# ----------------------------
def _build_projected_tf_matrix(
    per_doc_freqs: List[Dict[str, float]],
    vocab: Sequence[str],
) -> sp.csr_matrix:
    D = len(per_doc_freqs)
    V = len(vocab)
    index = {w: j for j, w in enumerate(vocab)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for d, freq in enumerate(per_doc_freqs):
        if not freq:
            continue
        for w, r in freq.items():  # r は「語彙制限前」の正規化TF（c/total）
            j = index.get(w)
            if j is not None and r > 0.0:
                rows.append(d); cols.append(j); data.append(float(r))
    return sp.csr_matrix((data, (rows, cols)), shape=(D, V), dtype=np.float64)

# ----------------------------
# 語彙制限「前」の周辺量（不変性のための分母）
# ----------------------------
def _compute_marginals_fullspace(
    per_doc_freqs: List[Dict[str, float]],
    vocab: Sequence[str],
) -> Tuple[np.ndarray, int]:
    """
    s_w[i] = ∑_d x_full[d, i] を語彙制限前の空間で計算（語が文書に無ければ0）。
    T = D（各文書の行和=1 なので、総量は文書数）
    """
    D = len(per_doc_freqs)
    V = len(vocab)
    s_w = np.zeros(V, dtype=np.float64)
    for d in range(D):
        freq = per_doc_freqs[d]
        for i, w in enumerate(vocab):
            r = freq.get(w, 0.0)
            if r:
                s_w[i] += r
    return s_w, D

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
def compute_ppmi_from_perdocfreqs(
    per_doc_freqs: List[Dict[str, float]],
    *,
    top_n: int,
    min_docs: int,
) -> PPMIOutputs:
    vocab = _select_vocab_by_rankings(per_doc_freqs, top_n=top_n, min_docs=min_docs)
    doc_ids = list(range(len(per_doc_freqs)))

    # 語彙へ射影した X（D x V）と、語彙制限「前」の周辺（s_w）を分離して作る
    X = _build_projected_tf_matrix(per_doc_freqs, vocab)  # D x V（各行は ≤1）
    s_w, D_docs = _compute_marginals_fullspace(per_doc_freqs, vocab)  # s_w: V, T=D_docs

    # 非対称・対称の PPMI（定義は元空間、分子のみ射影後）
    ppmi_wd = _ppmi_word_doc_from_fullspace(X, s_w, D_docs)  # V x D
    ppmi_ww = _ppmi_word_word_from_fullspace(X, s_w, D_docs) # V x V

    return PPMIOutputs(vocab, doc_ids, X.tocsr(), ppmi_wd.tocsr(), ppmi_ww.tocsr())

# ----------------------------
# キャッシュ付き入口（per-doc 前処理も含む）
# ----------------------------
def _hash_for_cache(obj: object) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def compute_ppmi_with_cache(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: List[str],
    policy: TokenPolicy,
    *,
    cache_dir: Optional[str],
    top_n: int,
    min_docs: int,
) -> PPMIOutputs:
    # per-doc 形態素処理＋正規化TFは既存のキャッシュを再利用
    per_doc, per_doc_freqs = analyze_with_cache(backend, normalizer, texts, policy, cache_dir=cache_dir)

    meta_key = {
        "backend": getattr(backend, "model_name", backend.__class__.__name__),
        "policy": {
            "target_pos": sorted(policy.target_pos),
            "exclude_ner": sorted(policy.exclude_ner),
            "exclude_propn": policy.exclude_propn,
            "exclude_aux": policy.exclude_aux,
            "keep_surface_for": sorted(policy.keep_surface_for),
            "forced_phrases": [list(t) for t in policy.forced_phrases],
            "forced_joiner": policy.forced_joiner,
        },
        "n_texts": len(per_doc),
        "top_n": top_n,
        "min_docs": min_docs,
        "ppmi_epsilon": EPS,
        "ppmi_log_base": "e",
        "mode": "tf_norm_fullspace_denominator",  # 不変性担保の方式
    }
    key = _hash_for_cache(meta_key)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        base = os.path.join(cache_dir, f"ppmi_{key}")
        meta_path  = base + "_meta.json"
        wd_path    = base + "_wd.npz"
        ww_path    = base + "_ww.npz"
        vocab_path = base + "_vocab.json"
        X_path     = base + "_tf.npz"
        if all(os.path.exists(p) for p in [meta_path, wd_path, ww_path, vocab_path, X_path]):
            vocab = json.load(open(vocab_path, "r", encoding="utf-8"))["vocab"]
            X = sp.load_npz(X_path)
            wd = sp.load_npz(wd_path)
            ww = sp.load_npz(ww_path)
            return PPMIOutputs(vocab, list(range(X.shape[0])), X, wd, ww)

    out = compute_ppmi_from_perdocfreqs(per_doc_freqs, top_n=top_n, min_docs=min_docs)

    if cache_dir is not None:
        base = os.path.join(cache_dir, f"ppmi_{key}")
        json.dump(meta_key, open(base + "_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump({"vocab": out.vocab}, open(base + "_vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        sp.save_npz(base + "_tf.npz", out.X_tf)
        sp.save_npz(base + "_wd.npz", out.ppmi_word_doc)
        sp.save_npz(base + "_ww.npz", out.ppmi_word_word)

    return out
