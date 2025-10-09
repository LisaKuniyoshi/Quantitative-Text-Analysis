# quant_text_analysis/cluster_cli.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Literal
from typing import Optional, Sequence, cast

import json
import numpy as np
import scipy.sparse as sp

from .config import default_columns, default_token_policy
from .io_loader import load_df
from .nlp_backend import SpacyBackend
from .normalize import build_normalizer
from .cache import analyze_with_cache
from .ppmi import compute_ppmi_from_perdocfreqs

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize as sk_normalize
from numpy.random import default_rng

# ----------------------------
# 設定
# ----------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

CSV_PATH: Path = RAW_DIR / "エクスポートされたアイテム.csv"

# PPMI -> 語彙制限（ppmi_cli.py と整合）
TOP_N: int = 10_000       # 語彙サイズ（上位）
MIN_DOCS: int = 7         # 最小出現文書数

# 次元圧縮（非対称のみ）
SVD_DIM: int = 200

# spherical k-means
K_LIST: Tuple[int, ...] = (12, 16, 20)
N_INIT: int = 20
MAX_ITER: int = 300
RANDOM_SEED: int = 42

TOP_WORDS_PER_CLUSTER: int = 20

_L2_NORM: Literal["l2"] = "l2"

# ----------------------------
# ユーティリティ
# ----------------------------
def l2_normalize_rows(X: np.ndarray | sp.spmatrix) -> np.ndarray:
    """行ごとのL2正規化（密にして返す）"""
    if sp.issparse(X):
        X_csr = sp.csr_matrix(X)
        X_norm = sk_normalize(X_csr, norm=_L2_NORM, axis=1, copy=False)
        return X_norm.toarray()
    else:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms

def cosine_inertia(X_unit: np.ndarray, labels: np.ndarray, centroids_unit: np.ndarray) -> float:
    """1 - cos を総和した損失。小さいほど良い。"""
    return float(np.sum(1.0 - np.einsum("ij,ij->i", X_unit, centroids_unit[labels])))

def top_terms_by_centroid(
    X_unit: np.ndarray,
    vocab: List[str],
    centroids_unit: np.ndarray,
    top_n: int
) -> Dict[int, List[Tuple[str, float]]]:
    """各クラスタ重心に最も近い語（cosが大）を上位N件返す"""
    sims = X_unit @ centroids_unit.T  # (V, k)
    out: Dict[int, List[Tuple[str, float]]] = {}
    for c in range(sims.shape[1]):
        idx = np.argsort(-sims[:, c])[:top_n]
        out[c] = [(vocab[i], float(sims[i, c])) for i in idx]
    return out

# ----------------------------
# spherical k-means（cos距離）
# ----------------------------
@dataclass
class SKMeansResult:
    labels_: np.ndarray           # (n_samples,)
    centroids_: np.ndarray        # (k, d) unit
    inertia_: float               # Σ(1 - cos)

def _kmeanspp_cos(X_unit: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """cos距離版k-means++初期化。返り値は (k, d) 単位ベクトル"""
    n, d = X_unit.shape
    centers = np.empty((k, d), dtype=X_unit.dtype)
    # 1つ目
    i0 = int(rng.integers(0, n))
    centers[0] = X_unit[i0]
    # 残り
    closest = 1.0 - X_unit @ centers[0].T  # 距離
    for c in range(1, k):
        probs = closest**2
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
    """X_unit は行L2=1"""
    if rng is None:
        rng = default_rng(RANDOM_SEED)
    best: Optional[SKMeansResult] = None
    n, d = X_unit.shape

    for _ in range(n_init):
        centers = _kmeanspp_cos(X_unit, k, rng)
        labels = np.zeros(n, dtype=np.int32)

        for it in range(max_iter):
            sims = X_unit @ centers.T  # (n,k)
            new_labels = np.argmax(sims, axis=1)
            if it > 0 and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # 重心更新（平均→正規化）
            for c in range(k):
                idx = np.where(labels == c)[0]
                if len(idx) == 0:
                    centers[c] = X_unit[int(rng.integers(0, n))]
                else:
                    mean = X_unit[idx].mean(axis=0)
                    norm = np.linalg.norm(mean)
                    centers[c] = X_unit[int(rng.integers(0, n))] if norm == 0.0 else (mean / norm)

        inertia = cosine_inertia(X_unit, labels, centers)
        res = SKMeansResult(labels, centers, inertia)
        if (best is None) or (res.inertia_ < best.inertia_):
            best = res
    assert best is not None
    return best

# ----------------------------
# 安定性（1回の50/50分割・上位語Jaccard）
# ----------------------------
def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))

def stability_top_terms_jaccard(
    per_doc_freqs: List[Dict[str, float]],
    *,
    top_n: int,
    min_docs: int,
    k: int,
    svd_dim: int,
    rng: np.random.Generator
) -> float:
    """非対称PPMIのみ対象。文書を半分に分け、各々でPPMI→SVD→クラスタ→上位語20語。Jaccardを平均。"""
    n_docs = len(per_doc_freqs)
    perm = rng.permutation(n_docs)
    A = sorted(perm[: n_docs // 2].tolist())
    B = sorted(perm[n_docs // 2 :].tolist())

    def build_word_space(doc_idx: List[int]) -> Tuple[List[str], np.ndarray]:
        sub = [per_doc_freqs[i] for i in doc_idx]
        out = compute_ppmi_from_perdocfreqs(sub, top_n=top_n, min_docs=min_docs)
        X_wd = out.ppmi_word_doc  # (V, D)
        svd = TruncatedSVD(n_components=min(svd_dim, max(2, min(X_wd.shape) - 1)), random_state=RANDOM_SEED)
        Z = svd.fit_transform(X_wd)
        return out.vocab, l2_normalize_rows(Z)

    vocabA, ZA = build_word_space(A)
    vocabB, ZB = build_word_space(B)

    resA = spherical_kmeans(ZA, k=k, n_init=10, max_iter=MAX_ITER, rng=rng)
    resB = spherical_kmeans(ZB, k=k, n_init=10, max_iter=MAX_ITER, rng=rng)

    topA = top_terms_by_centroid(ZA, vocabA, resA.centroids_, TOP_WORDS_PER_CLUSTER)
    topB = top_terms_by_centroid(ZB, vocabB, resB.centroids_, TOP_WORDS_PER_CLUSTER)

    usedB: set[int] = set()
    scores: List[float] = []
    for ca in range(k):
        cand = [
            (cb, jaccard([w for w, _ in topA[ca]], [w for w, _ in topB[cb]]))
            for cb in range(k) if cb not in usedB
        ]
        if not cand:
            continue
        cb, sc = max(cand, key=lambda t: t[1])
        usedB.add(cb)
        scores.append(sc)
    if not scores:
        return 0.0
    return float(np.mean(scores))

# ----------------------------
# クラスタ比率（要旨×k）
# ----------------------------
def abstract_cluster_ratio(
    per_doc_freqs: List[Dict[str, float]],
    vocab: List[str],
    labels: np.ndarray
) -> np.ndarray:
    """
    戻り値 shape = (D, k)。
    分母には OOV（未割当）も含める。OOV は分子に入れない。
    """
    word2idx = {w: i for i, w in enumerate(vocab)}
    k = int(labels.max()) + 1
    D = len(per_doc_freqs)
    M = np.zeros((D, k), dtype=np.float64)
    for d, freq in enumerate(per_doc_freqs):
        denom = 0.0
        for w, p in freq.items():
            denom += p  # 分母は常に加算
            j = word2idx.get(w)
            if j is not None:
                M[d, labels[j]] += p  # OOV は分子に入れない
        if denom > 0:
            M[d] /= denom
    return M

# ----------------------------
# 実行本体
# ----------------------------
def main() -> None:
    rng = default_rng(RANDOM_SEED)

    cols = default_columns()
    policy = default_token_policy()

    df = load_df(str(CSV_PATH), cols)
    texts: List[str] = df["abstract"].fillna("").astype(str).tolist()

    backend = SpacyBackend(model="en_core_web_sm")
    normalizer = build_normalizer(policy)

    # per_doc_freqs（キャッシュ利用）
    _, per_doc_freqs = analyze_with_cache(
        backend, normalizer, texts, policy, cache_dir=str(CACHE_DIR)
    )

    # PPMI（per_doc_freqs → 直接計算）
    out = compute_ppmi_from_perdocfreqs(per_doc_freqs, top_n=TOP_N, min_docs=MIN_DOCS)
    vocab = out.vocab

    # --- 非対称：SVD→L2 ---
    X_wd = out.ppmi_word_doc        # (V, D) sparse
    svd = TruncatedSVD(
        n_components=min(SVD_DIM, max(2, min(X_wd.shape) - 1)),
        random_state=RANDOM_SEED
    )
    Z_asym = svd.fit_transform(X_wd)     # (V, r)
    Z_asym = l2_normalize_rows(Z_asym)   # dense (V, r)

    # --- 対称：L2 ---
    X_ww = out.ppmi_word_word       # (V, V) sparse
    Z_sym = l2_normalize_rows(X_ww)      # dense (V, V)

    OUT_BASE = OUTPUTS_DIR / "clusters"
    (OUT_BASE / "asym").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / "sym").mkdir(parents=True, exist_ok=True)

    metrics = {"asym": {}, "sym": {}}

    # --- それぞれについて k を走らせる ---
    for space_name, Z in [("asym", Z_asym), ("sym", Z_sym)]:
        for k in K_LIST:
            res = spherical_kmeans(Z, k=k, n_init=N_INIT, max_iter=MAX_ITER, rng=rng)
            labels = res.labels_.astype(int)

            # silhouette（cosine）
            try:
                sil = float(silhouette_score(Z, labels, metric="cosine"))
            except Exception:
                sil = float("nan")

            # 安定性（非対称のみ）
            stab = None
            stab = stability_top_terms_jaccard(
                per_doc_freqs,
                top_n=TOP_N,
                min_docs=MIN_DOCS,
                k=k,
                svd_dim=SVD_DIM,
                rng=rng
            )

            metrics[space_name][str(k)] = {
                "silhouette_cosine": sil,
                "stability_jaccard_top20": stab,
                "inertia": float(res.inertia_)
            }

            # 上位語（解釈用）
            tops = top_terms_by_centroid(Z, vocab, res.centroids_, TOP_WORDS_PER_CLUSTER)

            # 保存
            out_dir = OUT_BASE / space_name
            # クラスタ割当
            with open(out_dir / f"labels_k{k}.csv", "w", encoding="utf-8") as f:
                print("word,cluster", file=f)
                for i, w in enumerate(vocab):
                    print(f"{w},{labels[i]}", file=f)
            # クラスタ上位語
            with open(out_dir / f"top_terms_k{k}.csv", "w", encoding="utf-8") as f:
                print("cluster,rank,word,cos_to_centroid", file=f)
                for c, pairs in tops.items():
                    for r, (w, s) in enumerate(pairs, start=1):
                        print(f"{c},{r},{w},{s:.6f}", file=f)

            # 要旨×k 比率（分母に OOV を含める）
            M = abstract_cluster_ratio(per_doc_freqs, vocab, labels)
            np.save(out_dir / f"abstract_ratio_k{k}.npy", M)

    with open(OUT_BASE / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== clustering done ===")
    print(f"- vocab size: {len(vocab)}")
    print(f"- saved to  : {OUT_BASE}")

if __name__ == "__main__":
    main()
