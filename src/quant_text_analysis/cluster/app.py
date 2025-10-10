# quant_text_analysis/cluster/app.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.random import default_rng
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from ..settings import Settings
from ..io_loader import load_df
from ..nlp_backend import SpacyBackend
from ..normalize import build_normalizer
from ..cache import analyze_with_cache
from ..ppmi import compute_ppmi_from_perdocfreqs
from .algorithms import l2_normalize_rows, spherical_kmeans
from .metrics import (
    top_terms_by_centroid,
    stability_top_terms_jaccard,
    abstract_cluster_ratio,
)
from .io import (
    save_vocab, save_ppmi, save_top_terms, save_labels, save_metrics, save_cluster_ratio
)

def main() -> None:
    s = Settings()
    rng = default_rng(s.random_seed)

    # 入力読込
    df = load_df(str(s.csv_path), s.columns)
    texts: List[str] = df["abstract"].fillna("").astype(str).tolist()

    # 形態素＋正規化（キャッシュ利用）
    backend = SpacyBackend(model=s.spacy_model)
    normalizer = build_normalizer(s.token_policy)
    _, per_doc_freqs = analyze_with_cache(
        backend, normalizer, texts, s.token_policy, cache_dir=str(s.cache_dir)
    )

    # PPMI（非対称・対称）と語彙
    ppmi_out = compute_ppmi_from_perdocfreqs(
        per_doc_freqs, top_n=s.top_n, min_docs=s.min_docs
    )
    out_dir = s.ensure_out_dir()
    save_vocab(out_dir, ppmi_out.vocab)
    save_ppmi(out_dir, ppmi_out.ppmi_word_doc, ppmi_out.ppmi_word_word)

    # 語埋め込み（非対称PPMI→SVD→L2 正規化）
    X_wd = ppmi_out.ppmi_word_doc  # (V, D)
    svd = TruncatedSVD(
        n_components=min(s.svd_dim, max(2, min(X_wd.shape) - 1)),
        random_state=s.random_seed,
    )
    Z = svd.fit_transform(X_wd)          # (V, z)
    Z = l2_normalize_rows(Z)             # (V, z) 行 L2=1

    # k ごとにクラスタリング
    for k in s.k_list:
        res = spherical_kmeans(Z, k=k, n_init=s.n_init, max_iter=s.max_iter, rng=rng)
        labels = res.labels_.astype(int)

        # 上位語・ラベル保存
        top = top_terms_by_centroid(Z, ppmi_out.vocab, res.centroids_, s.top_words_per_cluster)
        save_top_terms(out_dir, k, top)
        save_labels(out_dir, k, ppmi_out.vocab, labels)

        # 指標
        try:
            sil = float(silhouette_score(Z, labels, metric="cosine"))
        except Exception:
            sil = float("nan")

        stab = stability_top_terms_jaccard(
            per_doc_freqs,
            top_n=s.top_n,
            min_docs=s.min_docs,
            k=k,
            svd_dim=s.svd_dim,
            rng=rng,
            top_words_per_cluster=s.top_words_per_cluster,
            max_iter=s.max_iter,
        )
        save_metrics(out_dir, k, inertia=res.inertia_, silhouette=sil, stability_jaccard=stab)

        # 文書×クラスタ比率
        M = abstract_cluster_ratio(per_doc_freqs, ppmi_out.vocab, labels)
        save_cluster_ratio(out_dir, k, M)

    print("Clustering done.")
    print(f"- vocab size: {len(ppmi_out.vocab)}")
    print(f"- docs      : {len(ppmi_out.doc_ids)}")
    print(f"- outputs   : {out_dir}")
