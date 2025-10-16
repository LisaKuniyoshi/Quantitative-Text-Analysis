"""Word clustering pipeline (PPMI → SVD → spherical k-means).

概要:
    既定の設定 (`Settings`) に基づき、
    要旨コーパスから語×文書 PPMI と語×語 PPMI を計算し、
    SVD による語埋め込みを L2 正規化して球面 k-means でクラスタリングします。
    語彙・PPMI 行列・クラスタ結果と各種メトリクスを出力します。

I/O:
    読み込み:
        - CSV: Settings.csv_path に指定された書誌 CSV（要旨・年・手作業タグ列）
    書き込み:
        - outputs/vocab.json
        - outputs/PPMI_word_doc/top_terms_k{K}.csv
        - outputs/PPMI_word_doc/labels_k{K}.csv
        - outputs/PPMI_word_doc/metrics_k{K}.json
        - outputs/PPMI_word_doc/abstract_ratio_k{K}.npy
        - outputs/PPMI_word_doc/metrics.csv
        - outputs/PPMI_word_word/top_terms_k{K}.csv
        - outputs/PPMI_word_word/labels_k{K}.csv
        - outputs/PPMI_word_word/metrics_k{K}.json
        - outputs/PPMI_word_word/abstract_ratio_k{K}.npy
        - outputs/PPMI_word_word/metrics.csv

設定:
    すべて `Settings` で指定します（パス、`k_list`, `svd_dim`, `random_seed` など）。
    形態素解析は spaCy（`spacy_model`）を使用し、文書ごとの頻度はキャッシュ可能です。

使用例:
    >>> python -m quant_text_analysis.cluster.app
    >>> python path/to/cluster_cli.py
"""
from __future__ import annotations
import csv
from typing import List

from numpy.random import default_rng
from sklearn.metrics import silhouette_score

from ..settings import Settings
from ..io.loader import load_df
from ..preprocess.nlp_backend import SpacyBackend
from ..preprocess.normalize import build_normalizer
from ..preprocess.perdoc import get_or_analyze_docs
from ..features.ppmi import get_or_compute_ppmi
from ..cluster.algorithms import l2_normalize_rows, spherical_kmeans
from ..cluster.metrics import (
    top_terms_by_centroid,
    stability_top_terms_jaccard,
    abstract_cluster_ratio,
)
from ..io.writers import (
    save_vocab, save_top_terms, save_labels, save_metrics, save_cluster_ratio
)
from ..features.embeddings import get_or_svd_embedding

def main() -> None:
    """クラスタリングパイプライン全体を実行します。

    注意事項:
        * 乱数は `Settings.random_seed` に従います。
        * キャッシュは文書頻度・PPMI・SVD の計算で利用されます。
        * 語彙やクラスタリング結果は標準出力とファイルに出力されます。
    """
    s = Settings()
    rng = default_rng(s.random_seed)

    # 入力読込
    df = load_df(str(s.csv_path), s.columns)
    texts: List[str] = df["abstract"].fillna("").astype(str).tolist()

    # 形態素＋正規化（キャッシュ利用）
    backend = SpacyBackend(model=s.spacy_model)
    normalizer = build_normalizer(s.token_policy)
    _, per_doc_freqs = get_or_analyze_docs(
        backend, normalizer, texts, s.token_policy, cache_dir=str(s.cache_dir)
    )

    # PPMI（非対称・対称）と語彙
    ppmi_out = get_or_compute_ppmi(per_doc_freqs)
    out_dir = s.ensure_out_dir()
    save_vocab(out_dir, ppmi_out.vocab)

    for name, ppmi in [("PPMI_word_doc", ppmi_out.ppmi_word_doc), ("PPMI_word_word", ppmi_out.ppmi_word_word)]:
        out_dir_cluster = out_dir / name
        out_dir_cluster.mkdir(parents=True, exist_ok=True)
        metrics_rows = []

        for d in s.svd_dim_list:
            # 語埋め込み（非対称PPMI→SVD→L2 正規化）
            Z = get_or_svd_embedding(
                ppmi,
                svd_dim=d,
                cfg=s,
                ppmi_cache_key=ppmi_out.cache_key,
                random_state=s.random_seed,
            )
            Z = l2_normalize_rows(Z)             # (V, z) 行 L2=1

            # k ごとにクラスタリング
            for k in s.k_list:
                res = spherical_kmeans(Z, k=k, n_init=s.n_init, max_iter=s.max_iter, rng=rng)
                labels = res.labels_.astype(int)

                # 上位語・ラベル保存
                top = top_terms_by_centroid(Z, ppmi_out.vocab, res.centroids_, s.top_words_per_cluster)
                save_top_terms(out_dir_cluster, k, top)
                save_labels(out_dir_cluster, k, ppmi_out.vocab, labels)

                # 指標
                try:
                    sil = float(silhouette_score(Z, labels, metric="cosine"))
                except Exception:
                    sil = float("nan")

                stab = stability_top_terms_jaccard(
                    per_doc_freqs,
                    k=k,
                    svd_dim=d,
                    rng=rng,
                    top_words_per_cluster=s.top_words_per_cluster,
                    max_iter=s.max_iter,
                )
                save_metrics(out_dir_cluster, k, inertia=res.inertia_, silhouette=sil, stability_jaccard=stab)
                metrics_rows.append({"dim": d, "k": int(k), "silhouette": sil, "stability_jaccard": stab})

                # 文書×クラスタ比率
                M = abstract_cluster_ratio(per_doc_freqs, ppmi_out.vocab, labels)
                save_cluster_ratio(out_dir_cluster, k, M)

        metrics_csv_path = out_dir_cluster / "metrics.csv"
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dim", "k", "silhouette", "stability_jaccard"])
            writer.writeheader()
            writer.writerows(metrics_rows)

    print("Clustering done.")
    print(f"- vocab size: {len(ppmi_out.vocab)}")
    print(f"- docs      : {len(ppmi_out.doc_ids)}")
    print(f"- outputs   : {out_dir}")
