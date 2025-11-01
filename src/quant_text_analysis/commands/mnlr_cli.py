"""文書単位クラスタ・ロバストSE付き多項ロジットを可視化付きで実行する。

手順:
    (1) CSV読み込み (io.loader.load_df)
    (2) トークン化 (preprocess.perdoc.analyze_docs_with_cache)
    (3) コーディング展開（1トークン=1観測）
    (4) MNLogit を推定し、文書クラスタでロバストSEを付加
    (5) 観測ごとの予測選択確率を計算
    (6) 年と手法で二次当てはめ+95%CIの図を作成

注:
    図の作り方は Stata の `twoway qfitci`（二次回帰の当てはめ＋CI）に準拠。
"""

from __future__ import annotations

import pandas as pd

from quant_text_analysis.settings import Settings
from quant_text_analysis.grouping import method_group
from quant_text_analysis.io.loader import load_df
from quant_text_analysis.preprocess.nlp_backend import SpacyBackend
from quant_text_analysis.preprocess.normalize import build_normalizer
from quant_text_analysis.preprocess.perdoc import analyze_docs_with_cache
from quant_text_analysis.mnlr import (
    build_design,
    fit_multinomial_cluster_robust,
    invert_code_map,
    plot_prob_by_year_with_method,
    predict_probabilities,
    rows_from_tokens,
)
from ..config import CODE_MAP, MNLR_EXCLUDE_METHODS


def main() -> None:
    """パイプライン本体を実行し、推定結果と可視化を出力する。

    Returns:
        None
    """
    cfg = Settings()

    # 1) CSV読み込み + トークン化（キャッシュ利用）
    df = load_df(str(cfg.csv_path), cfg.columns)
    texts = df["abstract"].fillna("").astype(str).tolist()
    backend = SpacyBackend(cfg.spacy_model)
    normalizer = build_normalizer(cfg.token_policy)
    per_doc_tokens = analyze_docs_with_cache(
        backend,
        normalizer,
        texts,
        cfg.token_policy,
        cache_dir=str(cfg.cache_dir),
    )

    # 2) コーディング→観測展開
    methods = df["manual_tags"].apply(method_group)
    years = df["year"]

    exclude_methods = set(MNLR_EXCLUDE_METHODS)
    keep_mask = ~methods.isin(exclude_methods)
    if not keep_mask.any():
        raise RuntimeError(
            "除外設定後にMNLogitへ投入できる文書がありません。config.MNLR_EXCLUDE_METHODS を見直してください。"
        )

    keep_flags = keep_mask.to_list()
    per_doc_tokens = [tokens for tokens, keep in zip(per_doc_tokens, keep_flags) if keep]
    methods = methods[keep_mask].reset_index(drop=True)
    years = years[keep_mask].reset_index(drop=True)

    code_index = invert_code_map(CODE_MAP)
    df_obs = rows_from_tokens(per_doc_tokens, years, methods, code_index)
    if df_obs.empty:
        raise RuntimeError("コーディング後の観測が0件です。mlra_config.CODE_MAP を確認してください。")

    # 3) 設計→適合（文書クラスタに対するロバストSE）
    design = build_design(df_obs)
    robust, res = fit_multinomial_cluster_robust(design)

    # 4) 出力（推定結果）
    out_dir = cfg.ensure_out_dir()
    params = pd.DataFrame(robust.params, index=design.X.columns)
    bse = pd.DataFrame(robust.bse, index=design.X.columns)
    params.to_csv(out_dir / "mlra_params.csv", encoding="utf-8", index=True)
    bse.to_csv(out_dir / "mlra_bse_cluster.csv", encoding="utf-8", index=True)
    with open(out_dir / "mlra_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(robust.summary()))
    print("category map:", res.model._ynames_map)
    print(robust.summary())

    # 5) 観測ごとの予測選択確率（モデルに基づく）
    prob_df = predict_probabilities(res, design.X, design.categories)

    # 6) 年×手法の二次当てはめ＋95%CIを描画（qfitci相当）
    plot_prob_by_year_with_method(prob_df, design.year, design.method, out_dir)


if __name__ == "__main__":
    main()
