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

from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from pandas import Series
from statsmodels.discrete.discrete_model import MultinomialResultsWrapper

from quant_text_analysis.grouping import method_group
from quant_text_analysis.io.loader import load_df
from quant_text_analysis.mnlr import (
    fit_mnlogit,
    invert_code_map,
    rows_from_tokens,
)
from quant_text_analysis.preprocess.nlp_backend import SpacyBackend
from quant_text_analysis.preprocess.normalize import build_normalizer
from quant_text_analysis.preprocess.perdoc import analyze_docs_with_cache
from quant_text_analysis.settings import Settings

from quant_text_analysis.config import CODE_MAP, MNLR_EXCLUDE_METHODS
from quant_text_analysis.mnlr.statsmodels_fork import pairwise_ame_mnlogit


def main() -> None:
    """MNLR 推定から周辺効果の比較までを一括実行します。

    Returns:
        None
    """
    cfg: Settings = Settings()

    # 1) CSV読み込み + トークン化（キャッシュ利用）
    df: pd.DataFrame = load_df(str(cfg.csv_path), cfg.columns)
    texts: List[str] = df["abstract"].fillna("").astype(str).tolist()
    years: Series = pd.to_numeric(df["year"], errors="coerce")
    methods: Series = df["manual_tags"].apply(method_group)

    backend: SpacyBackend = SpacyBackend(cfg.spacy_model)
    normalizer: Any = build_normalizer(cfg.token_policy)
    per_doc_tokens: List[List[str]] = analyze_docs_with_cache(
        backend,
        normalizer,
        texts,
        cfg.token_policy,
        cache_dir=str(cfg.cache_dir),
    )

    # 2) コーディング→観測展開
    exclude_methods: set[str] = set(MNLR_EXCLUDE_METHODS)

    def _should_keep(tags: Sequence[str]) -> bool:
        return not any(tag in exclude_methods for tag in tags)

    keep_mask: Series = methods.apply(_should_keep)
    if not keep_mask.any():
        raise RuntimeError(
            "除外設定後にMNLogitへ投入できる文書がありません。"
            "config.MNLR_EXCLUDE_METHODS を見直してください。"
        )

    keep_flags: List[bool] = keep_mask.to_list()
    per_doc_tokens = [
        tokens for tokens, keep in zip(per_doc_tokens, keep_flags) if keep
    ]
    methods = methods[keep_mask].reset_index(drop=True)
    years = years[keep_mask].reset_index(drop=True)

    code_index: Dict[str, str] = invert_code_map(CODE_MAP)
    df_obs: pd.DataFrame = rows_from_tokens(per_doc_tokens, years, methods, code_index)
    if df_obs.empty:
        raise RuntimeError(
            "コーディング後の観測が0件です。"
            "mlra_config.CODE_MAP を確認してください。"
        )

    # 3) 設計→適合（文書クラスタに対するロバストSE）
    rob: MultinomialResultsWrapper
    res: MultinomialResultsWrapper
    cats: List[str]
    df_for_pred: pd.DataFrame
    rob, cats, df_for_pred = fit_mnlogit(df_obs)

    # pw: pd.DataFrame = pairwise_ame_mnlogit(
    #     rob,
    #     factor="method",
    #     base="qual",
    #     at="overall",
    #     alpha=0.05,
    #     mt_method="fdr_bh",
    # )

    # 4) 出力（推定結果）
    out_dir: Path = cfg.ensure_out_dir()

    # 4-1) rob のサマリ全文（テキスト）
    (out_dir / "mnlogit_summary.txt").write_text(
        rob.summary().as_text(), encoding="utf-8"
    )

    # 4-2) AME（overall, dydx）のうち year_centered のみ（CSV）
    rob.get_margeff(dummy=True).summary_frame().to_csv(out_dir / "margeff.csv", index=True, encoding="utf-8")

    # # 4-2) AME（overall, dydx）のうち year_centered のみ（CSV）
    # dm = rob.get_margeff(at="overall", method="dydx")
    # sf_year = dm.summary_frame().xs("year_centered", level="exog")
    # sf_year.reset_index().to_csv(out_dir / "margeff_year_centered.csv", index=False, encoding="utf-8")

    # # 4-3) pairwise_ame_mnlogit の結果（CSV）
    # pw.to_csv(out_dir / "pairwise_ame_mnlogit.csv", index=False, encoding="utf-8")

    # 4-4) コンソール出力（要約）
    print("category map:", {f"y={j}": cat for j, cat in enumerate(cats)})
    print(rob.summary())
    print(rob.get_margeff().summary())
    # print(rob.summary().tables[0])
    # print(
    #     rob.get_margeff(dummy=True)
    #     .summary_frame()
    #     .xs("year_centered", level="exog", drop_level=False)
    # )

    # for eq, df_eq in pw.groupby("eq"):
    #     eq_str: str = str(eq)
    #     df_eq_typed: pd.DataFrame = df_eq.drop(columns=["eq"])
    #     print(f"[eq={eq_str}]")
    #     print(df_eq_typed.to_string(index=False))


    # # 5) 観測ごとの予測選択確率（モデルに基づく）
    # prob = predict_probabilities(res, df_for_pred, cats)

    # # 6) 年と手法で二次当てはめ+95%CIの図を作成
    # plot_prob_by_year_with_method(
    #     prob,
    #     df_for_pred["year"],
    #     df_for_pred["method"],
    #     out_dir,
    # )


if __name__ == "__main__":
    main()
