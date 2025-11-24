"""文書単位クラスタ・ロバストSE付き二項ロジットを実行する。

手順:
    (1) CSV読み込み (io.loader.load_df)
    (2) トークン化 (preprocess.perdoc.analyze_docs_with_cache)
    (3) コーディング展開（1トークン=1観測）
    (4) コード別に二項ロジットを推定し、文書クラスタでロバストSEを付加
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import numpy as np
from pandas import Series
from quant_text_analysis.grouping import method_group
from quant_text_analysis.io.loader import load_df
from quant_text_analysis.mnlr import (
    NO_CODE_LABEL,
    build_observation_frame,
    fit_binary_logit,
    fit_binary_logit_for_codes,
    invert_code_map,
    plot_binary_year_effect_prediction,
    plot_method_odds_ratios,
    plot_year_odds_ratios,
)
from quant_text_analysis.mnlr.model import fit_binary_logit_for_code
from quant_text_analysis.mnlr.coding import METHOD_COLUMNS
from quant_text_analysis.preprocess.nlp_backend import SpacyBackend
from quant_text_analysis.preprocess.normalize import build_normalizer
from quant_text_analysis.preprocess.perdoc import analyze_docs_with_cache
from quant_text_analysis.settings import Settings

from quant_text_analysis.config import (
    CODE_MAP_CLUSTER,
    CODE_MAP_GENDER,
    MNLR_EXCLUDE_METHODS,
)


EXCLUDED_CLUSTER_CODES: set[str] = {"【手段】"}
EXCLUDED_CLUSTER_PLOT_CODES: set[str] = {"【代替戦略】"}


def summarize_token_counts(
    df_obs_all: pd.DataFrame,
    out_dir: Path,
    label: str,
) -> None:
    """トークン件数をCSVと標準出力へ出力する。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    total_tokens: int = len(df_obs_all)
    counts = (
        df_obs_all["code"]
        .value_counts(dropna=False)
        .rename_axis("code")
        .reset_index(name="token_count")
        .sort_values("code")
        .reset_index(drop=True)
    )

    counts_with_total = pd.concat(
        [
            counts,
            pd.DataFrame({"code": ["TOTAL"], "token_count": [total_tokens]}),
        ],
        ignore_index=True,
    )
    counts_with_total.to_csv(out_dir / "token_counts.csv", index=False, encoding="Shift_JIS")

    print(f"[Token Counts:{label}] total_tokens={total_tokens}")
    for _, row in counts.iterrows():
        code_label = str(row["code"])
        count_val = int(row["token_count"])
        print(f"code={code_label}: {count_val}")
    print(f"TOTAL: {total_tokens}")


def _exp_odds_ratio_columns(df: pd.DataFrame) -> None:
    """Convert coefficient-style columns into odds ratios in-place."""

    target_labels = {"coef", "0.025", "0.975"}
    for column in df.columns:
        normalized = column.strip().strip("[]")
        if normalized in target_labels:
            numeric_values = pd.to_numeric(df[column], errors="coerce")
            df[column] = np.exp(numeric_values)


def run_cluster_analysis(
    out_dir: Path,
    per_doc_tokens: Sequence[Sequence[str]],
    years: Series,
    methods: Series,
) -> None:
    """クラスタコードを対象にコードごとの二項ロジットを実施する。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    code_index: Dict[str, str] = invert_code_map(CODE_MAP_CLUSTER)
    df_obs_all: pd.DataFrame = build_observation_frame(
        per_doc_tokens,
        years,
        methods,
        code_index,
    )

    summarize_token_counts(df_obs_all, out_dir, label="cluster")

    codes_in_data = (
        df_obs_all["code"].loc[df_obs_all["code"] != NO_CODE_LABEL].dropna().unique()
    )
    if len(codes_in_data) == 0:
        raise RuntimeError(
            "no_code 以外の観測がありません。CODE_MAP_CLUSTER を確認してください。"
        )

    sorted_codes: List[str] = sorted(
        str(code) for code in codes_in_data if str(code) not in EXCLUDED_CLUSTER_CODES
    )
    if not sorted_codes:
        raise RuntimeError(
            "除外コードを考慮すると推定対象のクラスタコードが存在しません。"
        )
    combined_results: List[pd.DataFrame] = []

    def _safe_filename(label: str) -> str:
        forbidden = '<>:"/\\|?*'
        sanitized = "".join("_" if ch in forbidden else ch for ch in label)
        return sanitized or "code"

    for code_label in sorted_codes:
        res, _ = fit_binary_logit_for_code(df_obs_all, code_label=code_label)

        summary_obj = res.summary()
        print(f"[Cluster] Binary logit summary for code={code_label}")
        print(summary_obj)

        summary_tables = getattr(summary_obj, "tables", None) or []
        summary_path = out_dir / f"binary_logit_summary_{_safe_filename(code_label)}.txt"
        if summary_tables:
            summary_text = summary_tables[0].as_text()
        else:
            summary_text = str(summary_obj)
        summary_path.write_text(summary_text, encoding="Shift_JIS")

        coeff_df = pd.DataFrame(
            {
                "coef": res.params,
                "std_err": res.bse,
                "z": res.tvalues,
                "p_value": res.pvalues,
            }
        )
        conf_int = res.conf_int(alpha=0.05)
        conf_int.columns = ["0.025", "0.975"]
        coeff_df = coeff_df.join(conf_int)
        coeff_df = coeff_df.reset_index().rename(columns={"index": "exog"})
        coeff_df.insert(0, "endog", code_label)
        coeff_df = coeff_df[
            [
                "endog",
                "exog",
                "coef",
                "std_err",
                "z",
                "p_value",
                "0.025",
                "0.975",
            ]
        ]
        _exp_odds_ratio_columns(coeff_df)
        combined_results.append(coeff_df)

    if combined_results:
        merged = pd.concat(combined_results, ignore_index=True)
        merged.to_csv(out_dir / "margeff.csv", index=False, encoding="Shift_JIS")

        method_subset = merged[merged["exog"].isin(METHOD_COLUMNS)].copy()
        if not method_subset.empty:
            method_subset = method_subset[
                ~method_subset["endog"].isin(EXCLUDED_CLUSTER_PLOT_CODES)
            ]
            if not method_subset.empty:
                plot_method_odds_ratios(
                    method_subset,
                    METHOD_COLUMNS,
                    out_dir / "odds_ratio_methods.png",
                )

        year_subset = merged[merged["exog"] == "year_centered"].copy()
        if not year_subset.empty:
            year_subset = year_subset[
                ~year_subset["endog"].isin(EXCLUDED_CLUSTER_PLOT_CODES)
            ]
            if not year_subset.empty:
                plot_year_odds_ratios(year_subset, out_dir / "odds_ratio_year.png")
    else:
        empty_cols: List[str] = [
            "endog",
            "exog",
            "coef",
            "std_err",
            "z",
            "p_value",
            "0.025",
            "0.975",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(
            out_dir / "margeff.csv",
            index=False,
            encoding="Shift_JIS",
        )


def run_gender_analysis(
    out_dir: Path,
    per_doc_tokens: Sequence[Sequence[str]],
    years: Series,
    methods: Series,
) -> None:
    """ジェンダーコードを対象に二項ロジットを実施する。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    code_index: Dict[str, str] = invert_code_map(CODE_MAP_GENDER)
    df_obs_all: pd.DataFrame = build_observation_frame(
        per_doc_tokens,
        years,
        methods,
        code_index,
    )

    summarize_token_counts(df_obs_all, out_dir, label="gender")

    bin_res, df_bin_design = fit_binary_logit(df_obs_all, no_code_label=NO_CODE_LABEL)

    bin_summary = bin_res.summary()
    summary_str = bin_summary.as_text()
    summary_tables = bin_summary.tables

    print("[Gender] Binary logit summary table")
    if summary_tables:
        print(summary_tables[0])
        (out_dir / "binlogit_summary.txt").write_text(
            summary_tables[0].as_text(),
            encoding="Shift_JIS",
        )
    else:
        print(summary_str)
        (out_dir / "binlogit_summary.txt").write_text(
            summary_str,
            encoding="Shift_JIS",
        )

    if len(summary_tables) >= 2:
        table1 = summary_tables[1]
        table1_frame = pd.DataFrame(table1.data[1:], columns=table1.data[0])
        _exp_odds_ratio_columns(table1_frame)
        table1_frame.to_csv(
            out_dir / "binlogit_results.csv",
            index=False,
            encoding="Shift_JIS",
        )
    else:
        pd.DataFrame({"summary": [summary_str]}).to_csv(
            out_dir / "binlogit_results.csv",
            index=False,
            encoding="Shift_JIS",
        )

    coeff_df = pd.DataFrame(
        {
            "coef": bin_res.params,
            "std_err": bin_res.bse,
            "z": bin_res.tvalues,
            "p_value": bin_res.pvalues,
        }
    )
    conf_int = bin_res.conf_int(alpha=0.05)
    conf_int.columns = ["0.025", "0.975"]
    coeff_df = coeff_df.join(conf_int)
    coeff_df = coeff_df.reset_index().rename(columns={"index": "exog"})
    coeff_df.insert(0, "endog", "code_selected")
    coeff_df = coeff_df[
        [
            "endog",
            "exog",
            "coef",
            "std_err",
            "z",
            "p_value",
            "0.025",
            "0.975",
        ]
    ]
    _exp_odds_ratio_columns(coeff_df)

    method_subset = coeff_df[coeff_df["exog"].isin(METHOD_COLUMNS)].copy()
    if not method_subset.empty:
        plot_method_odds_ratios(
            method_subset,
            METHOD_COLUMNS,
            out_dir / "odds_ratio_methods.png",
            show_titles=False,
        )

    plot_binary_year_effect_prediction(
        bin_res,
        df_bin_design,
        out_dir / "year_effect_prediction.png",
        ylabel="コード選択確率",
        # title="Gender code selection vs year",
    )

    female_codes: tuple[str, ...] = ("【女性】",)
    male_codes: tuple[str, ...] = ("【男性】",)

    theoretic_doc_ids = set(
        df_obs_all.loc[df_obs_all["theoretic"] == 1, "doc_id"].unique().tolist()
    )
    if theoretic_doc_ids:
        print(
            f"[Gender] Female vs Male logit excludes {len(theoretic_doc_ids)} theoretic-docs"
        )
    df_obs_no_theoretic = df_obs_all[
        ~df_obs_all["doc_id"].isin(theoretic_doc_ids)
    ].copy()
    if df_obs_no_theoretic.empty:
        raise RuntimeError(
            "female_vs_male logit用の文書が理論的研究除外後に残っていません。"
        )

    fem_res, df_fem_design = fit_binary_logit_for_codes(
        df_obs_no_theoretic,
        positive_codes=female_codes,
        negative_codes=male_codes,
        target_column="is_female",
    )

    fem_summary = fem_res.summary()
    summary_str = fem_summary.as_text()
    summary_tables = fem_summary.tables

    print("[Gender] Female vs Male logit summary table")
    if summary_tables:
        print(summary_tables[0])
        (out_dir / "female_vs_male_summary.txt").write_text(
            summary_tables[0].as_text(),
            encoding="Shift_JIS",
        )
    else:
        print(summary_str)
        (out_dir / "female_vs_male_summary.txt").write_text(
            summary_str,
            encoding="Shift_JIS",
        )

    if len(summary_tables) >= 2:
        table1 = summary_tables[1]
        table1_frame = pd.DataFrame(table1.data[1:], columns=table1.data[0])
        _exp_odds_ratio_columns(table1_frame)
        table1_frame.to_csv(
            out_dir / "female_vs_male_results.csv",
            index=False,
            encoding="Shift_JIS",
        )
    else:
        pd.DataFrame({"summary": [summary_str]}).to_csv(
            out_dir / "female_vs_male_results.csv",
            index=False,
            encoding="Shift_JIS",
        )

    fem_coeff_df = pd.DataFrame(
        {
            "coef": fem_res.params,
            "std_err": fem_res.bse,
            "z": fem_res.tvalues,
            "p_value": fem_res.pvalues,
        }
    )
    fem_conf_int = fem_res.conf_int(alpha=0.05)
    fem_conf_int.columns = ["0.025", "0.975"]
    fem_coeff_df = fem_coeff_df.join(fem_conf_int)
    fem_coeff_df = fem_coeff_df.reset_index().rename(columns={"index": "exog"})
    fem_coeff_df.insert(0, "endog", "is_female")
    fem_coeff_df = fem_coeff_df[
        [
            "endog",
            "exog",
            "coef",
            "std_err",
            "z",
            "p_value",
            "0.025",
            "0.975",
        ]
    ]
    _exp_odds_ratio_columns(fem_coeff_df)

    fem_method_subset = fem_coeff_df[fem_coeff_df["exog"].isin(METHOD_COLUMNS)].copy()
    if not fem_method_subset.empty:
        plot_method_odds_ratios(
            fem_method_subset,
            METHOD_COLUMNS,
            out_dir / "female_vs_male_odds_ratio_methods.png",
            show_titles=False,
        )

    plot_binary_year_effect_prediction(
        fem_res,
        df_fem_design,
        out_dir / "female_vs_male_year_effect_prediction.png",
        ylabel="女性コード選択確率",
        y_limits=(0.0, 1.0),
    )


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

    out_dir: Path = cfg.ensure_out_dir()

    run_cluster_analysis(out_dir / "cluster", per_doc_tokens, years, methods)
    run_gender_analysis(out_dir / "gender", per_doc_tokens, years, methods)


if __name__ == "__main__":
    main()
