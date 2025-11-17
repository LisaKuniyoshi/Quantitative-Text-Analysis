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
from quant_text_analysis.grouping import method_group
from quant_text_analysis.io.loader import load_df
from quant_text_analysis.mnlr import (
    NO_CODE_LABEL,
    build_observation_frame,
    fit_binary_logit,
    fit_binary_logit_for_codes,
    fit_mnlogit,
    invert_code_map,
)
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


def run_cluster_analysis(
    out_dir: Path,
    per_doc_tokens: Sequence[Sequence[str]],
    years: Series,
    methods: Series,
) -> None:
    """クラスタコードを対象にMNLogitと二項ロジットを実施する。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    code_index: Dict[str, str] = invert_code_map(CODE_MAP_CLUSTER)
    df_obs_all: pd.DataFrame = build_observation_frame(
        per_doc_tokens,
        years,
        methods,
        code_index,
    )

    summarize_token_counts(df_obs_all, out_dir, label="cluster")

    df_obs: pd.DataFrame = df_obs_all[df_obs_all["code"] != NO_CODE_LABEL].reset_index(
        drop=True
    )
    if df_obs.empty:
        raise RuntimeError(
            "no_code 以外の観測がありません。CODE_MAP_CLUSTER を確認してください。"
        )

    rob, cats, _ = fit_mnlogit(df_obs)

    method_cols: List[str] = [col for col in METHOD_COLUMNS if df_obs[col].any()]
    pw: pd.DataFrame | None = None
    # if len(method_cols) >= 2:
    #     pw = pairwise_ame_multihot(
    #         rob,
    #         columns=method_cols,
    #         at="overall",
    #         alpha=0.05,
    #         mt_method="fdr_bh",
    #     )

    category_map: Dict[str, str] = {f"code_num={j}": cat for j, cat in enumerate(cats)}

    (out_dir / "mnlogit_summary.txt").write_text(
        rob.summary().as_text(),
        encoding="Shift_JIS",
    )

    def _map_category(value: Any) -> Any:
        if value in category_map:
            return category_map[value]
        value_str = str(value)
        return category_map.get(value_str, value)

    margeff = rob.get_margeff()
    margeff_frame = margeff.summary_frame()

    object_columns = margeff_frame.select_dtypes(include="object").columns
    for column_name in object_columns:
        margeff_frame[column_name] = margeff_frame[column_name].apply(_map_category)

    new_index = [
        tuple(_map_category(part) for part in idx_tuple)
        for idx_tuple in margeff_frame.index.tolist()
    ]
    margeff_frame.index = pd.MultiIndex.from_tuples(
        new_index, names=margeff_frame.index.names
    )

    margeff_frame.to_csv(out_dir / "margeff.csv", index=True, encoding="Shift_JIS")

    if pw is not None:
        pw.to_csv(out_dir / "pairwise_ame_mnlogit.csv", index=False, encoding="Shift_JIS")

    print("[Cluster] MNLogit summary table")
    print(rob.summary().tables[0])
    marg_summary = margeff.summary()
    if hasattr(marg_summary, "as_text"):
        summary_text: str = marg_summary.as_text()
        for code, label in category_map.items():
            summary_text = summary_text.replace(code, label)
        print(summary_text)
    else:
        print(marg_summary)

    if pw is not None:
        for eq, df_eq in pw.groupby("eq"):
            eq_str: str = str(eq)
            df_eq_typed: pd.DataFrame = df_eq.drop(columns=["eq"])
            print(f"[Cluster eq={eq_str}]")
            print(df_eq_typed.to_string(index=False))

    bin_res, _ = fit_binary_logit(df_obs_all, no_code_label=NO_CODE_LABEL)

    bin_summary = bin_res.summary()
    summary_str = bin_summary.as_text()
    summary_tables = bin_summary.tables

    print("[Cluster] Binary logit summary table")
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

    bin_res, _ = fit_binary_logit(df_obs_all, no_code_label=NO_CODE_LABEL)

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

    female_codes: tuple[str, ...] = ("【女性】",)
    male_codes: tuple[str, ...] = ("【男性】",)

    fem_res, _ = fit_binary_logit_for_codes(
        df_obs_all,
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
