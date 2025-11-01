"""code × method のクロス集計（文書単位の出現有無）。

既存ロジックを再利用し、「文書でそのコードが一度でも出たか」を基準に
code×method のクロス表を作成する。

手順:
    (1) CSVを読み込む (io.loader.load_df)
    (2) 文書をトークン化する (preprocess.perdoc.analyze_docs_with_cache)
    (3) コーディング規則 (CODE_MAP) に基づき、文書ごとのコード出現有無を作る
    (4) method_group で手法カテゴリ化し、code×method のクロス集計を作る
    (5) CSVに保存し、標準出力に要約を出す

出力:
    - {out_dir}/code_method_crosstab_docs.csv … 文書単位の出現有無ベースのクロス表
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
    build_code_method_crosstab,
    codes_per_doc,
    invert_code_map,
)

# 既存のコーディング規則を再利用
from ..config import CODE_MAP


def main() -> None:
    """エントリポイント。クロス集計を作成・保存する。"""
    cfg = Settings()

    # (1) 入力の読み込みとトークン化
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

    # (3) 文書ごとのコード出現（有無）
    code_index = invert_code_map(CODE_MAP)
    per_doc_codes = codes_per_doc(per_doc_tokens, code_index)

    # 手法カテゴリ（欠損は "other"）
    method = df["manual_tags"].apply(method_group).fillna("other").astype(str)

    # 文書単位のクロス集計（出現文書数）
    # 行=コード、列=手法、値=その手法の文書数のうちコードが出た文書数
    ctab = build_code_method_crosstab(
        per_doc_codes,
        method,
        code_order=sorted(CODE_MAP.keys()),
        include_methods=["other"],
    )

    # (4) 出力
    out_dir = cfg.ensure_out_dir()
    out_path = out_dir / "code_method_crosstab_docs.csv"
    ctab.to_csv(out_path, encoding="utf-8")

    # 標準出力に要約（上位数行）を表示
    print(f"[saved] {out_path}")
    with pd.option_context("display.max_rows", 20, "display.max_columns", None):
        print(ctab.head(20))


if __name__ == "__main__":
    main()
