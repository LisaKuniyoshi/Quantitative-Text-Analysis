"""Token frequency rankings for overall and predefined groups.

概要:
    `Settings` に基づいて CSV を読み込み、文書内相対頻度 r(d, w) を平均化して
    上位語を算出します。全体・年代・研究手法のランキングを表示し、必要に応じて
    CSV に保存します。

I/O:
    読み込み:
        - CSV: Settings.csv_path
    書き込み:
        - outputs/top_words_overall.csv
        - outputs/top_words_periods.csv
        - outputs/top_words_methods.csv

グルーピング:
    - 年代: "2014–2021" / "2022–2023" / "2024–2025"
    - 手法: "qual" / "quan" / "theoretic" / "review" / "other"

使用例:
    python -m quant_text_analysis.cli
    python path/to/freq_cli.py
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..settings import Settings
from ..io.loader import load_df
from ..grouping import (
    METHOD_CODE_TO_LABEL,
    period_group_year,
    method_group,
)
from ..preprocess.nlp_backend import SpacyBackend
from ..preprocess.normalize import build_normalizer
from ..features.frequency import frequency_rankings
from ..preprocess.perdoc import analyze_docs_with_cache, compute_term_frequencies

s: Settings = Settings()
cols = s.columns
policy = s.token_policy

# ---- 実行本体 ---------------------------------------------------------------
def main() -> None:
    """頻度ランキングを計算して表示し、必要に応じて保存します。

    Notes:
        - `Settings.out_dir` が設定されている場合に CSV を書き出す。
    """
    df: pd.DataFrame = load_df(str(s.csv_path), cols)
    df["period"] = df["year"].map(period_group_year)
    df["method"] = df["manual_tags"].map(method_group)

    backend = SpacyBackend(model=s.spacy_model)
    normalizer = build_normalizer(policy)

    abstracts: List[str] = df["abstract"].fillna("").tolist()
    tokenized_docs = analyze_docs_with_cache(
        backend, normalizer, abstracts, policy, cache_dir=str(s.cache_dir)
    )
    per_doc_freqs = compute_term_frequencies(tokenized_docs)

    overall = frequency_rankings(per_doc_freqs, None)
    period = frequency_rankings(per_doc_freqs, df["period"].tolist())
    method = frequency_rankings(per_doc_freqs, df["method"].tolist())

    method_label_map = {
        code: METHOD_CODE_TO_LABEL.get(code, code)
        for code in method.keys()
    }

    print(f"\n=== Overall (All documents) | Top {len(overall['ALL'])} ===")
    print(overall["ALL"].to_string(index=False))
    for g, dfr in period.items():
        print(f"\n=== Period: {g} | Top {len(dfr)} ===")
        print(dfr.to_string(index=False))
    for g, dfr in method.items():
        label = method_label_map.get(g, g)
        print(f"\n=== Method: {label} | Top {len(dfr)} ===")
        print(dfr.to_string(index=False))

    if s.out_dir is not None:
        s.out_dir.mkdir(parents=True, exist_ok=True)
        overall["ALL"].to_csv(s.out_dir / "top_words_overall.csv", index=False)
        def _concat_side_by_side(grouped: dict[str, pd.DataFrame]) -> pd.DataFrame:
            if not grouped:
                return pd.DataFrame()

            max_rows = max(len(frame) for frame in grouped.values())
            pieces: List[pd.DataFrame] = []
            for group_name, frame in grouped.items():
                expanded = frame.reset_index(drop=True).reindex(range(max_rows))
                renamed = expanded.rename(columns={
                    "word": f"{group_name}_word",
                    "mean_freq": f"{group_name}_mean_freq",
                    "n_docs_nonzero": f"{group_name}_n_docs_nonzero",
                    "doc_rate_nonzero": f"{group_name}_doc_rate_nonzero",
                })
                pieces.append(renamed)

            return pd.concat(pieces, axis=1)

        period_df = _concat_side_by_side(period)
        method_df = _concat_side_by_side({
            method_label_map.get(name, name): frame
            for name, frame in method.items()
        })
        period_df.to_csv(s.out_dir / "top_words_periods.csv", index=False)
        method_df.to_csv(s.out_dir / "top_words_methods.csv", index=False)

if __name__ == "__main__":
    main()
