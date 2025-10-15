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
        - outputs/top_words_period_{グループ名}.csv
        - outputs/top_words_method_{グループ名}.csv

グルーピング:
    - 年代: "2014–2021" / "2022–2023" / "2024–2025"
    - 手法: "qual" / "quan" / "theoretic" / "review" / "other"

使用例:
    python -m quant_text_analysis.cli
    python path/to/freq_cli.py
"""
from __future__ import annotations

from typing import List

from ..settings import Settings
from ..io.loader import load_df
from ..grouping import period_group_year, method_group
from ..preprocess.nlp_backend import SpacyBackend
from ..preprocess.normalize import build_normalizer
from ..features.frequency import frequency_rankings
from ..preprocess.perdoc import get_or_analyze_docs

s = Settings()
cols = s.columns
policy = s.token_policy

# ---- 実行本体 ---------------------------------------------------------------
def main() -> None:
    """頻度ランキングを計算して表示し、必要に応じて保存します。

    Notes:
        - 文書ごとの相対頻度を平均化し、`Settings.top_n` と `Settings.min_docs` でフィルタする。
        - `Settings.out_dir` が設定されている場合に CSV を書き出す。
    """
    df = load_df(str(s.csv_path), cols)
    df["period"] = df["year"].map(period_group_year)
    df["method"] = df["manual_tags"].map(method_group)

    backend = SpacyBackend(model=s.spacy_model)
    normalizer = build_normalizer(policy)

    abstracts: List[str] = df["abstract"].fillna("").tolist()
    _, per_doc_freqs = get_or_analyze_docs(
        backend, normalizer, abstracts, policy, cache_dir=str(s.cache_dir)
    )

    overall = frequency_rankings(per_doc_freqs, None, top_n=s.top_n, min_docs=s.min_docs)
    period = frequency_rankings(per_doc_freqs, df["period"].tolist(), top_n=s.top_n, min_docs=s.min_docs)
    method = frequency_rankings(per_doc_freqs, df["method"].tolist(), top_n=s.top_n, min_docs=s.min_docs)

    print(f"\n=== Overall (All documents) | Top {len(overall['ALL'])} ===")
    print(overall["ALL"].to_string(index=False))
    for g, dfr in period.items():
        print(f"\n=== Period: {g} | Top {len(dfr)} ===")
        print(dfr.to_string(index=False))
    for g, dfr in method.items():
        print(f"\n=== Method: {g} | Top {len(dfr)} ===")
        print(dfr.to_string(index=False))

    if s.out_dir is not None:
        s.out_dir.mkdir(parents=True, exist_ok=True)
        overall["ALL"].to_csv(s.out_dir / "top_words_overall.csv", index=False)
        for g, dfr in period.items():
            dfr.to_csv(s.out_dir / f"top_words_period_{g}.csv", index=False)
        for g, dfr in method.items():
            dfr.to_csv(s.out_dir / f"top_words_method_{g}.csv", index=False)

if __name__ == "__main__":
    main()
