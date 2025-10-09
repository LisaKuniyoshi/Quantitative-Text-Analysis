# quant_text_analysis/cli.py
from __future__ import annotations

from pathlib import Path
from typing import List

from .config import default_columns, default_token_policy, default_ranking_params, SPACY_MODEL
from .io_loader import load_df
from .grouping import period_group_year, method_group
from .nlp_backend import SpacyBackend
from .normalize import build_normalizer
from .frequency import frequency_rankings
from .cache import analyze_with_cache

# プロジェクトルート: .../（2つ上）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = DATA_DIR / "cache"

# 入力CSV（例: data/raw/エクスポートされたアイテム.csv）
CSV_PATH: Path = RAW_DIR / "エクスポートされたアイテム.csv"

# 出力（例: outputs/）
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUT_DIR: Path | None = OUTPUTS_DIR   # 保存不要なら None に設定

TOP_N: int = 200
MIN_DOCS: int = 1

# ---- 実行本体 ---------------------------------------------------------------
def main() -> None:
    cols = default_columns()
    policy = default_token_policy()
    rankp = default_ranking_params()
    # rankp は今は参照しないが将来拡張を考慮して残置

    df = load_df(str(CSV_PATH), cols)
    df["period"] = df["year"].map(period_group_year)
    df["method"] = df["manual_tags"].map(method_group)

    backend = SpacyBackend(model=SPACY_MODEL)
    normalizer = build_normalizer(policy)

    abstracts: List[str] = df["abstract"].fillna("").tolist()
    _, per_doc_freqs = analyze_with_cache(
        backend, normalizer, abstracts, policy, cache_dir=str(CACHE_DIR)
    )

    overall = frequency_rankings(per_doc_freqs, None, top_n=TOP_N, min_docs=MIN_DOCS)
    period  = frequency_rankings(per_doc_freqs, df["period"].tolist(), top_n=TOP_N, min_docs=MIN_DOCS)
    method  = frequency_rankings(per_doc_freqs, df["method"].tolist(), top_n=TOP_N, min_docs=MIN_DOCS)

    print(f"\n=== Overall (All documents) | Top {len(overall['ALL'])} ===")
    print(overall["ALL"].to_string(index=False))
    for g, dfr in period.items():
        print(f"\n=== Period: {g} | Top {len(dfr)} ===")
        print(dfr.to_string(index=False))
    for g, dfr in method.items():
        print(f"\n=== Method: {g} | Top {len(dfr)} ===")
        print(dfr.to_string(index=False))

    if OUT_DIR is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        overall["ALL"].to_csv(OUT_DIR / "top_words_overall.csv", index=False)
        for g, dfr in period.items():
            dfr.to_csv(OUT_DIR / f"top_words_period_{g}.csv", index=False)
        for g, dfr in method.items():
            dfr.to_csv(OUT_DIR / f"top_words_method_{g}.csv", index=False)

if __name__ == "__main__":
    main()
