# quant_text_analysis/cli.py
from __future__ import annotations

# --- クリック実行対応：親ディレクトリを import 対象に追加 --------------------
if __name__ == "__main__" and __package__ is None:  # pyright: ignore[reportConstantRedefinition]
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
from typing import List

# 絶対インポート（相対は使わない）
from quant_text_analysis.config import default_columns, default_token_policy, default_ranking_params
from quant_text_analysis.io_loader import load_df
from quant_text_analysis.grouping import period_group_year, method_group
from quant_text_analysis.nlp_backend import SpacyBackend
from quant_text_analysis.normalize import build_normalizer
from quant_text_analysis.frequency import frequency_rankings
from quant_text_analysis.cache import analyze_with_cache

# ---- 固定設定（必要に応じて変更） -------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parents[1]
CSV_PATH: Path = BASE_DIR / "エクスポートされたアイテム.csv"
CACHE_DIR: Path = BASE_DIR / ".cache"
OUT_DIR: Path | None = BASE_DIR / "out"   # 保存不要なら None
TOP_N: int = 200
MIN_DOCS: int = 1
SPACY_MODEL: str = "en_core_web_sm"

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
