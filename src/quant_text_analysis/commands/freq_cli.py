# quant_text_analysis/cli.py
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
