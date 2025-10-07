# quant_text_analysis/phrase_discovery.py
from __future__ import annotations

from breame.spelling import get_american_spelling

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import STOPWORDS

from .config import default_columns
from .io_loader import load_df

BASE_DIR: Path = Path(__file__).resolve().parents[2]

CSV_PATH: Path = BASE_DIR / "data" / "raw" / "エクスポートされたアイテム.csv"
OUT_DIR: Path = BASE_DIR / "outputs"
OUT_FILE: Path = OUT_DIR / "phrases_gensim.csv"

MIN_COUNT_BIGRAM: int = 5
THRESHOLD_BIGRAM: float = 10.0
MIN_COUNT_TRIGRAM: int = 3
THRESHOLD_TRIGRAM: float = 7.0

TOP_N_PRINT: int = 100

# ---- 素朴トークナイザ（英字のみ） ----
def simple_tokenize(text: str) -> List[str]:
    import re
    tokens = re.sub(r"[^a-z]+", " ", text.lower()).split()
    return [get_american_spelling(tok) for tok in tokens]

def build_corpus(texts: Sequence[str]) -> List[List[str]]:
    return [simple_tokenize(t) for t in texts]

# ---- Phrases 学習と候補抽出 ----
def train_phrases(
    corpus: Sequence[Sequence[str]],
    min_count: int,
    threshold: float,
    *,
    connector_words: Iterable[str],
) -> Phrases:
    # gensim 4.3.x では common_terms ではなく connector_words を使用
    model = Phrases(
        corpus,
        min_count=min_count,
        threshold=threshold,
        delimiter="_",
        connector_words=frozenset(connector_words),
    )
    return model

def phrase_df_from_model(model: Phrases) -> pd.DataFrame:
    """
    公開APIの export_phrases() を用いて候補を取得。
    戻り値は { 'phrase_with_joiner': score, ... }。
    """
    phrases: Dict[str, float] = model.export_phrases()  # type: ignore[attr-defined]
    if not phrases:
        return pd.DataFrame(columns=["phrase", "n_terms", "score"])
    rows: List[Tuple[str, int, float]] = []
    for ph, sc in phrases.items():
        n_terms = ph.count("_") + 1
        rows.append((ph, n_terms, float(sc)))
    df = pd.DataFrame(rows, columns=["phrase", "n_terms", "score"])
    return df.sort_values(["score", "phrase"], ascending=[False, True]).reset_index(drop=True)

def count_phrase_usage(
    tokenized: Sequence[Sequence[str]],
    joiner: str = "_",
) -> pd.DataFrame:
    from collections import Counter
    total_cnt: Counter[str] = Counter()
    doc_cnt: Counter[str] = Counter()

    for doc in tokenized:
        seen_in_doc: set[str] = set()
        for tok in doc:
            if joiner in tok and len(tok) >= 3:
                total_cnt[tok] += 1
                seen_in_doc.add(tok)
        for p in seen_in_doc:
            doc_cnt[p] += 1

    if not total_cnt:
        return pd.DataFrame(columns=["phrase", "count_total", "doc_count", "doc_rate"])

    rows: List[Tuple[str, int, int]] = [(p, c, doc_cnt[p]) for p, c in total_cnt.items()]
    df = pd.DataFrame(rows, columns=["phrase", "count_total", "doc_count"])
    n_docs = len(tokenized)
    df["doc_rate"] = df["doc_count"] / float(n_docs)
    return df.sort_values(["count_total", "doc_count"], ascending=[False, False]).reset_index(drop=True)

# ---- 実行本体 ----
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cols = default_columns()
    df = load_df(str(CSV_PATH), cols)
    texts: List[str] = df["abstract"].fillna("").astype(str).tolist()

    corpus = build_corpus(texts)

    # bigram 学習
    bigram_model = train_phrases(
        corpus,
        min_count=MIN_COUNT_BIGRAM,
        threshold=THRESHOLD_BIGRAM,
        connector_words=ENGLISH_CONNECTOR_WORDS,  # ← 修正点
    )
    bigram_phraser = Phraser(bigram_model)

    # trigram 学習（bigram 適用後のコーパスに対して）
    corpus_big: List[List[str]] = [list(bigram_phraser[doc]) for doc in corpus]
    trigram_model = train_phrases(
        corpus_big,
        min_count=MIN_COUNT_TRIGRAM,
        threshold=THRESHOLD_TRIGRAM,
        connector_words=ENGLISH_CONNECTOR_WORDS,
    )
    trigram_phraser = Phraser(trigram_model)

    # モデル由来スコア（公開APIを使用）
    df_score_bi = phrase_df_from_model(bigram_model)
    df_score_tri = phrase_df_from_model(trigram_model)
    df_score = pd.concat([df_score_bi, df_score_tri], ignore_index=True)
    df_score = df_score.sort_values(["phrase", "score"], ascending=[True, False]).drop_duplicates("phrase", keep="first")

    # 実コーパスでの使用統計（bigram→trigram を適用）
    corpus_tri: List[List[str]] = [list(trigram_phraser[doc]) for doc in corpus]
    df_usage = count_phrase_usage(corpus_tri)

    # マージ・保存・表示
    out = df_usage.merge(df_score, on="phrase", how="left").sort_values(
        ["score", "count_total", "doc_count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    out.to_csv(OUT_FILE, index=False)

    print(f"\n=== Gensim Phrases — Candidates (top {min(TOP_N_PRINT, len(out))}) ===")
    cols_to_show = ["phrase", "score", "count_total", "doc_count", "doc_rate", "n_terms"]
    print(out[cols_to_show].head(TOP_N_PRINT).to_string(index=False))

if __name__ == "__main__":
    main()
