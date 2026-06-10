"""Phrase mining with Gensim Phrases (bigrams/trigrams).

概要:
    英字のみのトークン化と英米表記統一を行い、Gensim Phrases で bigram と
    trigram を学習します。モデルのスコアとコーパスでの使用統計を結合し、
    スコア・出現数・文書率などの指標を CSV に保存します。

I/O:
    読み込み:
        - CSV: Settings.csv_path（要旨列）
    書き込み:
        - outputs/phrases_gensim.csv

注意事項:
    - Phrases 学習では `ENGLISH_CONNECTOR_WORDS` を接続語として使用します。
    - 表記統一に `breame.spelling.get_american_spelling` を利用します。

使用例:
    python -m quant_text_analysis phrases
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import re

from breame.spelling import get_american_spelling
import pandas as pd
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases, Phraser
from gensim.parsing.preprocessing import STOPWORDS

from ..config import default_columns
from ..settings import Settings
from ..io.loader import load_df

settings = Settings()
OUT_FILE: Path = settings.out_dir / "phrases_gensim.csv"

MIN_COUNT_BIGRAM: int = 5
THRESHOLD_BIGRAM: float = 10.0
MIN_COUNT_TRIGRAM: int = 3
THRESHOLD_TRIGRAM: float = 7.0

TOP_N_PRINT: int = 100

# ---- 素朴トークナイザ（英字のみ） ----
def simple_tokenize(text: str) -> List[str]:
    """英字のみを対象にトークン化し、表記統一を行う。

    Args:
        text (str): トークン化対象の文字列。

    Returns:
        list[str]: 小文字化・表記統一済みトークン列。
    """

    tokens = re.sub(r"[^a-z]+", " ", text.lower()).split()
    return [get_american_spelling(tok) for tok in tokens]

def build_corpus(texts: Sequence[str]) -> List[List[str]]:
    """文字列コーパスをトークン化済みコーポラへ変換する。

    Args:
        texts (Sequence[str]): トークン化対象の文書列。

    Returns:
        list[list[str]]: トークン化された文書ごとのトークン列。
    """

    return [simple_tokenize(t) for t in texts]

# ---- Phrases 学習と候補抽出 ----
def train_phrases(
    corpus: Sequence[Sequence[str]],
    min_count: int,
    threshold: float,
    *,
    connector_words: Iterable[str],
) -> Phrases:
    """Phrases モデルを学習して返す。

    Args:
        corpus (Sequence[Sequence[str]]): 学習対象のトークン化済み文書群。
        min_count (int): フレーズ抽出の最小出現回数。
        threshold (float): フレーズ候補を採用するスコア閾値。
        connector_words (Iterable[str]): 接続語として扱う語の集合。

    Returns:
        Phrases: 学習済みの Phrases モデル。
    """
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
    """Phrases モデルから候補フレーズを抽出する。

    Args:
        model (Phrases): 評価対象の Phrases モデル。

    Returns:
        pandas.DataFrame: フレーズ、語数、スコアの一覧。
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
    """トークン列から抽出フレーズの出現統計を集計する。

    Args:
        tokenized (Sequence[Sequence[str]]): トークン化済み文書群。
        joiner (str): フレーズを結合する際の区切り文字。

    Returns:
        pandas.DataFrame: フレーズごとの出現回数・文書数・文書率。
    """
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
    """bigram と trigram の候補を学習し、統計を出力する。

    Notes:
        - bi→tri の順に適用し、モデルの `export_phrases()` と実使用回数をマージする。
        - 上位候補は標準出力に表示し、全結果を CSV に保存する。
    """
    settings.ensure_out_dir()

    cols = default_columns()
    df = load_df(str(settings.csv_path), cols)
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
