"""Detailed inspection tool for the per-document preprocessing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, cast

import pandas as pd

from breame.spelling import get_american_spelling

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_text_analysis.io.loader import load_df
from quant_text_analysis.preprocess.nlp_backend import SpacyBackend
from quant_text_analysis.preprocess.normalize import build_normalizer
from quant_text_analysis.preprocess.perdoc import (
    PerDocFreq,
    _build_forced_index,
    _is_break_punct,
    _is_skip_punct,
)
from quant_text_analysis.settings import Settings

ForcedEvent = Dict[str, object]


def _build_alias_map(policy) -> Dict[Tuple[str, ...], str]:
    alias_map: Dict[Tuple[str, ...], str] = {}
    for key, alias in policy.forced_aliases:
        if not key or not alias:
            continue
        lowered = tuple(get_american_spelling(word.lower()) for word in key)
        alias_map[lowered] = alias.strip().lower()
    return alias_map


def _normalize_with_trace(
    doc,
    normalizer,
    forced_index: Dict[Tuple[str, ...], str],
    keys_by_len: List[Tuple[Tuple[str, ...], str]],
    alias_map: Dict[Tuple[str, ...], str],
) -> Dict[str, object]:
    raw_tokens = list(doc)
    raw_info: List[Dict[str, str]] = []

    lemma_lower: List[str] = []
    for tok in raw_tokens:
        spa_lemma = tok.lemma_
        lemma = (spa_lemma or tok.text).lower().strip()
        if lemma in ("-pron-", ""):
            lemma = tok.text.lower()
        lemma_us = get_american_spelling(lemma)
        lemma_lower.append(lemma_us)
        raw_info.append(
            {
                "text": tok.text,
                "lemma": spa_lemma,
                "lemma_us": lemma_us,
                "pos": tok.pos_,
                "ent": tok.ent_type_,
            }
        )

    normalized: List[str] = []
    ignored: List[Dict[str, str]] = []
    forced_events: List[ForcedEvent] = []

    i = 0
    n = len(raw_tokens)
    while i < n:
        if _is_break_punct(raw_tokens[i]):
            i += 1
            continue

        matched = False
        for key, joined in keys_by_len:
            j = i
            k = 0
            surface_seq: List[str] = []
            while j < n and k < len(key):
                tok = raw_tokens[j]
                if _is_break_punct(tok):
                    break
                if _is_skip_punct(tok):
                    j += 1
                    continue
                if lemma_lower[j] == key[k]:
                    surface_seq.append(tok.text)
                    j += 1
                    k += 1
                else:
                    break
            if k == len(key):
                alias = alias_map.get(key)
                forced_events.append(
                    {
                        "start": i,
                        "end": j,
                        "joined": joined,
                        "lemma_key": list(key),
                        "surface_seq": surface_seq,
                        "alias": alias,
                    }
                )
                normalized.append(joined)
                i = j
                matched = True
                break
        if matched:
            continue

        tok = raw_tokens[i]
        norm = normalizer(tok)
        if norm is not None:
            normalized.append(norm)
        else:
            ignored.append({"text": tok.text, "pos": tok.pos_, "ent": tok.ent_type_})
        i += 1

    if normalized:
        counts = Counter(normalized)
        total = float(sum(counts.values()))
        freq: PerDocFreq = {term: count / total for term, count in counts.items()}
    else:
        freq = {}

    return {
        "raw_tokens": raw_info,
        "normalized_tokens": normalized,
        "ignored_tokens": ignored,
        "forced_events": forced_events,
        "freq": freq,
    }


def run_debug(limit: int, output_path: Path) -> None:
    settings = Settings()
    df: pd.DataFrame = load_df(str(settings.csv_path), settings.columns)
    texts: List[str] = df["abstract"].fillna("").astype(str).tolist()
    if not texts:
        raise RuntimeError("入力データの abstract 列が空です。")

    subset = texts[:limit]
    backend = SpacyBackend(model=settings.spacy_model)
    normalizer = build_normalizer(settings.token_policy)
    forced_index = _build_forced_index(settings.token_policy)
    keys_by_len = sorted(forced_index.items(), key=lambda kv: len(kv[0]), reverse=True)
    alias_map = _build_alias_map(settings.token_policy)

    docs = list(backend.pipe(subset))

    results = []
    for idx, doc in enumerate(docs):
        trace = _normalize_with_trace(
            doc,
            normalizer,
            forced_index,
            keys_by_len,
            alias_map,
        )
        raw_tokens_info = cast(List[Dict[str, str]], trace["raw_tokens"])
        results.append(
            {
                "doc_index": idx,
                "abstract_preview": subset[idx][:300],
                "token_count": len(raw_tokens_info),
                **trace,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote per-doc debug info for {len(results)} docs -> {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=3, help="解析するドキュメント数 (default: 3)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out_edited") / "perdoc_debug.json",
        help="詳細情報を保存する JSON パス",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_debug(limit=args.limit, output_path=args.output)


if __name__ == "__main__":
    main()
