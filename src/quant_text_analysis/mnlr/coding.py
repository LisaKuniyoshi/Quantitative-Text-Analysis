"""MNLR（多項ロジスティック回帰）で用いる、トークン→コード写像の補助関数群。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

METHOD_COLUMNS: tuple[str, ...] = ("qual", "quan", "review", "theoretic")

def invert_code_map(code_map: Dict[str, Tuple[str, ...]]) -> Dict[str, str]:
    """最初に一致したトークンに対応するコードへの逆引きインデックスを作成して返す。

    Args:
        code_map (dict[str, tuple[str, ...]]): コード名をキー、当該コードに属する
            トークン列を値とする対応表。

    Returns:
        dict[str, str]: 各トークンをキー、最初に対応づけられたコード名を値とする辞書。

    Note:
        1つのトークンが複数のコードに列挙されている場合は、`code_map` の走査順に
        よって先勝ちで割り当てられる。
    """
    idx: Dict[str, str] = {}
    for code, tokens in code_map.items():
        for token in tokens:
            idx.setdefault(token, code)
    return idx


def codes_per_doc(
    per_doc_tokens: Sequence[Sequence[str]],
    code_index: Dict[str, str],
) -> List[List[str]]:
    """各文書のトークン列をコード列へ変換する。文書内では重複を排し、辞書順に整列する。

    Args:
        per_doc_tokens (Sequence[Sequence[str]]): 文書ごとのトークン列。
        code_index (dict[str, str]): トークン→コードの逆引きインデックス（`invert_code_map` の出力）。

    Returns:
        list[list[str]]: 文書ごとのコード列。各内側リストは一意なコードのみを含み、
        辞書順（昇順）で並ぶ。
    """
    collected: List[List[str]] = []
    for tokens in per_doc_tokens:
        codes = {code_index[token] for token in tokens if token in code_index}
        collected.append(sorted(codes))
    return collected


def rows_from_tokens(
    per_doc_tokens: Sequence[Sequence[str]],
    years: pd.Series,
    methods: pd.Series,
    code_index: Dict[str, str],
) -> pd.DataFrame:
    """文書ごとのトークン列を観測単位の行に展開し、マルチホット手法ダミーを付与する。

    各トークンが `code_index` に存在する都度、1 行（`doc_id`, `code`, `year`）を生成し、
    併せて手法カテゴリのダミー列（`qual`, `quan`, `review`, `theoretic`）を付与する。

    Args:
        per_doc_tokens (Sequence[Sequence[str]]): 文書ごとのトークン列。
        years (pandas.Series): 各文書の出版年。
        methods (pandas.Series): 各文書の研究手法ラベル集合（list[str]）。
        code_index (dict[str, str]): トークン→コードの逆引きインデックス。

    Returns:
        pandas.DataFrame: 列 `doc_id`, `code`, `year`, `qual`, `quan`, `review`, `theoretic`
        を持つ長データ。

    Raises:
        ValueError: `per_doc_tokens`, `years`, `methods` の長さが一致しない場合。
    """
    n_docs = len(per_doc_tokens)
    if len(years) != n_docs or len(methods) != n_docs:
        raise ValueError("per_doc_tokens, years, methods の長さが一致しません。")

    rows: List[dict[str, int | str | float]] = []
    for doc_id, toks in enumerate(per_doc_tokens):
        yr = years.iloc[doc_id]
        method_flags = {name: 0 for name in METHOD_COLUMNS}
        method_set = {str(tag) for tag in methods.iloc[doc_id]}
        for name in METHOD_COLUMNS:
            if name in method_set:
                method_flags[name] = 1

        for tok in toks:
            code = code_index.get(tok)
            if code is None:
                continue
            row = {
                "doc_id": doc_id,
                "code": code,
                "year": yr,
            }
            row.update(method_flags)
            rows.append(row)

    columns: List[str] = ["doc_id", "code", "year", *METHOD_COLUMNS]
    return pd.DataFrame(rows, columns=columns)
