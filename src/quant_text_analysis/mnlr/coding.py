"""MNLR（多項ロジスティック回帰）で用いる、トークン→コード写像の補助関数群。"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


RowType = Tuple[int, str, int | float | None, str | None]


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
    years: Iterable | pd.Series,
    methods: Iterable | pd.Series,
    code_index: Dict[str, str],
) -> pd.DataFrame:
    """文書ごとのトークン列を観測単位の行に展開し、DataFrame を返す。

    各トークンが `code_index` に存在する都度、1行（`doc_id`, `code`, `year`, `method`）を生成する。

    Args:
        per_doc_tokens (Sequence[Sequence[str]]): 文書ごとのトークン列。
        years (Iterable | pandas.Series): 各文書の出版年。`pandas.Series` でなくてもよいが、
            長さが文書数と同じであること。
        methods (Iterable | pandas.Series): 各文書の研究手法ラベル。
        code_index (dict[str, str]): トークン→コードの逆引きインデックス。

    Returns:
        pandas.DataFrame: 列 `doc_id`, `code`, `year`, `method` を持つ長データ。

    Raises:
        ValueError: `years` と `methods` の長さが文書数と一致しない場合。
    """
    rows: List[RowType] = []
    yrs = (
        years
        if isinstance(years, pd.Series)
        else pd.Series(list(years))
    )
    mds = (
        methods
        if isinstance(methods, pd.Series)
        else pd.Series(list(methods))
    )

    for doc_id, tokens in enumerate(per_doc_tokens):
        yr = yrs.iloc[doc_id]
        md = mds.iloc[doc_id]
        for token in tokens:
            code = code_index.get(token)
            if code is not None:
                rows.append((doc_id, code, yr, md))

    return pd.DataFrame(rows, columns=["doc_id", "code", "year", "method"])
