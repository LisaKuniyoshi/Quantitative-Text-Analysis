"""コード×研究手法のクロス集計を行うためのユーティリティ。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import pandas as pd

from quant_text_analysis.grouping import METHOD_CODE_TO_LABEL

METHOD_CODE_ORDER: tuple[str, ...] = ("qual", "quan", "theoretic", "review", "other")


def build_code_method_crosstab(
    per_doc_codes: Sequence[Sequence[str]],
    method: pd.Series,
    code_order: Sequence[str],
) -> pd.DataFrame:
    """文書単位で、コードの出現有無（1/0）を研究手法ごとに集計したクロス集計表を返す。

    Args:
        per_doc_codes (Sequence[Sequence[str]]): 文書ごとのコード列。
        method (pandas.Series): 文書ごとの研究手法ラベル。
        code_order (Sequence[str]): 行の表示順に用いるコード名の並び。

    Returns:
        pandas.DataFrame: 行がコード、列が手法の 0/1 クロス集計。行名は `code`、列名は `method`。
    """
    rows = []
    column_labels = [METHOD_CODE_TO_LABEL.get(code, code) for code in METHOD_CODE_ORDER]
    for doc_id, codes in enumerate(per_doc_codes):
        method_codes = [*dict.fromkeys(code for code in method.iloc[doc_id])]
        if not method_codes:
            method_codes = ["other"]
        for method_code in method_codes:
            for code in codes:
                rows.append((doc_id, code, method_code))

    code_labels = list(code_order)

    if rows:
        df_long = pd.DataFrame(rows, columns=["doc_id", "code", "method"])
        crosstab = (
            df_long.assign(present=1)
            .pivot_table(
                index="code",
                columns="method",
                values="present",
                aggfunc="sum",
                fill_value=0,
            )
        )
        crosstab = crosstab.rename(columns=METHOD_CODE_TO_LABEL)
    else:
        crosstab = pd.DataFrame(columns=column_labels)

    crosstab = crosstab.reindex(
        index=code_labels,
        columns=column_labels,
        fill_value=0,
    )
    crosstab.index.name = "code"
    crosstab.columns.name = "method"
    return crosstab
