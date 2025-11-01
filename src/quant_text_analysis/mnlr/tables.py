"""コード×研究手法のクロス集計を行うためのユーティリティ。"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def build_code_method_crosstab(
    per_doc_codes: Sequence[Sequence[str]],
    method: pd.Series,
    code_order: Sequence[str],
    include_methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    """文書単位で、コードの出現有無（1/0）を研究手法ごとに集計したクロス集計表を返す。

    Args:
        per_doc_codes (Sequence[Sequence[str]]): 文書ごとのコード列。
        method (pandas.Series): 文書ごとの研究手法ラベル。
        code_order (Sequence[str]): 行の表示順に用いるコード名の並び。
        include_methods (Iterable[str] | None): 列に必ず含めたい手法名。観測に現れない場合でも 0 で列を作成する。

    Returns:
        pandas.DataFrame: 行がコード、列が手法の 0/1 クロス集計。行名は `code`、列名は `method`。
    """
    method_series = method.fillna("other").astype(str)

    rows = []
    for doc_id, codes in enumerate(per_doc_codes):
        for code in codes:
            rows.append((doc_id, code, method_series.iloc[doc_id]))

    extra_methods = {str(m) for m in include_methods or ()}
    observed_methods = set(method_series.astype(str).tolist())
    column_labels = sorted(observed_methods | extra_methods)
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
