"""Data-loading helpers for reading prepared CSV corpora."""

from __future__ import annotations

import pandas as pd

from ..data_types import Columns

# pandas の戻り値型はライブラリ側型定義に委ねる

def load_df(csv_path: str, columns: Columns) -> pd.DataFrame:
    """研究用データセットを読み込み、必要列を抽出する。

    Args:
        csv_path (str): 入力 CSV ファイルのパス。
        columns (Columns): 取得対象となる列名設定。

    Returns:
        pandas.DataFrame: 列名を標準化した DataFrame。
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    out: pd.DataFrame = df[[columns.abstract, columns.year, columns.manual_tags]].copy()
    out.columns = ["abstract", "year", "manual_tags"]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out