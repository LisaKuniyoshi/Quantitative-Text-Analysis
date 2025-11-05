"""MNLR（多項ロジスティック回帰）モデルの設計・推定・予測を行う補助関数群。"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_mnlogit(df_obs: pd.DataFrame, base_method: str = "qual"):
    """式インターフェースでMNLogitを推定し、ロバストSEを付ける。

    Args:
        df_obs: 列 doc_id, code, year, method を持つ長形式データ。
        base_method: 手法の基準水準。

    Returns:
        tuple: (robust_results, vanilla_results, categories)
    """
    df = df_obs.copy()
    df["code"] = df["code"].astype("category")
    df["method"] = df["method"].astype("category")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_centered"] = df["year"] - df["year"].mean()

    df["code_num"] = df["code"].cat.codes

    mod = smf.mnlogit(
        f"code_num ~ year_centered + C(method, Treatment('{base_method}'))", data=df
    )
    res = mod.fit(method="newton", maxiter=200, disp=False)
    robust = mod.fit(
        method="newton",
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df["doc_id"]},
    )
    cats = df["code"].cat.categories.tolist()
    return robust, res, cats, df  # 予測では res を使う


def predict_probabilities(res, df_pred: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    """式モデルの予測確率を返す（行=観測、列=コード）。

    Args:
        res: MNLogitの通常結果。
        df_pred: 列 year_centered, method（カテゴリ）等を含むデータ。
        cats: コード名の順序。

    Returns:
        pd.DataFrame: 予測確率。
    """
    p = res.predict(df_pred)  # (n, K)
    cols = list(cats)
    return pd.DataFrame(np.asarray(p), columns=cols, index=df_pred.index)
