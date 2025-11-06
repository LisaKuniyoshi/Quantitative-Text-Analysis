"""MNLR（多項ロジスティック回帰）モデルの設計・推定・予測を行う補助関数群。"""

from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def make_formula(base_method: str, year_var="year_centered"):
    interact_levels = ["quan", "quan", "theoretic"]

    inter_terms = " + ".join(
        f'{year_var}:I(method == "{lvl}")' for lvl in interact_levels
    )

    formula = (
        f"code_num ~ {year_var} + C(method, Treatment('{base_method}'))"
        + (f" + {inter_terms}" if inter_terms else "")
    )

    return formula


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

    # どうしてハードコードを？？
    formula = (
        f"code_num ~ year_centered"
        f" + C(method, Treatment('{base_method}'))"
        f" + C(method, Treatment('{base_method}')):year_centered"
    )

    mod = smf.mnlogit(
        formula,
        data=df
    )

    res = mod.fit(method="newton", maxiter=200, disp=False)
    robust = mod.fit(
        method="newton",
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df["doc_id"]},
    )
    cats = df["code"].cat.categories.tolist()
    return robust, res, cats, df


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
