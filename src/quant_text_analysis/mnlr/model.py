"""MNLR（多項ロジスティック回帰）モデルの設計・推定・予測を行う補助関数群。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import MultinomialResultsWrapper

from .coding import METHOD_COLUMNS


def fit_mnlogit(
    df_obs: pd.DataFrame,
) -> Tuple[
    MultinomialResultsWrapper,
    List[str],
    pd.DataFrame,
]:
    """式インターフェースで MNLogit を推定し、クラスタロバスト推定量も取得します。

    Args:
        df_obs (pd.DataFrame): ``doc_id`` ``code`` ``year`` および
            手法ダミー列（`qual`, `quan`, `review`, `theoretic`）を持つ長形式データ。

    Returns:
        Tuple[MultinomialResultsWrapper, List[str], pd.DataFrame]:
            クラスタロバスト結果、コード名のリスト、学習に使用した
            前処理済みデータフレーム。
    """
    df: pd.DataFrame = df_obs.copy()
    df["code"] = df["code"].astype("category")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_centered"] = df["year"] - df["year"].mean()

    df["code_num"] = df["code"].cat.codes

    method_cols: List[str] = [col for col in METHOD_COLUMNS if col in df.columns]

    predictors: List[str] = ["year_centered", *method_cols]

    if not predictors:
        predictors = ["1"]

    formula_rhs: str = " + ".join(predictors)
    formula: str = f"code_num ~ {formula_rhs}"

    mod = smf.mnlogit(formula, data=df)
    robust = mod.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["doc_id"]},
    )
    cats: List[str] = df["code"].cat.categories.tolist()
    return robust, cats, df


def predict_probabilities(
    res: MultinomialResultsWrapper,
    df_pred: pd.DataFrame,
    cats: List[str],
) -> pd.DataFrame:
    """MNLogit の推定結果からコードごとの予測確率を計算します。

    Args:
        res (MultinomialResultsWrapper): `fit_mnlogit` が返す推定結果の一つ。
        df_pred (pd.DataFrame): 予測対象データ。``year_centered`` と
            `method_*` ダミー列など、推定時と同じ特徴量を含む。
        cats (List[str]): 予測確率の列順に対応するコード名のリスト。

    Returns:
        pandas.DataFrame: 観測を行方向、コードを列方向に持つ予測確率表。
    """
    p: npt.NDArray[np.float_] = np.asarray(res.predict(df_pred), dtype=float)
    cols: List[str] = list(cats)
    return pd.DataFrame(p, columns=cols, index=df_pred.index)
