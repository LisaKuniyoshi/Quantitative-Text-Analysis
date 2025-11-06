"""MNLR（多項ロジスティック回帰）モデルの設計・推定・予測を行う補助関数群。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import MultinomialResultsWrapper


def fit_mnlogit(
    df_obs: pd.DataFrame,
    base_method: str = "qual",
) -> Tuple[
    MultinomialResultsWrapper,
    MultinomialResultsWrapper,
    List[str],
    pd.DataFrame,
]:
    """式インターフェースで MNLogit を推定し、クラスタロバスト推定量も取得します。

    Args:
        df_obs (pd.DataFrame): ``doc_id`` ``code`` ``year`` ``method`` 列を持つ長形式データ。
        base_method (str, optional): ``method`` の基準水準。既定は ``"qual"`` 。

    Returns:
        Tuple[MultinomialResultsWrapper, MultinomialResultsWrapper, List[str], pd.DataFrame]:
            クラスタロバスト結果、通常推定結果、コード名のリスト、学習に使用した
            前処理済みデータフレーム。
    """
    df: pd.DataFrame = df_obs.copy()
    df["code"] = df["code"].astype("category")
    df["method"] = df["method"].astype("category")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_centered"] = df["year"] - df["year"].mean()

    df["code_num"] = df["code"].cat.codes

    mod = smf.mnlogit(
        f"code_num ~ year_centered + C(method, Treatment('{base_method}'))", data=df
    )
    res: MultinomialResultsWrapper = mod.fit(
        method="newton",
        maxiter=200,
        disp=False,
    )
    robust: MultinomialResultsWrapper = mod.fit(
        method="newton",
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": df["doc_id"]},
    )
    cats: List[str] = df["code"].cat.categories.tolist()
    return robust, res, cats, df


def predict_probabilities(
    res: MultinomialResultsWrapper,
    df_pred: pd.DataFrame,
    cats: List[str],
) -> pd.DataFrame:
    """MNLogit の推定結果からコードごとの予測確率を計算します。

    Args:
        res (MultinomialResultsWrapper): `fit_mnlogit` が返す推定結果の一つ。
        df_pred (pd.DataFrame): 予測対象データ。``year_centered`` ``method`` などの列を含む。
        cats (List[str]): 予測確率の列順に対応するコード名のリスト。

    Returns:
        pandas.DataFrame: 観測を行方向、コードを列方向に持つ予測確率表。
    """
    p: npt.NDArray[np.float_] = np.asarray(res.predict(df_pred), dtype=float)
    cols: List[str] = list(cats)
    return pd.DataFrame(p, columns=cols, index=df_pred.index)
