"""MNLR（多項ロジスティック回帰）モデルの設計・推定・予測を行う補助関数群。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class Design:
    """MNLogit 推定に必要な設計情報を保持するコンテナ。

    Attributes:
        y (pandas.Series): 目的変数。カテゴリを整数に符号化した系列。
        X (pandas.DataFrame): 説明変数のデザイン行列（定数項、中心化年、手法ダミー等）。
        clusters (pandas.Series): クラスタ変数（例: 文書 ID）。クラスタ・ロバスト分散の群指定に使用。
        year (pandas.Series): 年（数値化・欠損補完後）。
        method (pandas.Series): 研究手法カテゴリ（categorical）。
        categories (tuple[str, ...]): 目的変数の元ラベルの並び。
    """

    y: pd.Series
    X: pd.DataFrame
    clusters: pd.Series
    year: pd.Series
    method: pd.Series
    categories: Tuple[str, ...]


def build_design(df_obs: pd.DataFrame) -> Design:
    """観測単位の長形式 DataFrame から、MNLogit 用の `Design` を構築する。

    前処理として、`year` は数値化し欠損を中央値で補完、平均中心化を行う。`method` は
    第1カテゴリを基準にダミー化する。`y` は `code` のカテゴリを整数化したもの。

    Args:
        df_obs (pandas.DataFrame): 列 `doc_id`, `code`, `year`, `method` を含む長形式データ。

    Returns:
        Design: 推定に必要な y, X, clusters, year, method, categories を格納したオブジェクト。
    """
    y_cat = pd.Categorical(df_obs["code"])
    y = pd.Series(y_cat.codes, index=df_obs.index, dtype=int)

    method = df_obs["method"].fillna("other").astype("category")
    year_num = pd.to_numeric(df_obs["year"], errors="coerce")
    if year_num.isna().any():
        year_num = year_num.fillna(year_num.median())
    year_centered = (year_num - year_num.mean()).astype(float)

    d_method = pd.get_dummies(method, prefix="method", drop_first=True)
    X = pd.concat(
        [
            pd.Series(1.0, index=df_obs.index, name="const"),
            year_centered.rename("year_centered"),
        ] + ([d_method] if not d_method.empty else []),
        axis=1,
    ).astype(float)

    clusters = df_obs["doc_id"].astype(int)
    categories = tuple(y_cat.categories.astype(str).tolist())
    return Design(
        y=y,
        X=X,
        clusters=clusters,
        year=year_num,
        method=method,
        categories=categories,
    )


def fit_multinomial_cluster_robust(design: Design):
    """文書クラスタに対してロバストな共分散推定を伴う MNLogit を推定する。

    Args:
        design (Design): 設計オブジェクト。

    Returns:
        tuple[statsmodels.discrete.discrete_model.MNLogitResults,
              statsmodels.discrete.discrete_model.MNLogitResults]:
            0番目にクラスタ・ロバスト推定の結果、1番目に通常推定の結果。
    """
    model = sm.MNLogit(endog=design.y, exog=design.X)
    standard = model.fit(method="newton", disp=False)
    robust = model.fit(
        method="newton",
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": design.clusters},
    )
    return robust, standard


def predict_probabilities(
    res,
    X: pd.DataFrame,
    categories: Tuple[str, ...],
) -> pd.DataFrame:
    """各観測のカテゴリ別選択確率を推定し DataFrame で返す。

    Args:
        res: `statsmodels` の MNLogit 推定結果。
        X (pandas.DataFrame): 予測に用いるデザイン行列。
        categories (tuple[str, ...]): 列名に用いるカテゴリ名の並び。

    Returns:
        pandas.DataFrame: 行が観測、列がカテゴリの確率表。
    """
    probs = res.predict(X)
    n_class = probs.shape[1] if probs.ndim == 2 else 1
    cols = (
        list(categories)[:n_class]
        if n_class == len(categories)
        else [f"class_{j}" for j in range(n_class)]
    )
    return pd.DataFrame(probs, columns=cols, index=X.index)
