"""MNLR の確率要約を描画・保存するための補助関数。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def qfitci_like(ax, x: pd.Series, y: pd.Series, label: str) -> None:
    """Stata の `qfitci` 風に、2次曲線による近似と 95% 信頼区間を描く。

    モデルは OLS: `y ~ 1 + x + x^2` を当てはめ、予測平均と信頼区間を描画する。

    Args:
        ax (matplotlib.axes.Axes): 描画先の軸。
        x (pandas.Series): 説明変数。
        y (pandas.Series): 目的変数（確率など）。
        label (str): 凡例に表示する系列名。

    Returns:
        None
    """
    if len(x) < 3:
        return

    Xq = pd.DataFrame({"const": 1.0, "x": x, "x2": x**2})
    ols = sm.OLS(y, Xq).fit()

    grid = np.linspace(float(x.min()), float(x.max()), 100)
    Xg = pd.DataFrame({"const": 1.0, "x": grid, "x2": grid**2})
    pred = ols.get_prediction(Xg)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    ax.plot(grid, mean, lw=2, label=label)
    ax.fill_between(grid, ci[:, 0], ci[:, 1], alpha=0.2)


def plot_prob_by_year_with_method(
    prob_df: pd.DataFrame,
    year: pd.Series,
    method: pd.Series,
    out_dir,
) -> None:
    """各コード列について、年×手法で 2 次近似曲線と 95% 信頼区間を重ねて PNG で保存する。

    Args:
        prob_df (pandas.DataFrame): 予測確率の表。列はコード、行は観測。
        year (pandas.Series): 観測ごとの年。
        method (pandas.Series): 観測ごとの研究手法カテゴリ。
        out_dir (pathlib.Path | str): 出力ディレクトリ。

    Returns:
        None

    Notes:
        観測数が 3 未満の手法カテゴリはスキップする。ファイル名は `prob_by_year_{code}.png`。
    """
    methods = sorted(method.astype(str).unique())

    for code in prob_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

        for m in methods:
            mask = method.astype(str) == m
            if mask.sum() < 3:
                continue
            prob_series = pd.Series(prob_df.loc[mask, code])
            qfitci_like(ax, year[mask], prob_series, label=m)

        ax.set_xlabel("Year")
        ax.set_ylabel(f"P(Code={code})")
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Method", loc="best", frameon=False)
        ax.set_title(f"Probability (year × method): Code={code}")

        fig.tight_layout()
        fig.savefig(out_dir / f"prob_by_year_{code}.png")
        plt.close(fig)
