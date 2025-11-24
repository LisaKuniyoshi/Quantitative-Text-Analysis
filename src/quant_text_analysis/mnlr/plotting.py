"""MNLR の確率要約を描画・保存するための補助関数。"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import BinaryResultsWrapper
from matplotlib.ticker import FuncFormatter

from quant_text_analysis.grouping import METHOD_CODE_TO_LABEL
from .coding import METHOD_COLUMNS

FONT_FAMILY = "Yu Mincho"   # or "游明朝"

plt.rcParams["font.family"] = FONT_FAMILY

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
    ols = sm.OLS(y, Xq).fit(cov_type="HC3")

    grid = np.linspace(float(x.min()), float(x.max()), 100)
    Xg = pd.DataFrame({"const": 1.0, "x": grid, "x2": grid**2})
    pred = ols.get_prediction(Xg).summary_frame(alpha=0.05)

    mean  = np.clip(pred["mean"].to_numpy(),          0.0, 1.0)
    lower = np.clip(pred["obs_ci_lower"].to_numpy(),  0.0, 1.0)  # ← 観測予測区間
    upper = np.clip(pred["obs_ci_upper"].to_numpy(),  0.0, 1.0)

    line, = ax.plot(grid, mean, lw=2, label=label, zorder=2)
    ax.fill_between(grid, lower, upper, alpha=0.2,
                    color=line.get_color(), zorder=1)


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
    year = pd.Series(year).reset_index(drop=True)
    method = pd.Series(method).astype(str).reset_index(drop=True)
    prob_df = prob_df.reset_index(drop=True)

    for code in prob_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)

        for m in method.unique():
            mask = (method == m)

            if mask.sum() < 3:
                continue

            y = prob_df[code].loc[mask].astype(float)
            x = year.loc[mask].astype(float)

            x = x.reset_index(drop=True)
            y = y.reset_index(drop=True)


            qfitci_like(ax, x, y, label=m)

        ax.set_xlabel("Year")
        ax.set_ylabel(f"P(Code={code})")
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Method", loc="best", frameon=False)
        ax.set_title(f"Probability (year × method): Code={code}")

        fig.tight_layout()
        fig.savefig(out_dir / f"prob_by_year_{code}.png")
        plt.close(fig)


def _compute_log_symmetric_axis_limits(values: Iterable[float]) -> tuple[float, float]:
    arr = np.fromiter((v for v in values if np.isfinite(v) and v > 0), dtype=float)
    if arr.size == 0:
        return 0.5, 2.0

    log_vals = np.log(arr)
    if log_vals.size == 0 or not np.isfinite(log_vals).any():
        return 0.5, 2.0

    max_abs = float(np.max(np.abs(log_vals)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        max_abs = 0.25
    else:
        max_abs *= 1.2

    axis_min = float(np.exp(-max_abs))
    axis_max = float(np.exp(max_abs))
    axis_min = max(axis_min, 1e-4)
    axis_max = max(axis_max, axis_min * 1.5)
    return axis_min, axis_max


def _build_log_ticks(axis_min: float, axis_max: float) -> list[float]:
    if not np.isfinite(axis_min) or not np.isfinite(axis_max):
        return []

    ticks: set[float] = set()
    for exponent in range(-6, 7):
        for multiplier in (1.0, 2.0, 5.0):
            tick = multiplier * (10 ** exponent)
            if axis_min <= tick <= axis_max:
                ticks.add(tick)

    ordered = sorted(ticks)
    return ordered


def plot_method_odds_ratios(
    odds_df: pd.DataFrame,
    method_order: Iterable[str],
    out_path: Path,
    *,
    show_titles: bool = True,
) -> None:
    """Plot odds ratios for method dummies per code as grayscale facets."""

    method_order = list(method_order)
    if odds_df.empty or not method_order:
        return

    odds_df = odds_df.copy()
    for column in ("coef", "0.025", "0.975"):
        odds_df[column] = pd.to_numeric(odds_df[column], errors="coerce")

    odds_df = odds_df.dropna(subset=["coef", "0.025", "0.975"])
    if odds_df.empty:
        return

    method_set = set(method_order)
    odds_df = odds_df[odds_df["exog"].isin(method_set)]
    if odds_df.empty:
        return

    order_map = {name: idx for idx, name in enumerate(method_order)}
    odds_df["method_rank"] = odds_df["exog"].map(order_map)
    odds_df = odds_df.dropna(subset=["method_rank"])
    if odds_df.empty:
        return

    axis_min, axis_max = _compute_log_symmetric_axis_limits(
        itertools.chain(odds_df["coef"], odds_df["0.025"], odds_df["0.975"])
    )

    codes = sorted(odds_df["endog"].unique())
    n_codes = len(codes)
    ncols = 2 if n_codes > 1 else 1
    nrows = int(np.ceil(n_codes / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 3 * nrows),
        sharex=True,
        dpi=120,
    )

    axes = np.atleast_1d(axes).flatten()
    method_labels = {m: METHOD_CODE_TO_LABEL.get(m, m) for m in method_order}

    for ax, code in itertools.zip_longest(axes, codes):
        if code is None:
            ax.axis("off")
            continue

        data = odds_df[odds_df["endog"] == code].copy()
        data = data.sort_values("method_rank")
        if data.empty:
            ax.axis("off")
            continue

        positions = np.arange(len(data))
        errors = np.vstack(
            [data["coef"] - data["0.025"], data["0.975"] - data["coef"]]
        )

        ax.set_xscale("log")

        ax.errorbar(
            data["coef"],
            positions,
            xerr=errors,
            fmt="o",
            color="black",
            ecolor="dimgray",
            elinewidth=1.0,
            capsize=3.0,
        )
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlim(axis_min, axis_max)

        tick_values = _build_log_ticks(axis_min, axis_max)
        if tick_values:
            ax.set_xticks(tick_values)
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda val, _: f"{val:g}")
            )

        ax.set_yticks(positions)
        ax.set_yticklabels([method_labels.get(m, m) for m in data["exog"]])
        ax.invert_yaxis()
        ax.grid(axis="x", color="lightgray", linestyle=":", linewidth=0.5, which="both")
        if show_titles:
            ax.set_title(str(code))

    for idx, ax in enumerate(axes):
        if idx // ncols == nrows - 1:
            ax.set_xlabel("オッズ比")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_year_odds_ratios(
    odds_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot odds ratios for the year effect across codes as a single panel."""

    if odds_df.empty:
        return

    odds_df = odds_df.copy()
    for column in ("coef", "0.025", "0.975"):
        odds_df[column] = pd.to_numeric(odds_df[column], errors="coerce")

    odds_df = odds_df.dropna(subset=["coef", "0.025", "0.975"])
    if odds_df.empty:
        return

    odds_df = odds_df.sort_values("coef", ascending=False)

    axis_min, axis_max = _compute_log_symmetric_axis_limits(
        itertools.chain(odds_df["coef"], odds_df["0.025"], odds_df["0.975"])
    )

    positions = np.arange(len(odds_df))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    errors = np.vstack(
        [odds_df["coef"] - odds_df["0.025"], odds_df["0.975"] - odds_df["coef"]]
    )

    ax.set_xscale("log")

    ax.errorbar(
        odds_df["coef"],
        positions,
        xerr=errors,
        fmt="o",
        color="black",
        ecolor="dimgray",
        elinewidth=1.0,
        capsize=3.0,
    )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlim(axis_min, axis_max)

    tick_values = _build_log_ticks(axis_min, axis_max)
    if tick_values:
        ax.set_xticks(tick_values)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
    ax.set_yticks(positions)
    ax.set_yticklabels(list(odds_df["endog"]))
    ax.invert_yaxis()
    ax.grid(axis="x", color="lightgray", linestyle=":", linewidth=0.5, which="both")
    ax.set_xlabel("オッズ比")
    # ax.set_title("Effect of Year (Odds Ratio per 1-year increase)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_binary_year_effect_prediction(
    result: BinaryResultsWrapper,
    df_training: pd.DataFrame,
    out_path: Path,
    *,
    ylabel: str = "Predicted probability",
    y_limits: tuple[float, float] | None = None,
    # title: str = "Year effect (method dummies = 0)",
    n_points: int = 200,
) -> None:
    """Plot get_prediction-based year effect for a binary logit in grayscale."""

    if result is None or df_training is None or df_training.empty:
        return

    if "year" not in df_training.columns:
        return

    year_values = pd.to_numeric(df_training["year"], errors="coerce").dropna()
    if year_values.nunique() < 2:
        return

    exog_names = list(getattr(result.model, "exog_names", []))
    if "year_centered" not in exog_names:
        return

    year_min = float(year_values.min())
    year_max = float(year_values.max())
    if not np.isfinite(year_min) or not np.isfinite(year_max):
        return

    grid_years = np.linspace(year_min, year_max, max(2, int(n_points)))
    year_center = float(year_values.mean())
    centered_vals = grid_years - year_center

    predict_df = pd.DataFrame(index=range(len(grid_years)))
    if "Intercept" in exog_names:
        predict_df["Intercept"] = 1.0

    for name in exog_names:
        if name == "Intercept":
            continue
        if name == "year_centered":
            predict_df[name] = centered_vals
        else:
            predict_df[name] = 0.0

    try:
        pred = result.get_prediction(predict_df)
    except Exception:
        return

    frame = pred.summary_frame(alpha=0.05)
    mean_candidates = ("mean", "predicted")
    lower_candidates = ("mean_ci_lower", "obs_ci_lower", "ci_lower")
    upper_candidates = ("mean_ci_upper", "obs_ci_upper", "ci_upper")
    mean_col = next((c for c in mean_candidates if c in frame.columns), None)
    lower_col = next((c for c in lower_candidates if c in frame.columns), None)
    upper_col = next((c for c in upper_candidates if c in frame.columns), None)
    if mean_col is None or lower_col is None or upper_col is None:
        return

    mean = np.clip(frame[mean_col].to_numpy(), 0.0, 1.0)
    lower = np.clip(frame[lower_col].to_numpy(), 0.0, 1.0)
    upper = np.clip(frame[upper_col].to_numpy(), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(grid_years, mean, color="black", linewidth=2.0, label="予測値")
    ax.fill_between(grid_years, lower, upper, color="gainsboro", alpha=0.6, label="95% 信頼区間")
    ax.set_xlabel("出版年")
    ax.set_ylabel(ylabel)
    # ax.set_title(title)

    ymin, ymax = y_limits if y_limits is not None else (0.0, 0.2)
    ax.set_ylim(float(ymin), float(ymax))
    ax.grid(color="lightgray", linestyle=":", linewidth=0.5)
    ax.legend(frameon=False, loc="best")
    # ax.text(
    #     0.5,
    #     -0.18,
    #     "Method dummies fixed at 0",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="top",
    #     fontsize=9,
    #     color="dimgray",
    # )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
