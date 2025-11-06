from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.discrete.discrete_margins import DiscreteMargins
from statsmodels.discrete.discrete_model import MultinomialResults, MultinomialResultsWrapper
from statsmodels.stats.multitest import multipletests


FloatArray = npt.NDArray[np.float_]
MULTITEST_METHOD_MAP: Dict[str, str] = {
    "holm-sidak": "hs",
    "bonferroni": "bonf",
    "fdr_bh": "fdr_bh",
}


def _ensure_float(value: Any) -> float:
    """Return a Python float converted from statsmodels/pandas scalar outputs."""
    arr = np.asarray(value, dtype=float)
    return float(arr.item())


def t_test_pairwise_mnlogit(
    res: MultinomialResults | MultinomialResultsWrapper,
    term_name: str,
    alpha: float = 0.05,
    method: str = "holm-sidak",
) -> pd.DataFrame:
    """指定したカテゴリ効果に対して係数差のペアワイズ t 検定を実施します。

    Args:
        res (MultinomialResults | MultinomialResultsWrapper):
            `statsmodels.MNLogit` の推定結果。
        term_name (str): 係数名の接頭辞（例: ``"C(method, Treatment('qual'))"``）。
        alpha (float, optional): 多重比較補正後に用いる有意水準。既定は ``0.05`` 。
        method (str, optional): 多重比較補正の方式。``holm-sidak`` や ``bonferroni`` など。

    Returns:
        pandas.DataFrame: 各式ごとの水準ペアについて、効果量・標準誤差・t 値・
        p 値・補正後 p 値・棄却フラグを持つ結果表。

    Raises:
        ValueError: ``term_name`` に対応する係数が見つからない場合、または水準列が
            デザイン行列内に存在しない場合。
    """
    # 1) 基準水準と因子名を term_name から取得
    match_base: re.Match[str] | None = re.search(r"Treatment\('([^']+)\')", term_name)
    if not match_base:
        raise ValueError("term_name から基準水準が取得できません（Treatment('...') を明示してください）")
    base: str = match_base.group(1)

    # 2) この term に属する水準（非基準）を exog_names から抽出（順序は exog_names に従う）
    exog_names: List[str] = list(res.model.exog_names)
    pat: str = term_name + "[T."
    levels_nonbase: List[str] = []
    var_index: Dict[str, int] = {}
    for j, nm in enumerate(exog_names):
        var_index[nm] = j
        if nm.startswith(pat) and nm.endswith("]"):
            levels_nonbase.append(nm[len(pat):-1])

    if not levels_nonbase:
        raise ValueError(f"指定の term が見つかりません: {term_name}")

    levels: List[str] = [base] + levels_nonbase

    # 3) パラメタ並びに合わせた位置写像（変数 j, 式 eq -> フラット位置）
    k_vars: int
    k_eq: int
    k_vars, k_eq = res.params.shape
    k_params: int = int(res.cov_params().shape[0])
    assert k_vars * k_eq == k_params, "係数次元の不一致"

    def pos(var_name: str, eq: int) -> int:
        j = var_index[var_name]
        return j + eq * k_vars

    # 5) ペア比較を式ごとに実行（係数差の t 検定）。ロバスト共分散は res 側設定が反映される
    rows: List[Dict[str, float | int | str]] = []
    for a_i, a in enumerate(levels):
        for b in levels[a_i + 1 :]:
            for eq in range(k_eq):
                L: FloatArray = np.zeros((1, k_params))
                # 基準は係数0なので、非基準のみ位置に +1 / -1 を置く
                if a != base:
                    vn = f"{term_name}[T.{a}]"
                    if vn not in var_index:
                        # 訓練データに出現しない水準が基準扱いになっている可能性
                        raise ValueError(f"水準 {a} に対応する列が見つかりません: {vn}")
                    L[0, pos(vn, eq)] = 1.0
                if b != base:
                    vn = f"{term_name}[T.{b}]"
                    if vn not in var_index:
                        raise ValueError(f"水準 {b} に対応する列が見つかりません: {vn}")
                    L[0, pos(vn, eq)] -= 1.0

                tr = res.t_test(L)
                rows.append(
                    {
                        "eq": eq,
                        "level_a": a,
                        "level_b": b,
                        "effect": _ensure_float(tr.effect),
                        "stderr": _ensure_float(tr.sd),
                        "t": _ensure_float(tr.tvalue),
                        "p": _ensure_float(tr.pvalue),
                    }
                )

    out: pd.DataFrame = pd.DataFrame(rows)

    # 6) 式ごとに多重比較補正
    outs: List[pd.DataFrame] = []
    method_code: str = MULTITEST_METHOD_MAP.get(method, "hs")
    for eq in range(k_eq):
        sub: pd.DataFrame = out.loc[out["eq"] == eq].copy()
        # ペアが1つだけなら multipletests は不要
        if len(sub) > 1:
            rej, p_adj, _, _ = multipletests(
                sub["p"].to_numpy(),
                alpha=alpha,
                method=method_code,
            )
            sub["p_adj"] = p_adj
            sub["reject"] = rej
        else:
            sub["p_adj"] = sub["p"].values
            sub["reject"] = sub["p"] < alpha
        outs.append(sub)

    return pd.concat(outs, ignore_index=True)


def pairwise_ame_mnlogit(
    rob: MultinomialResults | MultinomialResultsWrapper,
    factor: str = "method",
    base: str = "qual",
    at: str = "overall",
    alpha: float = 0.05,
    mt_method: str = "holm-sidak",
) -> pd.DataFrame:
    """平均限界効果に基づくカテゴリ水準のペア比較を行います。

    Args:
        rob (MultinomialResults | MultinomialResultsWrapper): ロバスト共分散（任意）を
            含む MNLogit 推定結果。
        factor (str, optional): 対象となるカテゴリ説明変数名。既定は ``"method"`` 。
        base (str, optional): Treatment 符号化における基準水準。既定は ``"qual"`` 。
        at (str, optional): `get_margeff` に渡す ``at`` 引数。既定は ``"overall"`` 。
        alpha (float, optional): 多重比較補正後に用いる有意水準。既定は ``0.05`` 。
        mt_method (str, optional): 多重比較補正の方式。既定は ``"holm-sidak"`` 。

    Returns:
        pandas.DataFrame: 各アウトカム方程式・水準ペアごとの AME 差、標準誤差、
        z 値、p 値、補正後 p 値、棄却判定を含む表。
    """
    # 1) AME を取得（確率差の平均：dummy 変数は離散変化）
    mfx: DiscreteMargins = rob.get_margeff(at=at, method="dydx", dummy=False)

    # statsmodels のバージョン差に対応（summary_frame の列名）
    sf: pd.DataFrame = mfx.summary_frame()
    var_col: str = "exog"
    out_col: str = "endog"
    dy_col: str = "dy/dx"
    se_col: str = "Std. Err."

    # 共分散行列（行順は summary_frame と一致）
    cov: np.ndarray | None = (
        getattr(mfx, "cov", None)
        or getattr(mfx, "cov_margins", None)
        or getattr(mfx, "cov_margeff", None)
    )
    if cov is not None:
        cov = np.asarray(cov)

    # 2) (outcome, level) → 行インデックスの対応を作る
    idx_map: Dict[Tuple[str, str], int] = {}
    rows_sf: pd.DataFrame = sf.reset_index(drop=False)
    for row_index, (_, r) in enumerate(rows_sf.iterrows()):
        name: str = str(r[var_col])
        # 例: "C(method, Treatment('qual'))[T.quan]" → level="quan"
        if name.startswith(f"C({factor}") and "[T." in name and name.endswith("]"):
            level: str = name.split("[T.", 1)[1][:-1]
            idx_map[(str(r[out_col]), level)] = row_index

    # 3) 水準リスト（基準を先頭に）
    levels: List[str] = [base] + sorted({lvl for (_, lvl) in idx_map.keys() if lvl != base})

    # 4) アウトカムごとにペア差を検定
    out_rows: List[Dict[str, float | str]] = []
    outcomes: List[str] = [
        str(x)
        for x in sorted({key[0] for key in idx_map})
    ]
    for outcome in outcomes:
        for ai, a in enumerate(levels):
            for b in levels[ai + 1 :]:
                ia: int | None = idx_map.get((outcome, a))
                ib: int | None = idx_map.get((outcome, b))

                # AME 差と分散
                if ia is None and a == base and ib is not None:
                    # base vs 非基準: 差 = 0 - AME(b)
                    diff: float = -_ensure_float(rows_sf.loc[ib, dy_col])
                    if cov is not None:
                        var: float = _ensure_float(cov[ib, ib])
                    else:
                        sigma_b: float = _ensure_float(rows_sf.loc[ib, se_col])
                        var = sigma_b**2
                elif ib is None and b == base and ia is not None:
                    diff = _ensure_float(rows_sf.loc[ia, dy_col])
                    if cov is not None:
                        var = _ensure_float(cov[ia, ia])
                    else:
                        sigma_a: float = _ensure_float(rows_sf.loc[ia, se_col])
                        var = sigma_a**2
                elif ia is not None and ib is not None:
                    diff = _ensure_float(rows_sf.loc[ia, dy_col]) - _ensure_float(
                        rows_sf.loc[ib, dy_col]
                    )
                    if cov is not None:
                        var = _ensure_float(cov[ia, ia]) + _ensure_float(
                            cov[ib, ib]
                        ) - 2.0 * _ensure_float(cov[ia, ib])
                    else:
                        # 共分散が取れない場合の近似（独立仮定）。厳密には過小評価の恐れがある。
                        sigma_a = _ensure_float(rows_sf.loc[ia, se_col])
                        sigma_b = _ensure_float(rows_sf.loc[ib, se_col])
                        var = sigma_a**2 + sigma_b**2
                else:
                    continue

                se: float = float(np.sqrt(max(var, 0.0)))
                z: float = float(diff / se) if se > 0 else np.nan
                p: float = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else np.nan

                out_rows.append(
                    {
                        "eq": outcome,
                        "level_a": a,
                        "level_b": b,
                        "AME_diff": float(diff),
                        "SE": se,
                        "z": z,
                        "p": p,
                    }
                )

    # 多重補正（各 outcome 内）
    out_df: pd.DataFrame = pd.DataFrame(out_rows)
    outs: List[pd.DataFrame] = []
    method_code = MULTITEST_METHOD_MAP.get(mt_method, mt_method)
    for outcome in out_df["eq"].unique():
        sub: pd.DataFrame = out_df.loc[out_df["eq"] == outcome].copy()
        if len(sub) > 1:
            rej, p_adj, _, _ = multipletests(
                sub["p"].to_numpy(),
                alpha=alpha,
                method=method_code,
            )
            sub["p_adj"] = p_adj
            sub["reject"] = rej
        else:
            sub["p_adj"] = sub["p"]
            sub["reject"] = sub["p"] < alpha
        outs.append(sub)
    return pd.concat(outs, ignore_index=True)
