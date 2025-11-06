# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests
from statsmodels.base.model import LikelihoodModelResults
from scipy.stats import norm


def t_test_pairwise_mnlogit(res, term_name: str,
                                   alpha: float = 0.05,
                                   method: str = "holm-sidak") -> pd.DataFrame:
    """
    MNLogit 用 pairwise t-test（公開 API のみ使用）。
    term_name: 例 "C(method, Treatment('qual'))"
    戻り値: 各式(eq) × 全ペアの検定結果（effect, stderr, t, p, p_adj, reject）
    """
    # 1) 基準水準と因子名を term_name から取得
    m = re.search(r"Treatment\('([^']+)'\)", term_name)
    if not m:
        raise ValueError("term_name から基準水準が取得できません（Treatment('...') を明示してください）")
    base = m.group(1)

    m2 = re.search(r"C\(([^,]+)", term_name)
    factor_name = m2.group(1).strip() if m2 else None  # 使わないが記録

    # 2) この term に属する水準（非基準）を exog_names から抽出（順序は exog_names に従う）
    exog_names = res.model.exog_names
    pat = term_name + "[T."
    levels_nonbase = []
    var_index = {}  # exog名 -> 変数インデックス j
    for j, nm in enumerate(exog_names):
        var_index[nm] = j
        if nm.startswith(pat) and nm.endswith("]"):
            levels_nonbase.append(nm[len(pat):-1])

    if not levels_nonbase:
        raise ValueError(f"指定の term が見つかりません: {term_name}")

    levels = [base] + levels_nonbase

    # 3) パラメタ並びに合わせた位置写像（変数 j, 式 eq -> フラット位置）
    k_vars, k_eq = res.params.shape
    k_params = res.cov_params().shape[0]
    assert k_vars * k_eq == k_params, "係数次元の不一致"

    def pos(var_name: str, eq: int) -> int:
        j = var_index[var_name]
        return j + eq * k_vars

    # 4) この term に属する exog 名の一覧（拡張があっても頑健に）
    term_vars = [nm for nm in exog_names if nm == term_name or nm.startswith(term_name + "[")]

    # 5) ペア比較を式ごとに実行（係数差の t 検定）。ロバスト共分散は res 側設定が反映される
    rows = []
    for a_i, a in enumerate(levels):
        for b in levels[a_i+1:]:
            for eq in range(k_eq):
                L = np.zeros((1, k_params))
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

                tr = res.t_test(L)  # effect, sd, tvalue, pvalue を持つ
                rows.append({
                    "eq": eq,
                    "level_a": a,
                    "level_b": b,
                    "effect": float(tr.effect),
                    "stderr": float(tr.sd),
                    "t": float(tr.tvalue),
                    "p": float(tr.pvalue),
                })

    out = pd.DataFrame(rows)

    # 6) 式ごとに多重比較補正
    outs = []
    for eq in range(k_eq):
        sub = out.loc[out["eq"] == eq].copy()
        # ペアが1つだけなら multipletests は不要
        if len(sub) > 1:
            rej, p_adj, _, _ = multipletests(sub["p"].to_numpy(), alpha=alpha, method="hs")
            sub["p_adj"] = p_adj
            sub["reject"] = rej
        else:
            sub["p_adj"] = sub["p"].values
            sub["reject"] = sub["p"] < alpha
        outs.append(sub)

    return pd.concat(outs, ignore_index=True)


def pairwise_ame_mnlogit(
    rob, factor="method", base="qual", at="overall", alpha=0.05, mt_method="holm-sidak"
) -> pd.DataFrame:
    """
    MNLogit の平均限界効果(AME)に基づくペア比較。
    rob : MultinomialResults（ロバスト共分散つきでも可）
    factor : カテゴリ説明変数名（例 "method"）
    base : Treatment の基準水準
    """
    # 1) AME を取得（確率差の平均：dummy 変数は離散変化）
    mfx = rob.get_margeff(at=at, method="dydx", dummy=False)

    # statsmodels のバージョン差に対応（summary_frame の列名）
    sf = mfx.summary_frame()
    var_col = "exog"
    out_col = "endog"
    dy_col  = "dy/dx"
    se_col  = "Std. Err."

    # 共分散行列（行順は summary_frame と一致）
    cov = (
        getattr(mfx, "cov", None)
        or getattr(mfx, "cov_margins", None)
        or getattr(mfx, "cov_margeff", None)
    )
    if cov is not None:
        cov = np.asarray(cov)

    # 2) (outcome, level) → 行インデックスの対応を作る
    idx_map = {}
    rows_sf = sf.reset_index(drop=False)
    for i, r in rows_sf.iterrows():
        name = str(r[var_col])
        # 例: "C(method, Treatment('qual'))[T.quan]" → level="quan"
        if name.startswith(f"C({factor}") and "[T." in name and name.endswith("]"):
            level = name.split("[T.", 1)[1][:-1]
            idx_map[(str(r[out_col]), level)] = i

    # 3) 水準リスト（基準を先頭に）
    levels = [base] + sorted({lvl for (_, lvl) in idx_map.keys() if lvl != base})

    # 4) アウトカムごとにペア差を検定
    out_rows = []
    outcomes = [str(x) for x in sorted({k[0] for k in idx_map.keys()}, key=lambda x: x)]
    for outcome in outcomes:
        for ai, a in enumerate(levels):
            for b in levels[ai + 1 :]:
                ia = idx_map.get((outcome, a))
                ib = idx_map.get((outcome, b))

                # AME 差と分散
                if ia is None and a == base and ib is not None:
                    # base vs 非基準: 差 = 0 - AME(b)
                    diff = -rows_sf.loc[ib, dy_col]
                    if cov is not None:
                        var = cov[ib, ib]
                    else:
                        var = rows_sf.loc[ib, se_col] ** 2
                elif ib is None and b == base and ia is not None:
                    diff = rows_sf.loc[ia, dy_col]
                    if cov is not None:
                        var = cov[ia, ia]
                    else:
                        var = rows_sf.loc[ia, se_col] ** 2
                elif ia is not None and ib is not None:
                    diff = rows_sf.loc[ia, dy_col] - rows_sf.loc[ib, dy_col]
                    if cov is not None:
                        var = cov[ia, ia] + cov[ib, ib] - 2 * cov[ia, ib]
                    else:
                        # 共分散が取れない場合の近似（独立仮定）。厳密には過小評価の恐れがある。
                        var = rows_sf.loc[ia, se_col] ** 2 + rows_sf.loc[ib, se_col] ** 2
                else:
                    continue

                se = float(np.sqrt(max(var, 0.0)))
                z = float(diff / se) if se > 0 else np.nan
                p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else np.nan

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
    out_df = pd.DataFrame(out_rows)
    outs = []
    for outcome in out_df["eq"].unique():
        sub = out_df.loc[out_df["eq"] == outcome].copy()
        if len(sub) > 1:
            rej, p_adj, _, _ = multipletests(sub["p"].to_numpy(), alpha=alpha, method="hs" if mt_method is None else {"holm-sidak":"hs","bonferroni":"bonf","fdr_bh":"fdr_bh"}.get(mt_method, "hs"))
            sub["p_adj"] = p_adj
            sub["reject"] = rej
        else:
            sub["p_adj"] = sub["p"]
            sub["reject"] = sub["p"] < alpha
        outs.append(sub)
    return pd.concat(outs, ignore_index=True)

# ------------------------------------------------------------
# AME（get_margeff）のブロックWaldを行う最小アダプタと関数群
# ------------------------------------------------------------

class _VecResults:
    """
    LikelihoodModelResults.wald_test を流用するための極小結果オブジェクト。
    params と cov_params() だけを実装する。
    """
    def __init__(self, theta: np.ndarray, Sigma: np.ndarray, names: list[str] | None = None):
        self.params = np.asarray(theta)
        self._cov = np.asarray(Sigma)
        self.df_resid = np.inf
        self.use_t = False  # AMEは大標本のカイ二乗で扱う
        class _Model: pass
        class _Data: pass
        self.model = _Model()
        self.model.data = _Data()
        self.model.data.cov_names = (names if names is not None
                                     else [f"m{i}" for i in range(self.params.size)])

    def cov_params(self, r_matrix=None, cov_p=None):
        return self._cov


def _mfx_and_frame(rob, at: str = "overall"):
    """
    MultinomialResults -> (DiscreteMargins, summary_frame(DataFrame), cov(2D))
    """
    mfx = rob.get_margeff()
    sf = mfx.summary_frame()  # 列例：["equation","variable","dy/dx",...]
    # 共分散は statsmodels バージョンで属性名が異なることがある
    Sigma_full = getattr(mfx, "cov", None) or getattr(mfx, "margeff_cov", None)
    if Sigma_full is None:
        raise RuntimeError("DiscreteMargins の共分散が取得できません。")
    return mfx, sf, np.asarray(Sigma_full)


def _extract_block(theta_frame: pd.DataFrame,
                   Sigma_full: np.ndarray,
                   row_index: np.ndarray):
    """summary_frame の行インデックスに対応する (θ, Σ, names) を返す。"""
    theta = theta_frame.loc[row_index, "dy/dx"].to_numpy()
    Sigma = Sigma_full[np.ix_(row_index, row_index)]
    names = theta_frame.loc[row_index, "variable"].astype(str).tolist()
    return theta, Sigma, names


def wald_test_margeff(rob,
                      factor: str = "method",
                      base: str = "qual",
                      at: str = "overall",
                      eq_alias: dict[int, str] | None = None,
                      var_prefix: str | None = None) -> pd.DataFrame:
    """
    AME の「式ごとの factor 主効果（基準を除くカテゴリ）= 0」を
    同時に検定する（ブロックWald, χ^2）。

    Parameters
    ----------
    rob : MultinomialResults（ロバスト共分散付きが望ましい）
    factor : str
        例: "method"
    base : str
        例: "qual"（Treatment(base) の基準）
    at : {"overall", ...}
        get_margeff の 'at' 引数
    eq_alias : {int -> str}, optional
        出力で式番号に人間可読名を付けるマップ
    var_prefix : str, optional
        AME の summary_frame["variable"] を事前フィルタする接頭辞。
        省略時は C(factor, Treatment(base)) を自動生成。

    Returns
    -------
    DataFrame with columns:
        ["eq", "k", "chi2", "df", "pvalue"]
    """
    _, sf, Sigma_full = _mfx_and_frame(rob, at=at)

    # variable 名の接頭辞を決める（主効果）
    if var_prefix is None:
        var_prefix = f"C({factor}, Treatment('{base}'))["  # 例: C(method, Treatment('qual'))[T.quan]

    print(sf)
    # 「式」一覧（summary_frame の 'equation', 'eq', or 'endog' などを自動検出）
    eq_col = "equation"

    eq_values = pd.Index(sorted(sf[eq_col].unique()))
    out_rows = []
    
    for eq in eq_values:
        mask = (sf[eq_col] == eq) & sf["variable"].astype(str).str.startswith(var_prefix)
        row_idx = sf.index[mask].to_numpy()
        if row_idx.size == 0:
            # この式に該当する AME がない（=基準のみ等）ならスキップ
            continue

        theta, Sigma, names = _extract_block(sf, Sigma_full, row_idx)
        k = len(theta)
        R = np.eye(k)

        # LikelihoodModelResults.wald_test を「外部共分散 cov_p 指定」で呼ぶ
        vec = _VecResults(theta, Sigma, names)
        wt = LikelihoodModelResults.wald_test(vec, r_matrix=R, cov_p=Sigma,
                                              use_f=False, scalar=False)  # χ^2

        chi2 = float(np.asarray(wt.statistic))
        pval = float(np.asarray(wt.pvalue))
        eq_name = eq_alias.get(eq, eq) if eq_alias else eq
        out_rows.append({"eq": eq_name, "k": k, "chi2": chi2,
                         "df": int(wt.df_constraints), "pvalue": pval})

    return pd.DataFrame(out_rows).sort_values("eq").reset_index(drop=True)
