# -*- coding: utf-8 -*-
import re
import itertools
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
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
