"""個人層別化の適格性を測る指標。

4 つの評価軸のうち、本コホートで測れるのは 3 つ。
  内部整合性   : 構成遺伝子が個人間で共変動するか
  順位一貫性   : 同一個人を別の摂動で測ったとき順位が保たれるか
  条件効果     : 摂動で協調的に動くか（従来のパスウェイ解析が測ってきたもの）
反復測定信頼性（test-retest / ICC）は同一条件の反復測定が必要で、本コホートには
存在しない（config/datasets.yml notes 参照）。外部コホートで別途評価する。

内部整合性は「遺伝子ごとに個人方向の順位を取って標準化した行列」の内積で計算する。
Spearman 相関は順位に対する Pearson 相関なので、標準化済み順位行列 S に対して
corr(i, j) = (S_i . S_j) / n が成り立つ。これで数千セットを実用時間で処理できる。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_pairwise_rho(S_sub: np.ndarray) -> float:
    """セット内の遺伝子ペア平均 Spearman 相関（個人間の共変動の強さ）。"""
    g, n = S_sub.shape
    if g < 2:
        return np.nan
    ok = ~np.isnan(S_sub).any(axis=1)
    S_sub = S_sub[ok]
    g = S_sub.shape[0]
    if g < 2:
        return np.nan
    c = (S_sub @ S_sub.T) / n
    return float((c.sum() - np.trace(c)) / (g * (g - 1)))


def cronbach_alpha(S_sub: np.ndarray) -> float:
    """標準化済み行列に対する Cronbach の alpha。

    各行の分散が 1 なので alpha = g/(g-1) * (1 - g / var(合計)) になる。
    「セットを 1 つの尺度として見たときの信頼性」を古典的テスト理論の言葉で出す。
    """
    ok = ~np.isnan(S_sub).any(axis=1)
    S_sub = S_sub[ok]
    g = S_sub.shape[0]
    if g < 2:
        return np.nan
    total_var = S_sub.sum(axis=0).var(ddof=0)
    if total_var <= 0:
        return np.nan
    return float(g / (g - 1) * (1 - g / total_var))


def split_half_reliability(
    S_sub: np.ndarray, repeats: int, generator: np.random.Generator
) -> tuple[float, float]:
    """セットを 2 分割して作った 2 つのスコアの一致度。

    返り値は (生の相関の平均, Spearman-Brown 補正値)。
    セットが「同じものを測る項目の集まり」なら高く、寄せ集めなら低くなる。
    """
    ok = ~np.isnan(S_sub).any(axis=1)
    S_sub = S_sub[ok]
    g = S_sub.shape[0]
    if g < 4:
        return np.nan, np.nan

    rs = []
    half = g // 2
    for _ in range(repeats):
        perm = generator.permutation(g)
        a = S_sub[perm[:half]].mean(axis=0)
        b = S_sub[perm[half : 2 * half]].mean(axis=0)
        if a.std() == 0 or b.std() == 0:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    if not rs:
        return np.nan, np.nan
    r = float(np.mean(rs))
    sb = float(2 * r / (1 + r)) if r > -1 else np.nan
    return r, sb


def condition_effect(
    resting: np.ndarray, perturbed: np.ndarray
) -> dict[str, float]:
    """対応ありの条件効果。スコアは条件をまたいだ共通基準で作られている前提。"""
    delta = perturbed - resting
    ok = ~np.isnan(delta)
    delta = delta[ok]
    if delta.size < 10 or np.allclose(delta, 0):
        return {"delta_mean": np.nan, "delta_p": np.nan, "cohens_d": np.nan}
    try:
        p = float(stats.wilcoxon(delta)[1])
    except ValueError:
        p = np.nan
    sd = delta.std(ddof=1)
    return {
        "delta_mean": float(delta.mean()),
        "delta_p": p,
        "cohens_d": float(delta.mean() / sd) if sd > 0 else np.nan,
    }


def direction_concordance(
    resting_expr: np.ndarray, perturbed_expr: np.ndarray
) -> dict[str, float]:
    """構成遺伝子が摂動で同じ向きに動くか（従来のパスウェイ解析が見ている性質）。"""
    delta = np.nanmedian(perturbed_expr - resting_expr, axis=1)
    delta = delta[~np.isnan(delta)]
    g = delta.size
    if g < 2:
        return {"n_genes_tested": g, "frac_same_direction": np.nan, "direction_p": np.nan}
    up = int((delta > 0).sum())
    same = max(up, g - up)
    p = float(stats.binomtest(same, g, 0.5, alternative="greater").pvalue)
    return {
        "n_genes_tested": g,
        "frac_same_direction": same / g,
        "direction_p": p,
    }


def rank_consistency(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """2 つの測定間で個人の順位が保たれるか。"""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size < 10 or a.std() == 0 or b.std() == 0:
        return {"rho": np.nan, "rho_p": np.nan, "tertile_swap": np.nan}
    rho, p = stats.spearmanr(a, b)
    lo_a, hi_a = np.quantile(a, [1 / 3, 2 / 3])
    lo_b, hi_b = np.quantile(b, [1 / 3, 2 / 3])
    ta = np.digitize(a, [lo_a, hi_a])
    tb = np.digitize(b, [lo_b, hi_b])
    extreme = (ta != 1) | (tb != 1)
    swap = float(np.mean(np.abs(ta[extreme] - tb[extreme]) == 2)) if extreme.any() else np.nan
    return {"rho": float(rho), "rho_p": float(p), "tertile_swap": swap}


def icc_two_way(y: np.ndarray) -> float:
    """ICC(2,1)：二元配置変量モデル・絶対一致・単一測定。

    y は 個人 x 測定回 の行列。順位相関と違い、測定回の間の水準のずれを
    不一致として扱う（同じ順位でも全体が上下していれば下がる）。
    """
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y).any(axis=1)
    y = y[ok]
    n, k = y.shape
    if n < 3 or k < 2:
        return np.nan
    grand = y.mean()
    ss_rows = k * float(((y.mean(axis=1) - grand) ** 2).sum())
    ss_cols = n * float(((y.mean(axis=0) - grand) ** 2).sum())
    ss_total = float(((y - grand) ** 2).sum())
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom == 0:
        return np.nan
    return float((ms_rows - ms_err) / denom)


def matched_random_sets(
    genes: list[str],
    decile_of_gene: dict[str, int],
    genes_by_decile: dict[int, list[str]],
    n_sets: int,
    generator: np.random.Generator,
) -> list[list[str]]:
    """発現量の分位をそろえたランダムセットを作る（陰性対照）。

    サイズだけ合わせたランダムセットでは「高発現遺伝子ほど相関が出やすい」という
    交絡が残るため、遺伝子ごとに同じ発現分位から置き換える。
    """
    out = []
    for _ in range(n_sets):
        pick = []
        for g in genes:
            d = decile_of_gene.get(g)
            pool = genes_by_decile.get(d)
            if not pool:
                continue
            pick.append(pool[generator.integers(len(pool))])
        out.append(pick)
    return out


def empirical_null(observed: float, null_values: list[float]) -> dict[str, float]:
    """観測値を対照分布と比べる。

    empirical p は対照セット数 B に対して 1/(B+1) より小さくならないため、
    数千セットに FDR をかけると全滅する。そこで正規近似の p も併記し、
    多重比較補正にはそちらを使う（B は分散推定にだけ使う）。
    """
    vals = np.array([v for v in null_values if not np.isnan(v)], dtype=float)
    if vals.size < 5 or np.isnan(observed):
        return {
            "null_mean": np.nan, "null_sd": np.nan, "null_z": np.nan,
            "null_p_empirical": np.nan, "null_p": np.nan,
        }
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    z = (observed - mu) / sd if sd > 0 else np.nan
    return {
        "null_mean": mu,
        "null_sd": sd,
        "null_z": z,
        "null_p_empirical": float((np.sum(vals >= observed) + 1) / (vals.size + 1)),
        "null_p": float(stats.norm.sf(z)) if np.isfinite(z) else np.nan,
    }


def alpha_from_mean_rho(mean_rho: float, n_items: int) -> float:
    """平均項目間相関とセットサイズから Cronbach alpha を解析的に出す。

    alpha = g*r / (1 + (g-1)*r)。サイズが大きいほど r が小さくても alpha が上がる
    ため、「alpha が高い = まとまった遺伝子セット」とは言えない。ランダムセットでも
    どこまで alpha が出るかを示すのに使う。
    """
    if np.isnan(mean_rho) or n_items < 2:
        return np.nan
    denom = 1 + (n_items - 1) * mean_rho
    if denom == 0:
        return np.nan
    return float(n_items * mean_rho / denom)


def pooled_set_score(z: np.ndarray) -> np.ndarray:
    """共通基準で z 化済みの行列に対するセットスコア（条件間差を保つ）。

    ScoringContext は行を条件内で再標準化するので、条件効果の測定には使えない。
    """
    return np.nanmean(z, axis=0)
