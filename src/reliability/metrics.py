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


def mean_pairwise_rho_fast(S: np.ndarray, idx: np.ndarray) -> float:
    """S の idx 行に対する平均ペア Spearman 相関を O(gn) で出す。

    S は行ごとに順位を取って標準化した行列なので、各行の分散は ddof=0 で厳密に 1 に
    なる。したがって c = S S^T / n の対角は 1 であり、
        平均ペア相関 = (c.sum() - g) / (g(g-1)) = (||sum_i S_i||^2 / n - g) / (g(g-1))
    が成り立つ。O(g^2 n) の行列積が O(gn) の行和になる。対照を 20 個から 10,000 個に
    増やすためにこの式が必要で、同値性は tests/test_high_resolution_null.py で検査する。

    この式が使えるのは S に NaN がない場合だけである（順位は必ず分散を持つので
    実際には NaN が出ない）。呼び出し側で検査すること。
    """
    g = idx.size
    if g < 2:
        return np.nan
    acc = S[idx].sum(axis=0)
    return (float(acc @ acc) / S.shape[1] - g) / (g * (g - 1))


def draw_null_rho(
    S: np.ndarray,
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    n_draw: int,
    generator: np.random.Generator,
    chunk: int = 2_000,
) -> np.ndarray:
    """分位をそろえたランダムセットの平均ペア相関を n_draw 個まとめて作る。

    matched_random_sets と同じ抽出（遺伝子の位置ごとに同じ分位のプールから復元抽出）を
    行番号で行い、(かたまり, 個人) の作業配列に位置ごとの行を足し込む。
    行列を実体化しないのでメモリは chunk x 個人数で決まる。
    """
    g = decile_of_position.size
    n = S.shape[1]
    out = np.empty(n_draw, dtype=np.float64)
    filled = 0
    while filled < n_draw:
        b = min(chunk, n_draw - filled)
        acc = np.zeros((b, n), dtype=np.float64)
        for j in range(g):
            pool = pool_by_decile[int(decile_of_position[j])]
            acc += S[pool[generator.integers(pool.size, size=b)]]
        out[filled : filled + b] = (
            np.einsum("bn,bn->b", acc, acc) / n - g
        ) / (g * (g - 1))
        filled += b
    return out


def draw_null_condition_effect(
    delta_z: np.ndarray,
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    n_draw: int,
    generator: np.random.Generator,
    chunk: int = 2_000,
) -> np.ndarray:
    """分位をそろえたランダムセットの条件効果（対応あり Cohen の d）を n_draw 個作る。

    なぜ要るか
      条件効果は「条件間の差がゼロか」を問い、内部整合性は「ランダムセットより高いか」を
      問う。帰無仮説の違うものを交差させると、割合の差の一部は判定基準の非対称性から
      生じる。条件効果も同じ対照に対する超過として測れば、同じ尺度で並べられる。

    セットスコアは構成遺伝子の z の平均なので、セットの個人ごとの条件間差は
    delta_z（遺伝子 x 個人の条件間差）の該当行の列平均になる。draw_null_rho と同じ
    足し込み方式を使うため、行列を実体化せずメモリは chunk x 個人数で決まる。
    """
    g = decile_of_position.size
    n = delta_z.shape[1]
    out = np.empty(n_draw, dtype=np.float64)
    filled = 0
    while filled < n_draw:
        b = min(chunk, n_draw - filled)
        acc = np.zeros((b, n), dtype=np.float64)
        for j in range(g):
            pool = pool_by_decile[int(decile_of_position[j])]
            acc += delta_z[pool[generator.integers(pool.size, size=b)]]
        d = acc / g
        sd = d.std(axis=1, ddof=1)
        out[filled : filled + b] = np.where(sd > 0, d.mean(axis=1) / sd, np.nan)
        filled += b
    return out


def draw_null_direction(
    gene_delta: np.ndarray,
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    n_draw: int,
    generator: np.random.Generator,
    chunk: int = 2_000,
) -> np.ndarray:
    """ランダムセットの「同じ向きに動く遺伝子の割合」を n_draw 個まとめて作る。

    なぜ要るか
      direction_concordance は注釈セットの構成遺伝子が摂動で同方向に動くかを測るが、
      対照がない。摂動が転写全体を一方向に押すなら、どの遺伝子を集めても同方向性は
      高く出る。条件効果と同じ理由で、対照に対する超過として見ないと、この性質が
      注釈セット固有かどうかは判定できない。

    gene_delta は遺伝子ごとの中位数差（摂動後 - 安静時）。符号だけを使う。
    """
    g = decile_of_position.size
    sign_up = gene_delta > 0
    out = np.empty(n_draw, dtype=np.float64)
    filled = 0
    while filled < n_draw:
        b = min(chunk, n_draw - filled)
        up = np.zeros(b, dtype=np.int64)
        for j in range(g):
            pool = pool_by_decile[int(decile_of_position[j])]
            up += sign_up[pool[generator.integers(pool.size, size=b)]]
        out[filled : filled + b] = np.maximum(up, g - up) / g
        filled += b
    return out


def pools_as_rows(
    genes_by_decile: dict[int, list[str]], gene_index: dict[str, int]
) -> dict[int, np.ndarray]:
    """分位ごとの抽出プールを、遺伝子名ではなく行列の行番号で持ち直す。"""
    return {
        int(d): np.array([gene_index[g] for g in gs if g in gene_index], dtype=np.int64)
        for d, gs in genes_by_decile.items()
    }


def draw_index_matrix(
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    n_draw: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """(対照数, 遺伝子位置) の行番号行列。位置ごとに同じ分位のプールから復元抽出する。

    既存の matched_random_sets と同じ抽出（位置ごと・復元あり）を、遺伝子名ではなく
    行番号で行う。位置の順に乱数を消費するので、抽出の流れも元実装と同じ順序になる。
    """
    out = np.empty((n_draw, decile_of_position.size), dtype=np.int64)
    for j in range(decile_of_position.size):
        pool = pool_by_decile[int(decile_of_position[j])]
        out[:, j] = pool[generator.integers(pool.size, size=n_draw)]
    return out


def null_rho_multi(
    mats: dict[str, np.ndarray],
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    n_draw: int,
    generator: np.random.Generator,
    chunk: int = 2_000,
) -> dict[str, np.ndarray]:
    """同一の対照セット群を複数の行列で評価する。

    batch_check は「技術要因を抜く前後」を同じ対照セットで比べる。自由度を落とす補正は
    対照側の相関も下げるので、条件ごとに対照を引き直すと比較の意味が変わる。
    そのため対照の行番号行列を 1 回引き、それを各行列に当てる。
    consistency_definition の Spearman/Pearson 比較も同じ理由でここを通す。

    どの行列も行ごとに標準化されている（分散が厳密に 1）ことを前提にする。
    """
    acc: dict[str, list[np.ndarray]] = {k: [] for k in mats}
    filled = 0
    while filled < n_draw:
        take = min(chunk, n_draw - filled)
        idx = draw_index_matrix(pool_by_decile, decile_of_position, take, generator)
        for k, m in mats.items():
            acc[k].append(mean_pairwise_rho_batch(m, idx))
        filled += take
    return {k: np.concatenate(v) for k, v in acc.items()}


def null_abs_rho_with_target(
    mats: dict[str, np.ndarray],
    target: np.ndarray,
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    n_draw: int,
    generator: np.random.Generator,
    chunk: int = 2_000,
) -> dict[str, np.ndarray]:
    """対照セットのスコアと外部変数の |Spearman| をまとめて作る。

    phenotype_check が「表現型との相関がランダムセットの床を超えるか」を測るのに使う。
    2 時点を同じ対照セット群で評価するため、行番号行列は 1 回だけ引く
    （時点ごとに引き直すと「どちらの測定回でも同じ床か」の比較が崩れる）。
    """
    assert not np.isnan(target).any(), "target に NaN がある"
    acc: dict[str, list[np.ndarray]] = {k: [] for k in mats}
    filled = 0
    while filled < n_draw:
        take = min(chunk, n_draw - filled)
        idx = draw_index_matrix(pool_by_decile, decile_of_position, take, generator)
        g = idx.shape[1]
        for k, Z in mats.items():
            s = np.zeros((take, Z.shape[1]), dtype=np.float64)
            for j in range(g):
                s += Z[idx[:, j]]
            s /= g
            acc[k].append(
                np.abs(spearman_batch(s, np.broadcast_to(target, s.shape)))
            )
        filled += take
    return {k: np.concatenate(v) for k, v in acc.items()}


def icc_two_way_batch(y: np.ndarray) -> np.ndarray:
    """ICC(2,1) を (対照, 個人, 測定回) の配列に対してまとめて計算する。

    metrics.icc_two_way と同じ式。NaN がないことを前提にする（呼び出し側で検査）。
    """
    b, n, k = y.shape
    grand = y.mean(axis=(1, 2), keepdims=True)
    ss_rows = k * ((y.mean(axis=2, keepdims=True) - grand) ** 2).sum(axis=(1, 2))
    ss_cols = n * ((y.mean(axis=1, keepdims=True) - grand) ** 2).sum(axis=(1, 2))
    ss_total = ((y - grand) ** 2).sum(axis=(1, 2))
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ms_rows - ms_err) / denom
    return np.where(denom == 0, np.nan, out)


def spearman_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行ごとの Spearman 相関。順位に直してから Pearson を取る。"""
    ra = stats.rankdata(a, axis=1)
    rb = stats.rankdata(b, axis=1)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den == 0, np.nan, num / den)


def mean_pairwise_rho_batch(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(対照, 位置) の行番号行列に対する平均ペア Spearman 相関（行和による O(gn) 版）。"""
    b, g = idx.shape
    n = S.shape[1]
    acc = np.zeros((b, n), dtype=np.float64)
    for j in range(g):
        acc += S[idx[:, j]]
    return (np.einsum("bn,bn->b", acc, acc) / n - g) / (g * (g - 1))


def null_batch(
    idx: np.ndarray, z_a: np.ndarray, z_b: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """対照行番号行列に対する (平均ペア相関, ICC, 反復間 Spearman)。"""
    b, g = idx.shape
    acc_a = np.zeros((b, z_a.shape[1]), dtype=np.float64)
    acc_b = np.zeros((b, z_b.shape[1]), dtype=np.float64)
    for j in range(g):
        acc_a += z_a[idx[:, j]]
        acc_b += z_b[idx[:, j]]
    acc_a /= g
    acc_b /= g
    icc = icc_two_way_batch(np.stack([acc_a, acc_b], axis=2))
    rho = spearman_batch(acc_a, acc_b)
    ic = mean_pairwise_rho_batch(S, idx)
    return ic, icc, rho


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

    empirical p は対照セット数 B に対して 1/(B+1) より小さくならない。
    B=20 では下限が 0.048 で、数千セットに BH-FDR をかけると全滅するため、
    旧版は正規近似の p を作って多重比較補正に使っていた。しかし平均ペア Spearman の
    帰無分布は有界で右に歪み（歪度の中位数 0.99）、ICC の帰無分布は左に歪む
    （同 -0.22）。正規近似で裾の確率を出す根拠は弱く、方向も指標によって違う。

    そこで B を 10,000 に増やし、**判定には null_p_empirical を使う**。
    下限が 1/10001 になり、BH の最小閾値（alpha / 検定数）より十分下がる。
    null_p（正規近似）は旧版との比較のために返すが、判定には使わない。
    null_skew は正規近似がどれだけ外れるかの診断値として残す。
    """
    vals = np.array([v for v in null_values if not np.isnan(v)], dtype=float)
    if vals.size < 5 or np.isnan(observed):
        return {
            "null_mean": np.nan, "null_sd": np.nan, "null_z": np.nan,
            "null_skew": np.nan, "n_control": int(vals.size),
            "null_p_empirical": np.nan, "null_p": np.nan,
        }
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    z = (observed - mu) / sd if sd > 0 else np.nan
    return {
        "null_mean": mu,
        "null_sd": sd,
        "null_z": z,
        "null_skew": float(((vals - mu) ** 3).mean() / sd**3) if sd > 0 else np.nan,
        "n_control": int(vals.size),
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
