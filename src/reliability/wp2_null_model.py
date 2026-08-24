"""WP2 の帰無モデル。シグネチャと対照の交換可能性を検定する。

なぜこのモジュールが要るか
  WP2 の主要統計量は「40 シグネチャについての Spearman 相関」である。
  事前登録の初期草案は帰無モデルを「40 x 20 = 800 の対照セットで同じ相関を計算する」
  と書いていた。20 個は各シグネチャの枠に入れ子で、同じサイズ・同じ発現量分位から
  引かれているので独立ではない。n を 20 倍に膨らませた相関になり、
  標準誤差は約 sqrt(20) = 4.5 倍過小評価される。**主要統計量と単位が違う。**

  最初の修正案は「各枠から対照を 1 つ引いて n = 40 の相関を作る」というものだったが、
  合成データで検査したところ**それでも偏る**ことが分かった。理由は散らばりの非対称性で、
  シグネチャの z は対照 20 個を基準にしているので大きく外れうる（実測で z = 12 に達する）
  一方、対照の leave-one-out z は構成上ほぼ標準正規に収まる。Spearman は信号対雑音比で
  決まるので、z の幅が広い側が systematically 高い相関を示す。
  シグネチャ固有の効果がゼロの合成データで経験的 p = 0.002 が出た。

  そこで帰無を**交換可能性検定**に組み替えた。各枠にはシグネチャ 1 個と対照 20 個、
  合わせて 21 個の遺伝子集合がある。帰無仮説「そのシグネチャはサイズと発現量をそろえた
  ランダム集合と交換可能である」の下では、どれを「シグネチャ」と呼ぶかは任意である。
  そこで各枠から 21 個のうち 1 個を選び、**残り 20 個を基準に標準化する**。
  観測値は「常に index 0（実際のシグネチャ）を選んだ 1 つの引き方」にあたる。
  標準化の手続きが観測側と帰無側で完全に同一になるので、散らばりの非対称性が消える。

登録前に凍結する
  事前登録が「seed fixed in the frozen code」と書いているので、
  このファイルの sha256 を登録に記載する。登録後に中身を変えない。
  実データの説明変数では走らせない（説明変数は登録後に初めて計算する）。
  合成データでの検査は tests/test_wp2_null_model.py にある。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

N_REPLICATES = 10_000
SEED = 20260820          # 事前登録に記載する乱数種。変更しない
SIGNATURE_INDEX = 0      # 各枠の 0 番目が実際のシグネチャ、1 以降が対照


def excess_z(values: np.ndarray, index: int) -> float:
    """values[index] を、それ以外を基準に標準化する。

    観測側（シグネチャを対照 20 個で標準化）と帰無側（対照 1 個を残り 20 個で標準化）で
    まったく同じ手続きを使うためにここに一本化する。基準側が 3 個未満、または
    基準側の標準偏差が 0 なら nan を返す（判定不能）。
    """
    x = np.asarray(values, dtype=np.float64)
    rest = np.delete(x, index)
    rest = rest[np.isfinite(rest)]
    if len(rest) < 3:
        return float("nan")
    sd = rest.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float((x[index] - rest.mean()) / sd)


def slot_statistic(coherence: list[np.ndarray], target: list[np.ndarray],
                   picks: list[int]) -> float:
    """枠ごとに 1 個ずつ選んで、選んだ集合について Spearman 相関を計算する。

    coherence[i] / target[i] は枠 i の 21 個（シグネチャ + 対照 20）の整合性と目的変数。
    picks[i] はその枠で「シグネチャ役」に選ぶ添字。
    """
    zs, ys = [], []
    for coh, y, j in zip(coherence, target, picks):
        z = excess_z(coh, j)
        yv = float(np.asarray(y, dtype=np.float64)[j])
        if np.isfinite(z) and np.isfinite(yv):
            zs.append(z)
            ys.append(yv)
    if len(zs) < 3:
        return float("nan")
    return float(spearmanr(zs, ys).statistic)


def observed_statistic(coherence: list[np.ndarray], target: list[np.ndarray]) -> float:
    """実際のシグネチャ（各枠の 0 番目）についての主要統計量。"""
    return slot_statistic(coherence, target, [SIGNATURE_INDEX] * len(coherence))


def exchangeability_null(coherence: list[np.ndarray], target: list[np.ndarray],
                         n_replicates: int = N_REPLICATES,
                         seed: int = SEED) -> np.ndarray:
    """交換可能性の帰無分布。

    各枠から 21 個のうち 1 個を一様に選び、残り 20 個を基準に標準化して
    n = 枠数 の相関を計算する。これを n_replicates 回繰り返す。
    観測値はこの手続きで「全枠 index 0」を選んだ場合にあたるので、
    帰無分布と観測値は同一の構成法から出ている。
    """
    assert len(coherence) == len(target), "枠の数が合わない"
    sizes = [len(np.asarray(c)) for c in coherence]
    assert min(sizes) >= 4, "1 枠に 4 個以上（シグネチャ + 対照 3 個）必要"
    gen = np.random.default_rng(seed)
    out = np.empty(n_replicates)
    for b in range(n_replicates):
        picks = [int(gen.integers(n)) for n in sizes]
        out[b] = slot_statistic(coherence, target, picks)
    return out


def range_matched_control_null(coherence: list[np.ndarray], target: list[np.ndarray],
                               n_replicates: int = N_REPLICATES,
                               seed: int = SEED) -> tuple[np.ndarray, int]:
    """対照だけを、シグネチャの z の範囲にそろえて評価する。

    なぜ交換可能性検定だけでは足りないか
      交換可能性検定は「シグネチャは対照と入れ替えても同じか」を見る。しかし
      **同じ関係式 y = f(z) が両者に成り立っていて、シグネチャの z の幅だけが広い**
      という状況は捕まえられない。Spearman は信号対雑音比で決まるので、z の幅が広い側が
      systematically 高い相関を示すためである。合成データで、対照にも同じ効果を載せた
      世界（共分散構造の性質であってシグネチャの性質ではない世界）でも
      経験的 p = 0.002 が出た。

    どうするか、そしてこれでは足りないこと
      シグネチャの z が張る区間 [min, max] に入る対照だけを候補にし、
      各枠から 1 つ引いて n = 枠数 の相関を作る。区間内の対照が無い枠は除く。

      **これは判定には使わない。記述的な診断にとどめる。** 合成データで確かめたところ、
      シグネチャの z の範囲は対照の範囲をほぼ包含するので、区間で切っても
      散らばり（信号対雑音比）の差が残り、共有効果の世界でも観測値が高く出た
      （範囲そろえでも p = 0.002）。**区間の一致は散らばりの一致ではない。**
      この残余は 20 個の対照からは解けないので、事前登録では限界として明記し、
      シグネチャと対照の z の分布を並べて報告する規則にした。

    返り値は (相関の分布, 1 反復あたりに使えた枠数の中央値)。
    """
    sig_z = np.array([excess_z(c, SIGNATURE_INDEX) for c in coherence])
    finite = sig_z[np.isfinite(sig_z)]
    assert len(finite) >= 3, "シグネチャ側の z が足りない"
    lo, hi = float(finite.min()), float(finite.max())

    # 枠ごとに、区間内に入る対照の (z, y) を集めておく
    pools = []
    for coh, y in zip(coherence, target):
        c = np.asarray(coh, dtype=np.float64)
        yv = np.asarray(y, dtype=np.float64)
        cand = []
        for j in range(1, len(c)):          # index 0 はシグネチャなので除く
            z = excess_z(c, j)
            if np.isfinite(z) and lo <= z <= hi and np.isfinite(yv[j]):
                cand.append((z, float(yv[j])))
        pools.append(cand)
    usable = [p for p in pools if p]
    assert len(usable) >= 3, "区間内に対照が残る枠が 3 つ未満。この診断は成立しない"

    gen = np.random.default_rng(seed)
    out = np.empty(n_replicates)
    counts = np.empty(n_replicates, dtype=int)
    for b in range(n_replicates):
        zs, ys = [], []
        for cand in usable:
            z, yv = cand[gen.integers(len(cand))]
            zs.append(z)
            ys.append(yv)
        counts[b] = len(zs)
        out[b] = spearmanr(zs, ys).statistic if len(zs) >= 3 else np.nan
    return out, int(np.median(counts))


def empirical_p(observed: float, null: np.ndarray) -> float:
    """両側の経験的 p。(1 + #{|null| >= |obs|}) / (B + 1)。

    +1 は 0 を返さないための補正（無作為化検定の慣例）。
    """
    n = null[np.isfinite(null)]
    return float((1 + int(np.sum(np.abs(n) >= abs(observed)))) / (len(n) + 1))


def verdict(observed: float, ci: tuple[float, float], null: np.ndarray,
            min_rho: float = 0.30,
            range_matched_null: np.ndarray | None = None) -> dict:
    """事前登録の 3 条件をそのまま判定する。どれか 1 つでも欠ければ null 扱い。

    条件は (1) ブートストラップ CI が 0 を除く、(2) |rho| >= min_rho、
    (3) 交換可能性の帰無分布に対する経験的 p < 0.05。

    range_matched_null は渡せば経験的 p を併記するが、**判定には使わない**。
    区間をそろえても散らばりの差が残るので、条件にすると誤った安心を与える
    （range_matched_control_null の説明を参照）。
    """
    lo, hi = sorted(ci)
    p = empirical_p(observed, null)
    finite = null[np.isfinite(null)]
    cond = {
        "CI が 0 を除く": bool(lo > 0 or hi < 0),
        f"|rho| >= {min_rho}": bool(abs(observed) >= min_rho),
        "帰無に対する経験的 p < 0.05": bool(p < 0.05),
    }
    p_range = None
    if range_matched_null is not None:
        p_range = empirical_p(observed, range_matched_null)   # 併記のみ。条件にしない
    return {
        "observed": float(observed), "ci": (float(lo), float(hi)),
        "empirical_p": p, "empirical_p_range_matched": p_range,
        "null_median": float(np.median(finite)),
        "null_2.5%": float(np.percentile(finite, 2.5)),
        "null_97.5%": float(np.percentile(finite, 97.5)),
        "n_replicates": int(len(finite)),
        "conditions": cond,
        "supported": all(cond.values()),
    }


def pooled_correlation(coherence: list[np.ndarray], target: list[np.ndarray]) -> float:
    """入れ子を無視して全対照をまとめた相関。**比較用にだけ置いてある。**

    事前登録の初期草案がこれだった。疑似反復なので判定には使わない。
    テストで「まとめると帰無の幅が過小になる」ことを示すために残す。
    """
    zs, ys = [], []
    for coh, y in zip(coherence, target):
        for j in range(len(np.asarray(coh))):
            z = excess_z(coh, j)
            yv = float(np.asarray(y, dtype=np.float64)[j])
            if np.isfinite(z) and np.isfinite(yv):
                zs.append(z)
                ys.append(yv)
    return float(spearmanr(zs, ys).statistic) if len(zs) >= 3 else float("nan")
