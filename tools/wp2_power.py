"""WP2（臨床到達群 vs 未到達群）の検出力を、登録前に見積もる。

事前登録の目的は「後から都合よく動かせないこと」なので、何が検出できるかを
知らないまま登録しても意味がない。到達群は文献検索で集まる数が上限であり、
こちらで増やせない。したがって「その数で何が言えるか」を先に確定させる。

主要評価項目は Mann-Whitney の順位比較だが、判定は効果量（Cliff's delta）で行う。
ここでは、群サイズと真の効果量を振って検出力を数値実験で求める。

出力: results/tables/wp2_power.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ..common import TABLES  # noqa: F401  （tools からは使わないが配置を揃える）


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a>b) - P(a<b)。順位ベースなので分布形に依存しない。"""
    n_gt = sum((x > b).sum() for x in a)
    n_lt = sum((x < b).sum() for x in a)
    return (n_gt - n_lt) / (len(a) * len(b))


def simulate(n1: int, n2: int, delta_shift: float, n_sim: int, gen) -> dict:
    """到達群が未到達群より delta_shift（標準偏差単位）だけ高い場合の検出力。

    分布は標準正規で近似する。実際の z 値分布は右に裾を引くが、Mann-Whitney は
    順位しか使わないので、単調変換に対して不変であり、この近似で足りる。
    """
    sig, deltas = 0, []
    for _ in range(n_sim):
        a = gen.normal(delta_shift, 1.0, n1)   # 到達群
        b = gen.normal(0.0, 1.0, n2)           # 未到達群
        p = mannwhitneyu(a, b, alternative="greater").pvalue
        sig += p < 0.05
        deltas.append(cliffs_delta(a, b))
    return {
        "検出力(p<0.05)": round(sig / n_sim, 3),
        "Cliff's delta 中央値": round(float(np.median(deltas)), 3),
        "delta の 5 パーセンタイル": round(float(np.percentile(deltas, 5)), 3),
    }


def main() -> int:
    gen = np.random.default_rng(20260812)
    n_sim = 2000
    rows = []
    # 到達群は文献検索で集まる数が上限。悲観・中位・楽観の 3 通りを見る。
    for n1 in (10, 15, 20, 25):
        for ratio in (10,):
            n2 = n1 * ratio
            for shift in (0.3, 0.5, 0.8, 1.0, 1.2):
                r = simulate(n1, n2, shift, n_sim, gen)
                rows.append({"到達群 n": n1, "未到達群 n": n2,
                             "真の差(SD)": shift, **r})
    df = pd.DataFrame(rows)

    print("=== WP2 検出力（Mann-Whitney 片側 0.05、未到達群は到達群の 10 倍）===")
    print(df.to_string(index=False))

    print("\n=== 検出力 0.80 に到達する最小の差 ===")
    for n1, g in df.groupby("到達群 n"):
        ok = g[g["検出力(p<0.05)"] >= 0.80]
        if len(ok):
            r = ok.iloc[0]
            print(f"  到達群 {n1:>2} 件: 差 {r['真の差(SD)']} SD"
                  f"（Cliff's delta ≈ {r[chr(67)+'liff'+chr(39)+'s delta 中央値']}）")
        else:
            print(f"  到達群 {n1:>2} 件: 検討した範囲（〜1.2 SD）では 0.80 に届かない")

    out = TABLES / "wp2_power.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
