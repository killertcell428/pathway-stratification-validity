"""部分抽出区間が名目どおりの被覆率を持つかを、実データで確かめる。

なぜ要るか
  帰属推定値に付けた 95% 区間は、検体の 70% を復元なしで抽出し、中心化した分位を
  sqrt(m/n) で換算する方式（Politis-Romano）で作っている。ただし部分抽出の標準理論は
  m/n -> 0 を仮定しており、m = 0.7n はその設定に当てはまらない。
  したがって「理論が保証するから正しい」とは言えず、実測で確かめる必要がある。

やり方
  全検体で計算した推定値を真値とみなす。そこから検体の一部を復元なしで抜いて
  「小さいデータセット」を作り、そのデータセットだけを使って同じ手続きで区間を作る。
  その区間が真値を覆う割合を数える。名目 95% に対して実測が何%かを見る。

  この設計は真値を仮定しない。全検体の値が定義上の真値であり、
  小標本から作った区間がそれを覆うかどうかだけを問う。

同時に測るもの
  標本サイズによる偏り。統計量が n に依存して偏るなら、区間の幅とは別の理由で
  被覆率が落ちる。小標本での推定値の平均と全検体の値の差を報告する。

出力: results/tables/subsampling_coverage.csv
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from ..common import RESULTS, load_config, rng
from .attribution_uncertainty import (load_gse35846, weighted_by_kind)

TABLES_ROOT = RESULTS / "tables"


def interval(x, cols, att, gen, factors, n_pc, n_sub, frac, nperm):
    """cols のデータだけを使って、点推定と 95% 中心化部分抽出区間を作る。"""
    point = weighted_by_kind(x, cols, att, gen, factors, n_pc, nperm)
    m = int(round(len(cols) * frac))
    reps = []
    for _ in range(n_sub):
        sub = [cols[i] for i in gen.choice(len(cols), size=m, replace=False)]
        reps.append(weighted_by_kind(x, sub, att, gen, factors, n_pc, nperm))
    rep = pd.DataFrame(reps)
    scale = np.sqrt(m / len(cols))
    out = {}
    for kind, v in point.items():
        if kind not in rep:
            continue
        centred = rep[kind].dropna().to_numpy() - v
        lo, hi = np.percentile(centred, [2.5, 97.5])
        out[kind] = (v, v - scale * hi, v + -scale * lo)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-datasets", type=int, default=40,
                    help="疑似データセットの数（既定 40）")
    ap.add_argument("--n-sub", type=int, default=100,
                    help="1 データセットあたりの部分標本数（既定 100）")
    ap.add_argument("--dataset-frac", type=float, default=0.7,
                    help="疑似データセットが全検体に占める割合（既定 0.7）")
    ap.add_argument("--frac", type=float, default=0.7,
                    help="区間を作るときの m/n（既定 0.7、本走行と同じ）")
    ap.add_argument("--n-perm", type=int, default=20,
                    help="並べ替え回数（既定 20。被覆率の判定には帰無水準の精度より反復数が効く）")
    a = ap.parse_args()

    cfg = load_config("analysis")
    att = cfg.get("attribution", {"n_permutations": 200, "n_pcs": 5})
    gen = rng(211)

    x, factors, n_pc = load_gse35846()
    n = x.shape[1]
    print(f"GSE35846 全血 / 検体 {n} / 因子 {len(factors)}")

    print("[1/2] 全検体での推定値（これを真値とみなす）")
    truth = weighted_by_kind(x, list(range(n)), att, rng(31), factors, n_pc,
                             int(att["n_permutations"]))
    for k, v in sorted(truth.items(), key=lambda kv: -kv[1]):
        print(f"  {k:8s} {v:6.3f}")

    n_small = int(round(n * a.dataset_frac))
    print(f"[2/2] 疑似データセット {a.n_datasets} 個（各 {n_small}/{n} 検体）× "
          f"部分標本 {a.n_sub} 個で区間を作り、真値を覆うか数える")
    hit = {k: 0 for k in truth}
    total = {k: 0 for k in truth}
    widths = {k: [] for k in truth}
    points = {k: [] for k in truth}
    for d in range(a.n_datasets):
        cols = list(gen.choice(n, size=n_small, replace=False))
        got = interval(x, cols, att, gen, factors, n_pc, a.n_sub, a.frac, a.n_perm)
        for k, (pt, lo, hi) in got.items():
            if k not in truth:
                continue
            total[k] += 1
            points[k].append(pt)
            widths[k].append(hi - lo)
            if lo <= truth[k] <= hi:
                hit[k] += 1
        if (d + 1) % 10 == 0:
            print(f"  {d+1}/{a.n_datasets}")

    rows = []
    for k in sorted(truth, key=lambda k: -truth[k]):
        if not total[k]:
            continue
        cov = hit[k] / total[k]
        rows.append({
            "分類": k,
            "全検体の値（真値）": round(truth[k], 3),
            "小標本の推定値 平均": round(float(np.mean(points[k])), 3),
            "偏り": round(float(np.mean(points[k]) - truth[k]), 3),
            "区間幅 中央値": round(float(np.median(widths[k])), 3),
            "被覆率（名目 95%）": f"{cov:.0%}",
            "データセット数": total[k],
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_ROOT / "subsampling_coverage.csv", index=False,
               encoding="utf-8")
    print()
    print(out.to_string(index=False))
    print(f"\n-> {TABLES_ROOT / 'subsampling_coverage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
