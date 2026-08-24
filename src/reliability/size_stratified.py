"""合格率がセットサイズに規定されていないかを層別で確かめる。

なぜ要るか
  本研究は「対照を上回ったセットの割合」を主要な提示量にしている。ところがこの量は
  効果量ではなく**検出力**である。対照 20 個から作る z の分母（対照の標準偏差）は
  セットが大きいほど小さくなるので、同じ超過分でも大きいセットのほうが通りやすい。

  実測すると主コホートで、対照 SD の中央値は 3-5 遺伝子の 0.096 から
  61-200 遺伝子の 0.009 まで 10 倍縮み、合格率は 3.6% から 23.6% まで 6.6 倍上がる。
  ファミリー別のサイズ中央値は complex 4 / regulon 8 / pathway 20 /
  celltype 105 / signature 125 で、**サイズ中央値と合格率の順位相関は 0.800** である。

  したがって「複合体は対照と最も区別できない」という所見は、
  生物学ではなく 4 遺伝子セットの検出力で説明できてしまう。
  これを切り分けるには (1) 効果量（超過分）で比べる、
  (2) 同一サイズ帯の中でファミリーを比べる、の 2 つが要る。

  あわせて、プラットフォーム間でファミリー順位が保たれるかも層別で取り直す。
  順位が保たれていても、サイズがプラットフォーム間で不変なら
  「保たれたのはサイズの効果」という読みが残る。層別で消えるか残るかで決まる。

出力
  results/tables/size_stratified.csv          サイズ帯 x 指標
  results/tables/size_stratified_family.csv   ファミリー x サイズ帯
  results/tables/cross_platform_rank.csv      プラットフォーム間の順位一致（全体と層別）
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import TABLES

BANDS = [2, 5, 10, 25, 60, 10 ** 6]
LABELS = ["3-5", "6-10", "11-25", "26-60", "61-200"]
# 効果量の比較に使う共通サイズ帯。ここに 3 ファミリー以上が同居する帯を選ぶ。
COMMON_BANDS = ["11-25", "26-60"]


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(TABLES / path)
    d["passed"] = d.null_q.lt(0.05) & d.var_null_q.lt(0.05)
    d["passed_1ctrl"] = d.null_q.lt(0.05)
    d["excess"] = d.internal_consistency - d.null_mean
    d["band"] = pd.cut(d.n_genes_present, bins=BANDS, labels=LABELS)
    return d


def by_band(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("band", observed=True).agg(
        n=("set", "size"),
        pass_rate=("passed", lambda s: 100 * s.mean()),
        excess_median=("excess", "median"),
        null_sd_median=("null_sd", "median"),
        ic_median=("internal_consistency", "median"),
    )
    # 同じ超過分がサイズによってどれだけ通りやすくなるかを直接示す。
    # 超過分の中央値を対照 SD の中央値で割ったものが、実効的な z である。
    g["excess_over_null_sd"] = g.excess_median / g.null_sd_median
    return g.round(4)


def by_family_band(d: pd.DataFrame) -> pd.DataFrame:
    t = d.groupby(["family", "band"], observed=True).agg(
        n=("set", "size"),
        pass_rate=("passed", lambda s: round(100 * s.mean(), 1)),
        excess_median=("excess", "median"),
    ).reset_index()
    return t[t.n >= 3].round(4)


def cross_platform(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """プラットフォーム間のファミリー順位一致を、全体と層別で取る。

    比較する量を 3 つ並べる。
      pass_rate  原稿が提示している合格率。サイズに規定されている疑いがある
      excess     効果量。サイズの影響を受けにくい
      ic_median  原稿が「保存されない（ρ = 0.00）」と報告している中央値
    """
    rows = []

    def rank(x: pd.DataFrame, y: pd.DataFrame, col: str, agg, label: str, note: str):
        ta = x.groupby("family").apply(agg, include_groups=False)
        tb = y.groupby("family").apply(agg, include_groups=False)
        idx = [i for i in ta.index.intersection(tb.index) if i != "anchor"]
        if len(idx) < 3:
            return
        r = spearmanr(ta[idx], tb[idx])
        rows.append({"量": col, "層": label, "ファミリー数": len(idx),
                     "順位相関": round(float(r.statistic), 3),
                     "p": round(float(r.pvalue), 4), "注": note})

    for col, agg in (("pass_rate", lambda g: 100 * g.passed.mean()),
                     ("excess_median", lambda g: g.excess.median()),
                     ("ic_median", lambda g: g.internal_consistency.median())):
        rank(a, b, col, agg, "全体", "サイズ層別なし")
        for band in COMMON_BANDS:
            rank(a[a.band == band], b[b.band == band], col, agg, band,
                 "同一サイズ帯に限定")
    return pd.DataFrame(rows)


def main() -> int:
    main_c = load("gene_set_metrics.csv")
    rna = pd.read_csv(TABLES / "gse81046" / "gene_set_metrics.csv")
    rna["passed"] = rna.null_q.lt(0.05) & rna.var_null_q.lt(0.05)
    rna["passed_1ctrl"] = rna.null_q.lt(0.05)
    rna["excess"] = rna.internal_consistency - rna.null_mean
    rna["band"] = pd.cut(rna.n_genes_present, bins=BANDS, labels=LABELS)

    print("=== サイズ帯別（主コホート、n = %d）===" % len(main_c))
    bb = by_band(main_c)
    print(bb.to_string())
    bb.to_csv(TABLES / "size_stratified.csv", encoding="utf-8")

    print("\n=== サイズ帯別（GSE81046、n = %d）===" % len(rna))
    print(by_band(rna).to_string())

    print("\n=== ファミリー別のサイズ中央値と合格率（主コホート）===")
    f = main_c.groupby("family").agg(
        size_median=("n_genes_present", "median"), n=("set", "size"),
        pass_rate=("passed", lambda s: round(100 * s.mean(), 1)),
        excess_median=("excess", "median"), null_sd_median=("null_sd", "median"))
    print(f.sort_values("size_median").round(4).to_string())
    ann = f.drop(index=[i for i in ("data_derived", "anchor") if i in f.index])
    r = spearmanr(ann.size_median, ann.pass_rate)
    print(f"  サイズ中央値 vs 合格率 の順位相関（注釈由来 {len(ann)} ファミリー）: "
          f"{r.statistic:.3f} (p = {r.pvalue:.3f})")
    r2 = spearmanr(ann.size_median, ann.excess_median)
    print(f"  サイズ中央値 vs 超過分 の順位相関: {r2.statistic:.3f} (p = {r2.pvalue:.3f})")

    print("\n=== 同一サイズ帯内のファミリー比較（主コホート）===")
    fb = by_family_band(main_c)
    for band in LABELS:
        sub = fb[fb.band == band]
        if len(sub) < 2:
            continue
        print(f"  [{band}] " + " / ".join(
            f"{r.family} {r.pass_rate:.0f}%(n={int(r.n)}, 超過 {r.excess_median:+.4f})"
            for _, r in sub.iterrows()))
    fb.to_csv(TABLES / "size_stratified_family.csv", index=False, encoding="utf-8")

    print("\n=== プラットフォーム間のファミリー順位一致 ===")
    cp = cross_platform(main_c, rna)
    print(cp.to_string(index=False))
    cp.to_csv(TABLES / "cross_platform_rank.csv", index=False, encoding="utf-8")

    print(f"\n-> {TABLES/'size_stratified.csv'}")
    print(f"-> {TABLES/'size_stratified_family.csv'}")
    print(f"-> {TABLES/'cross_platform_rank.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
