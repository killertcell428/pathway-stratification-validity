"""反復測定信頼性と、別プラットフォーム・別細胞での外部再現性（L1 / L2 の解消）。

検証コホート GSE47353（PBMC, Affymetrix Gene 1.0 ST）の day-7 と day0 は
どちらもワクチン接種前なので、同一個人・同一条件を 7 日間隔で測った対になる。

測るもの
  1. 反復測定信頼性: ICC(2,1) と順位相関。**対照を必ず取る**。個人の細胞組成や
     技術要因は 2 時点で共有されるため、ランダムな遺伝子セットでも順位は一定程度
     再現する。素の ICC を見ても「セットが良い」ことにはならない
  2. 別プラットフォーム・別細胞での内部整合性。ここは陽性対照も兼ねる。精製単球で
     低かった細胞種マーカーは、PBMC では組成の個人差を拾って高くなるはずである。
     そうならなければ指標側を疑う

出力: results/tables/retest_metrics.csv, retest_by_family.csv
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..common import INTERIM, TABLES, load_config, rng
from ..scoring.methods import _rank_rows, _standardize_rows
from .metrics import (
    empirical_null,
    icc_two_way,
    matched_random_sets,
    mean_pairwise_rho,
    pooled_set_score,
    rank_consistency,
)
from .run_evaluation import load_all_sets

RETEST = ("day-7", "day0")


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    gen = rng(13)

    a = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[0]}.parquet")
    b = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[1]}.parquet")
    genes = a.index.intersection(b.index)
    donors = [d for d in a.columns if d in set(b.columns)]
    a, b = a.loc[genes, donors], b.loc[genes, donors]
    print(f"反復測定の対: {len(donors)} 名 / 遺伝子 {len(genes)}（{RETEST[0]} と {RETEST[1]}、どちらも接種前）")

    # 2 時点をまたいだ共通基準で z 化する（ICC が水準のずれも拾えるようにする）
    both = np.concatenate([a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)], axis=1)
    mu = both.mean(axis=1, keepdims=True)
    sd = both.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = np.nan
    z = (both - mu) / sd
    n = a.shape[1]
    z_a, z_b = z[:, :n], z[:, n:]

    # 内部整合性は day0 の順位行列で測る
    S_day0 = _standardize_rows(_rank_rows(b.to_numpy(dtype=np.float64)))

    gene_mean = pd.read_csv(INTERIM / "valid_gene_expression.csv", index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(genes)]
    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    by_dec: dict[int, list[str]] = defaultdict(list)
    for g, d in dec.items():
        by_dec[int(d)].append(g)

    index = {g: i for i, g in enumerate(genes)}
    all_sets = load_all_sets(gs_cfg)
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]

    def scores(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pooled_set_score(z_a[idx]), pooled_set_score(z_b[idx])

    rows = []
    for name, (family, gene_list) in all_sets.items():
        present = [g for g in gene_list if g in index]
        cov = len(present) / len(gene_list) if gene_list else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        idx = np.array([index[g] for g in present], dtype=int)

        s_a, s_b = scores(idx)
        rc = rank_consistency(s_a, s_b)
        icc = icc_two_way(np.column_stack([s_a, s_b]))
        ic = mean_pairwise_rho(S_day0[idx])

        random_sets = matched_random_sets(present, dec, by_dec, nc, gen)
        null_ic, null_icc, null_rho = [], [], []
        for rs in random_sets:
            ridx = np.array([index[g] for g in rs], dtype=int)
            null_ic.append(mean_pairwise_rho(S_day0[ridx]))
            ra, rb = scores(ridx)
            null_icc.append(icc_two_way(np.column_stack([ra, rb])))
            null_rho.append(rank_consistency(ra, rb)["rho"])
        n_ic = empirical_null(ic, null_ic)
        n_icc = empirical_null(icc, null_icc)
        n_rho = empirical_null(rc["rho"], null_rho)

        rows.append({
            "set": name, "family": family, "n_genes_present": len(present), "coverage": cov,
            "ic_day0": ic, "ic_null_mean": n_ic["null_mean"], "ic_z": n_ic["null_z"], "ic_p": n_ic["null_p"],
            "icc": icc, "icc_null_mean": n_icc["null_mean"], "icc_z": n_icc["null_z"], "icc_p": n_icc["null_p"],
            "retest_rho": rc["rho"], "retest_rho_null": n_rho["null_mean"], "retest_rho_p": n_rho["null_p"],
            "retest_tertile_swap": rc["tertile_swap"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("評価できたセットが 0 件")
        return 1
    for col, out in (("ic_p", "ic_q"), ("icc_p", "icc_q"), ("retest_rho_p", "retest_rho_q")):
        ok = df[col].notna()
        df[out] = np.nan
        if ok.sum() > 1:
            df.loc[ok, out] = multipletests(df.loc[ok, col], method="fdr_bh")[1]
    df.to_csv(TABLES / "retest_metrics.csv", index=False, encoding="utf-8")
    print(f"評価 {len(df)} セット -> retest_metrics.csv")

    fam = df.groupby("family").apply(lambda d: pd.Series({
        "n": len(d),
        "サイズ中央値": int(d.n_genes_present.median()),
        "整合性": round(d.ic_day0.median(), 3),
        "整合性対照": round(d.ic_null_mean.median(), 3),
        "整合性合格率": round(float((d.ic_q < 0.05).mean()), 3),
        "ICC": round(d.icc.median(), 3),
        "ICC対照": round(d.icc_null_mean.median(), 3),
        "ICC合格率": round(float((d.icc_q < 0.05).mean()), 3),
        "再測定rho": round(d.retest_rho.median(), 3),
        "再測定rho対照": round(d.retest_rho_null.median(), 3),
        "三分位入替": round(d.retest_tertile_swap.median(), 3),
    }), include_groups=False)
    fam.to_csv(TABLES / "retest_by_family.csv", encoding="utf-8")
    print("\nファミリー別")
    print(fam.to_string())

    print("\n全体")
    print(f"  ICC 中央値 {df.icc.median():.3f}（対照 {df.icc_null_mean.median():.3f}）")
    print(f"  ICC >= 0.5 のセット: {int((df.icc >= 0.5).sum())} / {len(df)} ({(df.icc >= 0.5).mean():.1%})")
    print(f"  ICC が対照を上回る（FDR 0.05）: {int((df.icc_q < 0.05).sum())} ({(df.icc_q < 0.05).mean():.1%})")
    print(f"  内部整合性が対照を上回る: {int((df.ic_q < 0.05).sum())} ({(df.ic_q < 0.05).mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
