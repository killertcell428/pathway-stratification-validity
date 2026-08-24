"""表現型予測力の検証（§3.11 予測 4）。

検証コホート GSE47353 は、接種前の 2 時点（day-7 / day0）と、個人ごとのワクチン
応答クラス（mn_adjmfc_class = 0 低 / 1 中 / 2 高）を持つ。両方を満たすのは 42 名。

n = 42 で 2,093 セットを個別に検定しても、当たりを引いたセットが本物かは判定できない。
そこで問いを 3 段に分ける。

  問 1 見かけの予測力はどれだけ出るか（対照との比較で floor を出す）
  問 2 **接種前の 2 時点で予測力が再現するか**。同じ人を 1 週間前に測っただけで
       予測力が消えるなら、それは表現型の予測ではなく偶然である
  問 3 **適格性が予測力を予告するか**（予測 4 の本体）。整合性や ICC で合格した
       セットのほうが予測に効くのか、条件効果の大きさは予測力と無関係なのか

出力: results/tables/phenotype_metrics.csv
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..common import INTERIM, METADATA, TABLES, load_config, rng
from ..scoring.methods import _standardize_rows
from .metrics import null_abs_rho_with_target  # noqa: F401
from .metrics import empirical_null, matched_random_sets
from .run_evaluation import load_all_sets

TIMEPOINTS = ("day-7", "day0")
PRIMARY = "day0"


def residualize_on(y: np.ndarray, covars: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(y)), covars])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    return y - a @ beta


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    gen = rng(17)

    meta = pd.read_csv(METADATA / "valid_samples.csv")
    meta["individual"] = meta["individual"].astype(str)
    info = meta.drop_duplicates("individual").set_index("individual")
    cls = pd.to_numeric(info["response_class"], errors="coerce")

    expr = {tp: pd.read_parquet(INTERIM / f"valid_expr_{tp}.parquet") for tp in TIMEPOINTS}
    donors = [d for d in expr[TIMEPOINTS[0]].columns if d in set(expr[TIMEPOINTS[1]].columns)]
    donors = [d for d in donors if not np.isnan(cls.get(d, np.nan))]
    y = cls.loc[donors].to_numpy(dtype=float)
    print(f"解析対象: {len(donors)} 名（クラス 0/1/2 = "
          f"{int((y==0).sum())}/{int((y==1).sum())}/{int((y==2).sum())}）")

    # 年齢・性別を調整した副次解析用の共変量
    age = pd.to_numeric(info.loc[donors, "age"], errors="coerce").to_numpy(dtype=float)
    male = (info.loc[donors, "gender"].astype(str).str.lower().str[0] == "m").to_numpy(dtype=float)
    covars = np.column_stack([np.nan_to_num(age, nan=float(np.nanmean(age))), male])
    y_adj = residualize_on(y, covars)

    genes = expr[TIMEPOINTS[0]].index.intersection(expr[TIMEPOINTS[1]].index)
    Z = {}
    for tp in TIMEPOINTS:
        x = expr[tp].loc[genes, donors].to_numpy(dtype=np.float64)
        Z[tp] = np.nan_to_num(_standardize_rows(x))   # 時点内で基準化（実務での使い方）

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
    pool_by_decile = {
        int(d): np.array([index[g] for g in gs if g in index], dtype=np.int64)
        for d, gs in by_dec.items()
    }
    for tp, mat in Z.items():
        assert not np.isnan(mat).any(), f"Z[{tp}] に NaN。まとめ計算の前提が崩れる"

    def assoc(idx: np.ndarray, tp: str, target: np.ndarray) -> tuple[float, float]:
        s = np.nanmean(Z[tp][idx], axis=0)
        if np.nanstd(s) == 0:
            return np.nan, np.nan
        rho, p = stats.spearmanr(s, target)
        return float(rho), float(p)

    rows = []
    for name, (family, gene_list) in all_sets.items():
        present = [g for g in gene_list if g in index]
        cov = len(present) / len(gene_list) if gene_list else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        idx = np.array([index[g] for g in present], dtype=int)

        rho0, p0 = assoc(idx, PRIMARY, y)
        rho7, p7 = assoc(idx, "day-7", y)
        rho0_adj, p0_adj = assoc(idx, PRIMARY, y_adj)

        # 対照 nc 個（既定 10,000）を 1 回引き、2 時点の両方に当てる。
        # 時点ごとに引き直すと「どちらの測定回でも同じ床か」の比較が崩れる。
        dec_pos = np.array([dec[g] for g in present if g in dec], dtype=np.int64)
        nulls = null_abs_rho_with_target(
            {"day0": Z[PRIMARY], "day-7": Z["day-7"]}, y,
            pool_by_decile, dec_pos, nc, gen,
        )
        nn = empirical_null(abs(rho0), nulls["day0"].tolist())
        nn7 = empirical_null(abs(rho7), nulls["day-7"].tolist())

        rows.append({
            "set": name, "family": family, "n_genes_present": len(present),
            "rho_day0": rho0, "p_day0": p0,
            "rho_day-7": rho7, "p_day-7": p7,
            "rho_day0_adj": rho0_adj, "p_day0_adj": p0_adj,
            "abs_rho_null_mean": nn["null_mean"], "abs_rho_z": nn["null_z"],
            "abs_rho_p": nn["null_p"], "abs_rho_p_empirical": nn["null_p_empirical"],
            "abs_rho_null_skew": nn["null_skew"],
            "abs_rho_null_mean_day-7": nn7["null_mean"], "abs_rho_z_day-7": nn7["null_z"],
            "abs_rho_p_empirical_day-7": nn7["null_p_empirical"],
        })

    df = pd.DataFrame(rows)
    # 対照との比較（abs_rho）は経験 p に BH-FDR をかける。p_day0 は
    # 相関そのものの検定で対照とは無関係なので、そのまま Spearman の p を使う。
    for col, out in (("p_day0", "q_day0"), ("abs_rho_p_empirical", "abs_rho_q"),
                    ("abs_rho_p_empirical_day-7", "abs_rho_q_day-7")):
        ok = df[col].notna()
        df[out] = np.nan
        if ok.sum() > 1:
            df.loc[ok, out] = multipletests(df.loc[ok, col], method="fdr_bh")[1]

    # 適格性の情報を貼る（同一セット名で結合）
    retest = pd.read_csv(TABLES / "retest_metrics.csv")[["set", "ic_q", "icc_q", "ic_day0", "icc"]]
    disc = pd.read_csv(TABLES / "gene_set_metrics.csv")[["set", "cohens_d", "delta_q", "internal_consistency"]]
    df = df.merge(retest, on="set", how="left").merge(disc, on="set", how="left", suffixes=("", "_disc"))
    df.to_csv(TABLES / "phenotype_metrics.csv", index=False, encoding="utf-8")
    print(f"評価 {len(df)} セット -> phenotype_metrics.csv\n")

    print("[問 1] 見かけの予測力")
    sig0 = df.p_day0 < 0.05
    print(f"  p<0.05 のセット: {int(sig0.sum())} ({sig0.mean():.1%})　偶然の期待値 5.0%")
    print(f"  BH-FDR 0.05 を通るセット: {int((df.q_day0 < 0.05).sum())}")
    print(f"  |rho| 中央値 {df.rho_day0.abs().median():.3f} / 対照 {df.abs_rho_null_mean.median():.3f}")
    print(f"  |rho| が対照を上回る（FDR）: {int((df.abs_rho_q < 0.05).sum())}")

    print("\n[問 2] 接種前 2 時点での再現（同じ人を 1 週間前に測っただけ）")
    same_sign = np.sign(df.rho_day0) == np.sign(df["rho_day-7"])
    rep = sig0 & (df["p_day-7"] < 0.05) & same_sign
    print(f"  day0 で p<0.05 のうち、day-7 でも p<0.05 かつ同符号: {int(rep.sum())} / {int(sig0.sum())}"
          f" ({rep.sum()/max(int(sig0.sum()),1):.1%})　偶然の期待値 ~2.5%")
    print(f"  2 時点の rho の相関: {df.rho_day0.corr(df['rho_day-7'], method='spearman'):.3f}")
    print(f"  符号一致率（全セット）: {same_sign.mean():.1%}")

    print("\n[問 3] 適格性は予測力を予告するか")
    for label, mask in (("整合性合格 (ic_q<0.05)", df.ic_q < 0.05), ("ICC 合格 (icc_q<0.05)", df.icc_q < 0.05)):
        a, b = df.loc[mask, "rho_day0"].abs().dropna(), df.loc[~mask, "rho_day0"].abs().dropna()
        if len(a) < 5 or len(b) < 5:
            print(f"  {label}: 件数不足"); continue
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"  {label}: 合格 |rho| 中央値 {a.median():.3f} (n={len(a)}) vs 不合格 {b.median():.3f} (n={len(b)})"
              f"  Mann-Whitney p={u.pvalue:.3g}")
    d2 = df.dropna(subset=["cohens_d", "rho_day0"])
    print(f"  条件効果 |d| と |rho| の相関: {d2.cohens_d.abs().corr(d2.rho_day0.abs(), method='spearman'):.3f}"
          f"  (n={len(d2)})　※予測は「無相関」")
    d3 = df.dropna(subset=["ic_day0", "rho_day0"])
    print(f"  内部整合性と |rho| の相関: {d3.ic_day0.corr(d3.rho_day0.abs(), method='spearman'):.3f} (n={len(d3)})")
    d4 = df.dropna(subset=["icc", "rho_day0"])
    print(f"  ICC と |rho| の相関: {d4.icc.corr(d4.rho_day0.abs(), method='spearman'):.3f} (n={len(d4)})")

    print("\n[問 2b] 予測力の水準そのものが測定回で違うのか")
    print(f"  全 {len(df)} セットの |rho| 中央値: {PRIMARY} {df.rho_day0.abs().median():.3f} / "
          f"day-7 {df['rho_day-7'].abs().median():.3f}")
    print(f"  同じ対照の |rho| 中央値:        {PRIMARY} {df.abs_rho_null_mean.median():.3f} / "
          f"day-7 {df['abs_rho_null_mean_day-7'].median():.3f}")
    print(f"  対照に対する超過（中央値）:     {PRIMARY} "
          f"{(df.rho_day0.abs() - df.abs_rho_null_mean).median():+.3f} / day-7 "
          f"{(df['rho_day-7'].abs() - df['abs_rho_null_mean_day-7']).median():+.3f}")
    print("  → 注釈セットと対照が同じ向きに動くなら、差は測定回そのものの性質であって"
          "セットの性質ではない")

    print("\n[参考] 年齢・性別で調整した場合")
    print(f"  p<0.05: {int((df.p_day0_adj < 0.05).sum())} ({(df.p_day0_adj < 0.05).mean():.1%})"
          f"  調整前後の rho の相関: {df.rho_day0.corr(df.rho_day0_adj, method='spearman'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
