"""見かけの整合性の主因が技術要因であることを、別コホート・別の技術変数で検証する（L3 の一般化）。

発見コホートでは「第 1 主成分の 0.899 が測定チップで説明される」ことを示したが、
技術変数はチップ・位置・処理日しかなく、生物学的共変量は民族・保存状態などすべて
単一値だった。したがって「技術か生物学か」の比較そのものはできていない。

GSE35846（Preininger et al. 2013、全血 189 名、Illumina HT-12 = 発見コホートと同系列）は
技術と生物学の両方の共変量を持つ唯一の候補である。

  技術   プレート ID（17 枚）／チップ内位置（A-L）／処理日（5 群）／RNA integrity score
  生物学 性別／年齢／体脂肪率／民族

やること
  1. 主成分 PC1-PC5 の分散を、技術変数と生物学変数がそれぞれどれだけ説明するかを比べる
     （群数の差で R^2 が変わるので、ラベル並べ替えの偶然水準を引く）
  2. 技術変数を抜いた前後で、ファミリー別の内部整合性と順序が保たれるかを見る
  3. 全血なので細胞組成の交絡がある。細胞種マーカーのスコアを組成の代理として
     主成分の説明変数に加え、組成と技術のどちらが優勢かも比べる

出力: results/tables/technical_axes.csv, technical_axes_by_family.csv
"""

from __future__ import annotations

import gzip
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, RAW, TABLES, load_config, rng
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import _rank_rows, _standardize_rows
from .batch_check import dummies, r2_with_permutation, residualize
from .metrics import empirical_null, matched_random_sets, mean_pairwise_rho
from .run_evaluation import load_all_sets

SERIES = "GSE35846_series_matrix.txt.gz"
ANNOT = "GPL10558.annot.gz"
N_PC = 5

TECHNICAL = ["plate id", "chip_position", "batch effect_date", "rna integrity score"]
BIOLOGICAL = ["gender", "age", "percentage of body fat", "ethnicity"]
CONTINUOUS = {"rna integrity score", "age", "percentage of body fat"}
# 組成の代理に使う細胞種マーカー（全血で比率が大きく変わるもの）
COMPOSITION_SETS = ["celltype|Neutrophils", "celltype|T Cells Naive", "celltype|B Cells",
                    "celltype|NK Cells", "celltype|Platelets", "celltype|Monocytes"]


def parse_series(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gsms: list[str] = []
    chars: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_geo_accession"):
                gsms = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append([v.strip('"') for v in line.rstrip("\n").split("\t")[1:]])
        expr = pd.read_csv(f, sep="\t", index_col=0, comment="!", low_memory=False)
    expr.index = expr.index.astype(str).str.strip('"')
    expr.columns = [str(c).strip('"') for c in expr.columns]

    meta = pd.DataFrame({"gsm": gsms})
    for vals in chars:
        key = vals[0].split(":")[0].strip()
        meta[key] = [re.sub(rf"^{re.escape(key)}:\s*", "", v).strip() for v in vals]
    return expr[[c for c in expr.columns if c in set(gsms)]].astype(np.float32), meta.set_index("gsm")


def parse_annotation(path) -> pd.Series:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            if line.startswith("ID\t"):
                break
            skip += 1
    df = pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False, compression="gzip",
                     encoding_errors="replace")
    col = next((c for c in df.columns if c.lower().startswith("gene symbol")), None)
    s = df[[df.columns[0], col]].dropna()
    s[col] = s[col].astype(str).str.strip().str.upper()
    s = s[~s[col].str.contains(r"///|^$")]
    return s.drop_duplicates(df.columns[0]).set_index(df.columns[0])[col]


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    gen = rng(19)

    print("[1/4] 行列とメタデータを読む")
    expr, meta = parse_series(RAW / SERIES)
    probe2sym = parse_annotation(RAW / ANNOT)
    meta = meta.loc[[g for g in expr.columns if g in meta.index]]
    expr = expr[meta.index.tolist()]
    print(f"  {expr.shape[0]} probes x {expr.shape[1]} samples  値域 "
          f"{expr.to_numpy().min():.2f}〜{expr.to_numpy().max():.2f}")
    for c in TECHNICAL + BIOLOGICAL:
        print(f"  {c:26s} 種類={meta[c].nunique()}")

    print("[2/4] 遺伝子レベルに畳む")
    # このコホートの公開行列はプローブ単位で中心化済み（全プローブの平均が厳密に 0）で、
    # かつ著者側で検出プローブ 14,343 本に絞られている。したがって
    #   ・平均発現によるフィルタは成立しない（発現フィルタは適用しない）
    #   ・代表プローブは平均発現ではなく標準偏差が最大のものを選ぶ
    #   ・陰性対照のマッチングは発現量分位ではなく標準偏差分位で行う
    # 内部整合性は個人方向の順位相関なので、プローブ単位の中心化には影響されない。
    common = expr.index.intersection(probe2sym.index.astype(str))
    sym = probe2sym.loc[common]
    sd_all = expr.loc[common].std(axis=1)
    order = pd.DataFrame({"symbol": sym.values, "sd": sd_all.values}, index=common)
    rep = order.sort_values(["symbol", "sd"], ascending=[True, False]).groupby("symbol").head(1)
    gene_expr = expr.loc[rep.index]
    gene_expr.index = pd.Index(rep["symbol"].values, name="gene")
    gmean = gene_expr.std(axis=1)   # 以降のマッチングは標準偏差で行う
    print(f"  {gene_expr.shape[0]} 遺伝子 x {gene_expr.shape[1]} サンプル"
          f"（公開行列が中心化済みのため発現フィルタは適用しない）")
    gene_expr.astype(np.float32).to_parquet(INTERIM / "tech_expr_wholeblood.parquet")
    meta.to_csv(METADATA / "tech_samples.csv", encoding="utf-8")

    x = gene_expr.to_numpy(dtype=np.float64)
    z = np.nan_to_num(_standardize_rows(x))
    zc = z - z.mean(axis=1, keepdims=True)
    _, sv, vt = np.linalg.svd(zc, full_matrices=False)
    var_ratio = (sv ** 2) / float((sv ** 2).sum())

    print("\n[3/4] 主成分の分散を技術要因と生物学要因が説明する割合")
    index = {g: i for i, g in enumerate(gene_expr.index)}
    all_sets = load_all_sets(gs_cfg)

    # 組成の代理（細胞種マーカーの z 平均）
    comp = {}
    for name in COMPOSITION_SETS:
        if name in all_sets:
            genes = [g for g in all_sets[name][1] if g in index]
            if len(genes) >= 10:
                comp[name.split("|")[1]] = np.nanmean(z[[index[g] for g in genes]], axis=0)
    print(f"  組成の代理に使えた細胞種マーカー: {list(comp)}")

    rows = []
    for k in range(N_PC):
        pc = vt[k]
        for kind, cols in (("technical", TECHNICAL), ("biological", BIOLOGICAL)):
            for c in cols:
                v = meta[c].to_numpy()
                if c in CONTINUOUS:
                    num = pd.to_numeric(meta[c], errors="coerce")
                    v = pd.qcut(num, 5, labels=False, duplicates="drop").fillna(-1).to_numpy()
                res = r2_with_permutation(pc, v, gen)
                rows.append({"pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]),
                             "kind": kind, "factor": c, **res})
        for cname, vec in comp.items():
            res = r2_with_permutation(pc, pd.qcut(pd.Series(vec), 5, labels=False,
                                                  duplicates="drop").to_numpy(), gen)
            rows.append({"pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]),
                         "kind": "composition", "factor": cname, **res})
    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(TABLES / "technical_axes.csv", index=False, encoding="utf-8")
    piv = rep_df.pivot_table(index=["kind", "factor"], columns="pc", values="r2_excess").round(3)
    print(piv.to_string())

    print("\n[4/4] 技術要因を抜く前後の内部整合性")
    design_tech = np.column_stack([
        dummies(meta["plate id"]), dummies(meta["chip_position"]),
        pd.to_numeric(meta["rna integrity score"], errors="coerce")
        .fillna(pd.to_numeric(meta["rna integrity score"], errors="coerce").mean()).to_numpy()[:, None],
    ])
    design_bio = np.column_stack([
        dummies(meta["gender"]), dummies(meta["ethnicity"]),
        pd.to_numeric(meta["age"], errors="coerce").fillna(0).to_numpy()[:, None],
        pd.to_numeric(meta["percentage of body fat"], errors="coerce").fillna(0).to_numpy()[:, None],
    ])
    print(f"  設計行列: 技術 {design_tech.shape[1]} 列 / 生物学 {design_bio.shape[1]} 列 (n={x.shape[1]})")

    S = {
        "none": _standardize_rows(_rank_rows(x)),
        "tech_removed": _standardize_rows(_rank_rows(residualize(x, design_tech))),
        "bio_removed": _standardize_rows(_rank_rows(residualize(x, design_bio))),
    }

    dec = pd.qcut(gmean, 10, labels=False, duplicates="drop").to_dict()
    by_dec: dict[int, list[str]] = defaultdict(list)
    for g, d in dec.items():
        by_dec[int(d)].append(g)

    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]
    out = []
    for name, (family, gene_list) in all_sets.items():
        present = [g for g in gene_list if g in index]
        cov = len(present) / len(gene_list) if gene_list else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        idx = np.array([index[g] for g in present], dtype=int)
        rec = {"set": name, "family": family, "n_genes_present": len(present)}
        random_sets = matched_random_sets(present, dec, by_dec, nc, gen)
        for cond, mat in S.items():
            rec[f"ic_{cond}"] = mean_pairwise_rho(mat[idx])
            nulls = [mean_pairwise_rho(mat[np.array([index[g] for g in rs], dtype=int)])
                     for rs in random_sets]
            nn = empirical_null(rec[f"ic_{cond}"], nulls)
            rec[f"null_{cond}"] = nn["null_mean"]
            rec[f"z_{cond}"] = nn["null_z"]
        out.append(rec)

    df = pd.DataFrame(out)
    cols = [c for c in df.columns if c.startswith(("ic_", "null_", "z_"))]
    fam = df.groupby("family")[cols].median().round(4)
    fam["n"] = df.groupby("family").size()
    fam.to_csv(TABLES / "technical_axes_by_family.csv", encoding="utf-8")
    print(fam[["n", "ic_none", "null_none", "ic_tech_removed", "null_tech_removed",
               "ic_bio_removed", "null_bio_removed"]].to_string())
    print(f"\n評価 {len(df)} セット -> technical_axes.csv / technical_axes_by_family.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
