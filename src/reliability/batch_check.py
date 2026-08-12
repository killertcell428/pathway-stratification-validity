"""整合性の主因だった第 1 主成分が、技術由来か否かを判定する（L3 の解消）。

E-MTAB-2232 の SDRF には測定の技術情報が入っている。
  Assay Name       = Illumina BeadChip のバーコード + チップ内位置（例 7509352085_F）
  Array Data File  = 生データの処理バッチ（2010 年ファイル / 2013 年ファイル）
一方、民族・保存状態・疾患・細胞種はすべて単一値なので、これらでは説明できない。

やること
  1. 主成分 PC1-PC5 の分散を、チップ / チップ内位置 / 処理バッチがどれだけ説明するかを測る
     （群数が多いと R^2 は偶然でも上がるので、ラベルの並べ替えで偶然水準を出して比較する）
  2. 技術要因（処理バッチ + 位置）を各遺伝子から抜いて内部整合性を再計算し、
     PC1 を抜いた場合と比べる。PC1 除去は過補正の疑いがあるが、技術要因の除去は
     解釈が明確なので、結論の根拠として強い

出力: results/tables/batch_check.csv, batch_by_family.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, RAW, TABLES, load_config, rng
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import _rank_rows, _standardize_rows
from .metrics import empirical_null, matched_random_sets, mean_pairwise_rho
from .run_evaluation import load_all_sets

N_PC = 5
N_PERM = 200


def load_sdrf_technical(resting_stimulus: str = "naive") -> pd.DataFrame:
    """安静時サンプルの技術メタデータを individual をキーに返す。"""
    path = RAW / "E-MTAB-2232.sdrf.txt"
    rows = list(csv.DictReader(path.open(encoding="utf-8", errors="replace"), delimiter="\t"))
    recs = []
    for r in rows:
        if r["Factor value [stimulus]"] != resting_stimulus:
            continue
        assay = r["Assay Name"]
        chip, _, position = assay.partition("_")
        recs.append({
            "individual": str(r["Characteristics[individual]"]).strip(),
            "chip": chip,
            "position": position or "NA",
            "batch": r["Array Data File"],
        })
    df = pd.DataFrame(recs).drop_duplicates("individual").set_index("individual")
    return df


def r2_oneway(y: np.ndarray, labels: np.ndarray) -> float:
    """一元配置の説明率。群平均で説明できる分散の割合。"""
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return np.nan
    within = 0.0
    for lab in np.unique(labels):
        v = y[labels == lab]
        within += float(((v - v.mean()) ** 2).sum())
    return 1.0 - within / total


def r2_with_permutation(y: np.ndarray, labels: np.ndarray, gen) -> dict[str, float]:
    obs = r2_oneway(y, labels)
    perm = np.array([r2_oneway(y, gen.permutation(labels)) for _ in range(N_PERM)])
    p = float((np.sum(perm >= obs) + 1) / (N_PERM + 1))
    return {
        "r2": obs,
        "r2_chance": float(np.nanmean(perm)),
        "r2_excess": obs - float(np.nanmean(perm)),
        "p": p,
        "n_groups": int(len(np.unique(labels))),
    }


def dummies(labels: pd.Series) -> np.ndarray:
    d = pd.get_dummies(labels, drop_first=True).to_numpy(dtype=np.float64)
    return d


def residualize(x: np.ndarray, design: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(design.shape[0]), design])
    beta, *_ = np.linalg.lstsq(a, x.T, rcond=None)
    return x - (a @ beta).T


def std_ranks(x: np.ndarray) -> np.ndarray:
    return _standardize_rows(_rank_rows(x))


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    resting = cfg["conditions"]["resting"]
    gen = rng(11)

    expr = pd.read_parquet(INTERIM / f"expr_{resting}.parquet")
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]

    tech = load_sdrf_technical()
    have = [d for d in val if d in tech.index]
    print(f"検証側 {len(val)} 名のうち技術情報が付いたのは {len(have)} 名")
    sub = expr[have]
    meta = tech.loc[have]
    print(f"  チップ {meta.chip.nunique()} 枚 / 位置 {meta.position.nunique()} 種 / 処理バッチ {meta.batch.nunique()} 群")
    print("  処理バッチの内訳:", dict(meta.batch.value_counts()))

    x = sub.to_numpy(dtype=np.float64)
    z = np.nan_to_num(_standardize_rows(x))
    zc = z - z.mean(axis=1, keepdims=True)
    _, sv, vt = np.linalg.svd(zc, full_matrices=False)
    var_ratio = (sv ** 2) / float((sv ** 2).sum())

    print("\n[1] 主成分の分散を技術要因がどれだけ説明するか")
    rows = []
    for k in range(N_PC):
        pc = vt[k]
        for fac in ("chip", "position", "batch"):
            res = r2_with_permutation(pc, meta[fac].to_numpy(), gen)
            rows.append({"pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]), "factor": fac, **res})
    rep = pd.DataFrame(rows)
    rep.to_csv(TABLES / "batch_check.csv", index=False, encoding="utf-8")
    show = rep.copy()
    for c in ("var_ratio", "r2", "r2_chance", "r2_excess"):
        show[c] = show[c].round(3)
    print(show.to_string(index=False))

    print("\n[2] 技術要因を抜いたときの内部整合性")
    design_tech = np.column_stack([dummies(meta["batch"]), dummies(meta["position"])])
    design_chip = dummies(meta["chip"])
    print(f"  設計行列: バッチ+位置 {design_tech.shape[1]} 列 / チップ {design_chip.shape[1]} 列（n={len(have)}）")

    S = {
        "none": std_ranks(x),
        "batch_position": std_ranks(residualize(x, design_tech)),
        "chip": std_ranks(residualize(x, design_chip)),
        "pc1": std_ranks(residualize(x, vt[0][:, None])),
    }

    gene_mean = pd.read_csv(INTERIM / "gene_expression_naive.csv", index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(sub.index)]
    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    by_dec: dict[int, list[str]] = defaultdict(list)
    for g, d in dec.items():
        by_dec[int(d)].append(g)

    index = {g: i for i, g in enumerate(sub.index)}
    all_sets = load_all_sets(gs_cfg)
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]

    out = []
    for name, (family, genes) in all_sets.items():
        present = [g for g in genes if g in index]
        cov = len(present) / len(genes) if genes else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        idx = np.array([index[g] for g in present], dtype=int)
        rec = {"set": name, "family": family, "n_genes_present": len(present)}
        for cond, mat in S.items():
            rec[f"ic_{cond}"] = mean_pairwise_rho(mat[idx])
        # 対照は補正条件ごとに取り直す。自由度を落とす補正は対照側の相関も下げるため、
        # 同じ補正を通した対照と比べないと意味がない。
        random_sets = matched_random_sets(present, dec, by_dec, nc, gen)
        for cond, tag in (("batch_position", "bp"), ("chip", "chip"), ("none", "raw")):
            nulls = [
                mean_pairwise_rho(S[cond][np.array([index[g] for g in rs], dtype=int)])
                for rs in random_sets
            ]
            nn = empirical_null(rec[f"ic_{cond}"], nulls)
            rec[f"null_{tag}_mean"] = nn["null_mean"]
            rec[f"null_{tag}_z"] = nn["null_z"]
            rec[f"null_{tag}_p"] = nn["null_p"]
        out.append(rec)

    df = pd.DataFrame(out)
    from statsmodels.stats.multitest import multipletests
    for tag in ("raw", "bp", "chip"):
        ok = df[f"null_{tag}_p"].notna()
        df[f"null_{tag}_q"] = np.nan
        if ok.sum() > 1:
            df.loc[ok, f"null_{tag}_q"] = multipletests(df.loc[ok, f"null_{tag}_p"], method="fdr_bh")[1]
    df.to_csv(TABLES / "batch_check_sets.csv", index=False, encoding="utf-8")

    cols = [c for c in df.columns if c.startswith("ic_")] + [
        "null_raw_mean", "null_chip_mean", "null_chip_z"
    ]
    fam = df.groupby("family")[cols].median().round(4)
    fam["n"] = df.groupby("family").size()
    for tag in ("raw", "chip"):
        fam[f"合格_{tag}"] = df.groupby("family")[f"null_{tag}_q"].apply(lambda s: round(float((s < 0.05).mean()), 3))
    fam.to_csv(TABLES / "batch_by_family.csv", encoding="utf-8")
    print(fam.to_string())
    print("\nチップを抜いた後も対照を超えるセット: "
          f"{int((df.null_chip_q < 0.05).sum())} / {len(df)} "
          f"({(df.null_chip_q < 0.05).mean():.1%})　"
          f"／抜く前: {int((df.null_raw_q < 0.05).sum())} ({(df.null_raw_q < 0.05).mean():.1%})")
    print(f"\n評価 {len(df)} セット -> batch_check.csv / batch_by_family.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
