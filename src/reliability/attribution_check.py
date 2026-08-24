"""内部整合性が「何に由来するか」を特定する（帰属検証）。

きっかけ: 細胞種マーカーの中で整合性が高かったのは Hemangioblasts / Trophoblast
Progenitor Cells / Reticulocytes など、精製 CD14+ 単球に存在しない細胞種だった。
一方 "Monocytes" セット自体は対照を超えなかった。つまり整合性はセットが名乗る
生物学とは無関係な軸から来ている疑いがある。

そこで解釈可能な軸を明示的に作り、各軸を各遺伝子から回帰で抜いたあとに
内部整合性がどれだけ残るかを測る。残らなければ、その整合性は軸の産物である。

  monocyte_subset : 単球サブセット比率（CD16 系 vs CD14/classical 系）
  platelet        : 血小板混入
  erythrocyte     : 赤血球混入
  lymphocyte      : T/NK 混入
  global_pc1      : 全遺伝子の第 1 主成分（技術・品質軸の代理）

出力: results/tables/attribution_check.csv, attribution_by_family.csv
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, TABLES, load_config, rng
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import _rank_rows, _standardize_rows
from .metrics import draw_null_rho  # noqa: F401
from .metrics import empirical_null, matched_random_sets, mean_pairwise_rho
from .run_evaluation import load_all_sets

# (増える側, 減る側) の対比で軸を作る。片側だけなら平均 z をそのまま軸にする。
AXES: dict[str, tuple[list[str], list[str]]] = {
    "monocyte_subset": (
        ["FCGR3A", "CX3CR1", "MS4A7", "CDKN1C", "LYPD2", "TCF7L2"],
        ["CD14", "CCR2", "VCAN", "SELL", "S100A12", "CD36"],
    ),
    "platelet": (["PF4", "PPBP", "ITGA2B", "GP9", "TUBB1", "SELP"], []),
    "erythrocyte": (["HBB", "HBA1", "HBA2", "ALAS2", "AHSP", "SLC4A1"], []),
    "lymphocyte": (["CD3D", "CD3E", "CD2", "IL7R", "TRAC", "GZMA", "NKG7"], []),
}


def build_axis(expr: pd.DataFrame, up: list[str], down: list[str]) -> tuple[np.ndarray, list[str]]:
    """マーカーの平均 z の差を軸にする。使えたマーカー名も返す。"""
    z = pd.DataFrame(
        _standardize_rows(expr.to_numpy(dtype=np.float64)),
        index=expr.index, columns=expr.columns,
    )
    used_up = [g for g in up if g in z.index]
    used_down = [g for g in down if g in z.index]
    if not used_up:
        return np.zeros(expr.shape[1]), []
    v = z.loc[used_up].mean(axis=0).to_numpy()
    if used_down:
        v = v - z.loc[used_down].mean(axis=0).to_numpy()
    return v, used_up + [f"-{g}" for g in used_down]


def residualize(x: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """各遺伝子（行）を軸（列方向の説明変数）で回帰して残差にする。"""
    a = np.column_stack([np.ones(axes.shape[0]), axes])
    beta, *_ = np.linalg.lstsq(a, x.T, rcond=None)
    return x - (a @ beta).T


def std_ranks(x: np.ndarray) -> np.ndarray:
    return _standardize_rows(_rank_rows(x))


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    resting = cfg["conditions"]["resting"]

    expr = pd.read_parquet(INTERIM / f"expr_{resting}.parquet")
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    sub = expr[val]
    x = sub.to_numpy(dtype=np.float64)
    print(f"評価個人 {len(val)} 名 / 遺伝子 {sub.shape[0]}")

    # 軸を作る
    axis_vec: dict[str, np.ndarray] = {}
    for name, (up, down) in AXES.items():
        v, used = build_axis(sub, up, down)
        if not used:
            print(f"  [skip] {name}: マーカーが 1 つも検出されていない")
            continue
        axis_vec[name] = v
        print(f"  {name:16s} マーカー {len(used)}: {', '.join(used)}")

    # 全体第 1 主成分（技術・品質軸の代理）
    z_all = _standardize_rows(x)
    z_all = np.nan_to_num(z_all)
    _, _, vt = np.linalg.svd(z_all - z_all.mean(axis=1, keepdims=True), full_matrices=False)
    axis_vec["global_pc1"] = vt[0]
    print(f"  {'global_pc1':16s} 全遺伝子 SVD の第 1 右特異ベクトル")

    # 軸どうしの相関（独立性の確認）
    names = list(axis_vec)
    corr = pd.DataFrame(
        [[round(float(np.corrcoef(axis_vec[a], axis_vec[b])[0, 1]), 2) for b in names] for a in names],
        index=names, columns=names,
    )
    print("\n軸どうしの相関\n" + corr.to_string())

    # 残差行列を用意する
    S = {"none": std_ranks(x)}
    for name, v in axis_vec.items():
        S[name] = std_ranks(residualize(x, v[:, None]))
    S["all"] = std_ranks(residualize(x, np.column_stack([axis_vec[n] for n in names])))

    # 陰性対照は "all" 条件だけで計算する（結論に必要なのは「対照水準まで落ちたか」）
    gene_mean = pd.read_csv(INTERIM / "gene_expression_naive.csv", index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(sub.index)]
    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    by_dec: dict[int, list[str]] = {}
    for g, d in dec.items():
        by_dec.setdefault(int(d), []).append(g)

    index = {g: i for i, g in enumerate(sub.index)}
    all_sets = load_all_sets(gs_cfg)
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]
    pool_by_decile = {
        int(d): np.array([index[g] for g in gs if g in index], dtype=np.int64)
        for d, gs in by_dec.items()
    }
    for cond, mat in S.items():
        assert not np.isnan(mat).any(), f"S[{cond}] に NaN。まとめ計算の前提が崩れる"
    gen = rng(7)

    rows = []
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
        dec_pos = np.array([dec[g] for g in present if g in dec], dtype=np.int64)
        nulls_all = draw_null_rho(S["all"], pool_by_decile, dec_pos, nc, gen)
        nn = empirical_null(rec["ic_all"], nulls_all.tolist())
        rec["null_all_mean"] = nn["null_mean"]
        rec["null_all_z"] = nn["null_z"]
        rec["null_all_p"] = nn["null_p"]
        rec["null_all_p_empirical"] = nn["null_p_empirical"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["retained_frac"] = df["ic_all"] / df["ic_none"].where(df["ic_none"].abs() > 1e-9)
    df.to_csv(TABLES / "attribution_check.csv", index=False, encoding="utf-8")
    print(f"\n評価 {len(df)} セット -> attribution_check.csv")

    cols = [c for c in df.columns if c.startswith("ic_")]
    fam = df.groupby("family")[cols + ["null_all_mean", "null_all_z"]].median().round(4)
    fam["n"] = df.groupby("family").size()
    fam.to_csv(TABLES / "attribution_by_family.csv", encoding="utf-8")
    print("\nファミリー別の内部整合性（軸を抜く前 / 各軸を抜いた後）")
    print(fam.to_string())

    print("\n細胞種マーカー 上位 12 セット")
    c = df[df.family == "celltype"].copy()
    c["name"] = c["set"].str.split("|").str[1]
    c = c.sort_values("ic_none", ascending=False).head(12)
    show = ["name", "n_genes_present", "ic_none", "ic_all", "null_all_mean", "null_all_z"]
    print(c[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
