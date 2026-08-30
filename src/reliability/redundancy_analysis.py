"""遺伝子セットの重複を考慮して、有効な単位数と合格率を数え直す。

なぜ要るか
  2,195 セットは互いに独立ではない。Reactome には親子関係にある集合が、PanglaoDB には
  細胞種をまたいで共有されるマーカーが含まれる。同じ遺伝子プログラムを複数のセットが
  指していれば、それは同じ観測を何度も数えていることになり、割合の分母が実質より大きくなる。

  Benjamini-Hochberg と Benjamini-Yekutieli は検定間の依存を扱うが、「何を 1 件と数えるか」は
  変えない。合格率・ファミリー別の割合・「17 件中 14 件」のような集計は、すべてセットを
  独立な単位として数えている。

  ここではセット間の Jaccard 類似度で階層クラスタリングし、クラスタを 1 単位として
  合格率を数え直す。クラスタの合否は 2 通りで数える。

    any  : クラスタ内に 1 件でも合格があればクラスタを合格とする（上限側）
    rep  : クラスタの代表 1 件（遺伝子数が中央に最も近いもの）の合否を採る（中立側）

  この 2 つが主解析の値をはさむなら、重複は割合の水準を作っていないと言える。

使い方:
  pixi run redundancy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..common import TABLES, load_config
from .run_evaluation import load_all_sets

# Jaccard 距離を切る閾値。0.5 は「遺伝子の半分以上を共有するセットを 1 単位にまとめる」。
THRESHOLDS = (0.9, 0.75, 0.5, 0.25)


def main() -> int:
    print("[1/4] 主解析の結果と遺伝子セットを読む")
    metrics = pd.read_csv(TABLES / "gene_set_metrics.csv")
    gs_cfg = load_config("gene_sets")
    all_sets = load_all_sets(gs_cfg)

    # 主解析が評価したセットだけを対象にする（母集団をそろえる）
    evaluated = metrics.set_index("set")
    names = [n for n in evaluated.index if n in all_sets]
    print(f"  主解析が評価したセット {len(names):,}")

    print("[2/4] Jaccard 類似度を計算する")
    gene_list = sorted({g for n in names for g in all_sets[n][1]})
    gene_pos = {g: i for i, g in enumerate(gene_list)}
    M = np.zeros((len(names), len(gene_list)), dtype=np.float32)
    for i, n in enumerate(names):
        for g in all_sets[n][1]:
            M[i, gene_pos[g]] = 1.0
    sizes = M.sum(axis=1)
    inter = M @ M.T
    union = sizes[:, None] + sizes[None, :] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(jac, 1.0)
    dist = 1.0 - jac
    dist = (dist + dist.T) / 2.0            # 数値誤差で非対称になるのを均す
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 1.0)
    print(f"  重複の強いペア（Jaccard > 0.5）: {int(((jac > 0.5).sum() - len(names)) / 2):,} 組")

    print("[3/4] 階層クラスタリングして単位を数え直す")
    Z = linkage(squareform(dist, checks=False), method="average")
    both = (evaluated.null_q < 0.05) & (evaluated.var_null_q < 0.05)
    pass_by_name = both.reindex(names).fillna(False).to_numpy()
    n_genes = evaluated.n_genes_present.reindex(names).to_numpy()
    family = evaluated.family.reindex(names).to_numpy()

    rows = []
    for th in THRESHOLDS:
        labels = fcluster(Z, t=1.0 - th, criterion="distance")
        n_cluster = int(labels.max())
        df = pd.DataFrame({"cluster": labels, "passed": pass_by_name,
                           "n_genes": n_genes, "family": family})
        any_pass = df.groupby("cluster").passed.any()
        # 代表は遺伝子数がクラスタ中央値に最も近いもの
        rep_idx = (df.assign(d=lambda x: (x.n_genes - x.groupby("cluster").n_genes
                                          .transform("median")).abs())
                     .sort_values(["cluster", "d"]).groupby("cluster").head(1).index)
        rep_pass = df.loc[rep_idx].groupby("cluster").passed.first()
        rows.append({
            "jaccard_threshold": th,
            "n_clusters": n_cluster,
            "n_sets": len(names),
            "reduction_pct": 100 * (1 - n_cluster / len(names)),
            "pass_any_pct": 100 * any_pass.mean(),
            "pass_rep_pct": 100 * rep_pass.mean(),
        })
        print(f"  Jaccard {th}: {n_cluster:,} クラスタ"
              f"（{100*(1-n_cluster/len(names)):.1f}% 削減）"
              f" / any {100*any_pass.mean():.1f}%  rep {100*rep_pass.mean():.1f}%")

    print("[4/4] 書き出す")
    out = pd.DataFrame(rows)
    base = 100 * pass_by_name.mean()
    out["set_level_pct"] = base
    path = TABLES / "redundancy_analysis.csv"
    out.to_csv(path, index=False)

    print()
    print(f"セット単位の合格率（主解析と同じ数え方）: {base:.1f}%")
    print(out.to_string(index=False))
    print(f"\n書き出し: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
