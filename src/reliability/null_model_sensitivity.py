"""対照の作り方を変えても判定が維持されるかを確かめる（帰無モデルの感度分析）。

なぜ要るか
  主解析の対照は、セットサイズと構成遺伝子ごとの平均発現量十分位をそろえて全遺伝子から
  引いている。しかし「何らかの注釈に現れる遺伝子である」という性質そのものは保存していない。
  注釈遺伝子は高発現側に偏っており（発現量の最下位十分位で 61.6%、最上位十分位で 84.4% が
  いずれかの注釈に現れる）、全遺伝子から引いた対照はこの偏りを持たない。

  対照が注釈セットの構造的な性質を保存していなければ、観測される超過は注釈の質ではなく
  プールの違いを測っていることになる。ここでは注釈に現れる遺伝子だけからなるプールで
  同じ手順を繰り返し、合格判定がどれだけ動くかを測る。

  発現量十分位でのマッチは両方の腕で同じようにかけている。したがって差が出るとすれば、
  それは十分位の中に残っている「注釈されやすさ」の偏りによる。

  第 3 の腕として、第 1 主成分への寄与（loading の絶対値）をそろえた対照も引く。
  3.7 節は「見かけの整合性はそのコホートで最大の変動軸と重なる」ことを示した。支配的な軸への
  寄与が同程度の遺伝子から対照を引けば、その重なりを対照側も持つ。ここで超過が消えるなら、
  観測された整合性は「PC1 への寄与が高い遺伝子を集めていること」の帰結だと言える。

  ただし PC1 の十分位だけでそろえると、発現量のマッチが外れる。低発現の遺伝子は相関の推定が
  雑音に埋もれて 0 に寄るので、対照に低発現遺伝子が混ざれば対照側の水準が下がり、超過は
  自動的に大きく出る。それでは「PC1 で説明されるか」を問うたことにならない。そこで発現量
  十分位は主解析と同じに保ったまま、その十分位の中で PC1 寄与の五分位をそろえる二重マッチに
  する。遺伝子数が 30 未満になるセルは、同じ発現量十分位の全体に落とす。

  乱数系列は腕ごとに分ける。1 つの系列を共有すると、腕を足しただけで既存の腕の抽出順が
  変わり、以前の結果が再現しなくなる。

使い方:
  pixi run null-model
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..common import METADATA, TABLES, expr_path, gene_mean_path, load_config, rng
from ..scoring.methods import ScoringContext
from .metrics import draw_null_rho, empirical_null, mean_pairwise_rho, pools_as_rows
from .run_evaluation import load_all_sets

# PC1 二重マッチでこれ未満の遺伝子しかないセルは、発現量十分位の全体に落とす。
MIN_CELL = 30


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    resting = cfg["conditions"]["resting"]

    print("[1/4] 行列と遺伝子セットを読む")
    expr = pd.read_parquet(expr_path(resting))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    rest_val = expr[val]
    ctx = ScoringContext(rest_val)

    all_sets = load_all_sets(gs_cfg)
    annotated: set[str] = set()
    for _family, genes in all_sets.values():
        annotated |= set(genes)
    in_matrix = annotated & set(rest_val.index)
    print(f"  注釈に現れる遺伝子 {len(annotated):,}（行列にあるもの {len(in_matrix):,}）")

    print("[2/4] 2 種類の抽出プールを作る")
    gene_mean = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(rest_val.index)]
    deciles = pd.qcut(gene_mean, 10, labels=False, duplicates="drop")
    decile_of_gene = deciles.to_dict()

    by_dec_all: dict[int, list[str]] = defaultdict(list)
    by_dec_annot: dict[int, list[str]] = defaultdict(list)
    for g, d in decile_of_gene.items():
        by_dec_all[int(d)].append(g)
        if g in annotated:
            by_dec_annot[int(d)].append(g)

    pool_all = pools_as_rows(by_dec_all, ctx.index)
    pool_annot = pools_as_rows(by_dec_annot, ctx.index)

    # 第 3 の対照: 発現量十分位 x PC1 寄与五分位の二重マッチ（docstring 参照）。
    # 既存の帰属解析は個人方向の主成分（右特異ベクトル）を使うが、ここで要るのは
    # 遺伝子ごとの寄与なので左特異ベクトルを取る。
    z = np.nan_to_num(ctx.Z)
    u, _s, _vt = np.linalg.svd(z - z.mean(axis=1, keepdims=True), full_matrices=False)
    pc1_abs = pd.Series(np.abs(u[:, 0]), index=ctx.genes)
    cell_of_gene: dict[str, int] = {}
    for d, gs in by_dec_all.items():
        s = pc1_abs.reindex(gs).dropna()
        if s.size < 5:
            continue
        q = pd.qcut(s, 5, labels=False, duplicates="drop")
        for g, k in q.items():
            cell_of_gene[g] = int(d) * 5 + int(k)
    by_cell: dict[int, list[str]] = defaultdict(list)
    for g, k in cell_of_gene.items():
        by_cell[k].append(g)
    thin = [k for k, gs in by_cell.items() if len(gs) < MIN_CELL]
    for k in thin:
        by_cell[k] = by_dec_all[k // 5]
    print(f"  PC1 二重マッチのセル {len(by_cell)}（うち発現量十分位に落としたもの {len(thin)}）")
    pool_pc1 = pools_as_rows(by_cell, ctx.index)
    for d in sorted(by_dec_all):
        print(f"  十分位 {d}: 全体 {len(by_dec_all[d]):5,} / 注釈のみ {len(by_dec_annot[d]):5,}")

    print("[3/4] セットごとに 2 つの対照で判定する")
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]
    # 主解析とは別の乱数系列。腕ごとに独立にする（docstring 参照）。
    gen_all, gen_ann, gen_pc1 = rng(11), rng(12), rng(13)

    rows = []
    for i, (name, (family, genes)) in enumerate(all_sets.items(), 1):
        if i % 500 == 0:
            print(f"  {i}/{len(all_sets)} ...")
        present = ctx.present(genes)
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        coverage = len(present) / len(genes) if genes else 0.0
        if coverage < filt["min_coverage"] and family != "anchor":
            continue
        pos = [g for g in present if g in decile_of_gene]
        if len(pos) < 2:
            continue

        ic = mean_pairwise_rho(ctx.S[ctx.idx(present)])
        dec_arr = np.array([decile_of_gene[g] for g in pos])
        n_all = draw_null_rho(ctx.S, pool_all, dec_arr, nc, gen_all)
        n_ann = draw_null_rho(ctx.S, pool_annot, dec_arr, nc, gen_ann)
        e_all = empirical_null(ic, n_all.tolist())
        e_ann = empirical_null(ic, n_ann.tolist())

        pos_pc1 = [g for g in present if g in cell_of_gene]
        if len(pos_pc1) >= 2:
            dec_pc1 = np.array([cell_of_gene[g] for g in pos_pc1])
            n_pc1 = draw_null_rho(ctx.S, pool_pc1, dec_pc1, nc, gen_pc1)
            e_pc1 = empirical_null(ic, n_pc1.tolist())
        else:
            e_pc1 = empirical_null(np.nan, [])
        rows.append({
            "set": name,
            "family": family,
            "n_genes_present": len(present),
            "internal_consistency": ic,
            "null_all_mean": e_all["null_mean"],
            "null_annot_mean": e_ann["null_mean"],
            "null_pc1_mean": e_pc1["null_mean"],
            "p_all": e_all["null_p_empirical"],
            "p_annot": e_ann["null_p_empirical"],
            "p_pc1": e_pc1["null_p_empirical"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  評価できたセットが 0 件")
        return 1

    print("[4/4] 多重比較補正して書き出す")
    for src, dst in (("p_all", "q_all"), ("p_annot", "q_annot"), ("p_pc1", "q_pc1")):
        ok = df[src].notna()
        df[dst] = np.nan
        if ok.sum() > 1:
            df.loc[ok, dst] = multipletests(df.loc[ok, src], method="fdr_bh")[1]
    df["pass_all"] = df["q_all"] < 0.05
    df["pass_annot"] = df["q_annot"] < 0.05
    df["pass_pc1"] = df["q_pc1"] < 0.05

    out = TABLES / "null_model_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n = len(df)
    print()
    print(f"評価セット {n:,}")
    print(f"  全遺伝子プールで合格   : {100*df.pass_all.mean():5.1f}%  ({df.pass_all.sum():,})")
    print(f"  注釈遺伝子プールで合格 : {100*df.pass_annot.mean():5.1f}%  ({df.pass_annot.sum():,})")
    print(f"  判定が変わったセット   : {int((df.pass_all != df.pass_annot).sum()):,}"
          f"（{100*(df.pass_all != df.pass_annot).mean():.1f}%）")
    print(f"  PC1 寄与そろえで合格   : {100*df.pass_pc1.mean():5.1f}%  ({df.pass_pc1.sum():,})")
    print(f"  対照の水準（中央値）   : 全体 {df.null_all_mean.median():.4f}"
          f" / 注釈のみ {df.null_annot_mean.median():.4f}"
          f" / PC1 そろえ {df.null_pc1_mean.median():.4f}")
    print()
    print("=== ファミリー別の合格率 ===")
    g = df.groupby("family").agg(
        n=("set", "size"),
        pass_all=("pass_all", lambda s: 100 * s.mean()),
        pass_annot=("pass_annot", lambda s: 100 * s.mean()),
        pass_pc1=("pass_pc1", lambda s: 100 * s.mean()),
    ).round(1)
    g["diff"] = (g.pass_annot - g.pass_all).round(1)
    print(g.to_string())
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
