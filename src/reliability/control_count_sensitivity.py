"""対照セット数と p 値の作り方が合格判定を変えるかを確かめる（感度分析）。

なぜ要るか
  合格判定には 2 つの選択が入っている。(1) 対照を何個引くか、(2) 対照分布から
  p 値をどう作るか（正規近似か経験分布か）。旧版は 20 個 + 正規近似だった。
  経験 p は 1/(B+1) より小さくならず、20 個では下限 0.048 で BH-FDR を通せない。
  そのため平均と SD から正規近似の p を作っていた。しかし平均ペア Spearman の
  帰無分布は上下に有界で右に歪むので、正規近似は右裾で甘い側に外れる。

  この 2 つの選択を分離して測る。同じ対照の引きに対して、
    - 正規近似の p での合格率
    - 経験 p での合格率
  を B = 20 から 50,000 まで並べる。モンテカルロ誤差なら B を増やすと収束するが、
  近似の誤りなら B をいくら増やしても正規近似側は動かない。どちらなのかが分かる。

なぜ全数を再走行しないか
  ここで測るのは絶対的な合格率ではなく「B と p の作り方で合格率がどう動くか」である。
  同じセット集合を条件だけ替えて比べれば足りるので、層化抽出した部分集合で測る。
  合格率は本文の全数（2,195 セット）の値とは一致しない。抽出した部分集合に
  BH-FDR をかけているため検定数が違う。比較可能なのは同じ行の中の列である。

出力
  results/tables/control_count_sensitivity.csv       条件別の要約
  results/tables/control_count_sensitivity_sets.csv  セット別の z と p
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..common import (METADATA, TABLES, expr_path, gene_mean_path, load_config,
                      rng)
from ..scoring.methods import ScoringContext
from .metrics import (draw_null_rho, empirical_null, mean_pairwise_rho_fast,
                      pools_as_rows)
from .run_evaluation import load_all_sets

# 20 は旧版、10,000 は現行の主解析。50,000 は「主解析の値がまだ動くか」の確認用。
CONTROL_COUNTS = (20, 100, 500, 2_000, 10_000, 50_000)
N_SETS = 300          # ファミリー比率を保って抽出する数
FDR = 0.05


def build_pools(rest_val: pd.DataFrame, gene_index: dict[str, int]):
    """発現量分位と個人間分散分位、2 種類の抽出プールを作る。

    run_evaluation.main と同じ手順にそろえる。ここがずれると
    感度分析の対照が本文の対照と別物になり、比較の意味がなくなる。
    """
    gene_mean = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(rest_val.index)]

    def pools(values: pd.Series):
        dec = pd.qcut(values, 10, labels=False, duplicates="drop").to_dict()
        by_dec: dict[int, list[str]] = defaultdict(list)
        for g, d in dec.items():
            by_dec[int(d)].append(g)
        return {g: int(d) for g, d in dec.items()}, pools_as_rows(by_dec, gene_index)

    return (pools(gene_mean),
            pools(rest_val.loc[gene_mean.index].var(axis=1)))


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    filt = gs_cfg["filters"]

    expr = pd.read_parquet(expr_path(cfg["conditions"]["resting"]))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    rest_val = expr[val]
    ctx = ScoringContext(rest_val)
    if np.isnan(ctx.S).any():
        print("S に NaN がある。行和によるまとめ計算の前提が崩れる")
        return 1
    print(f"検証側 {len(val)} 名")

    (expr_dec, expr_pool), (var_dec, var_pool) = build_pools(rest_val, ctx.index)

    # 本文で評価した 2,195 セットと同じフィルタを通す
    eligible = []
    for name, (family, genes) in load_all_sets(gs_cfg).items():
        present = ctx.present(genes)
        cov = len(present) / len(genes) if genes else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        eligible.append((name, family, present))
    print(f"フィルタ通過 {len(eligible)} セット")

    # ファミリー比率を保って抽出する。合格率はファミリーで大きく違うので、
    # 単純無作為抽出だと合格率の水準がファミリー構成に振られる。
    gen = rng(7)
    df_e = pd.DataFrame({"i": range(len(eligible)),
                         "family": [f for _, f, _ in eligible]})
    take = []
    for fam, g in df_e.groupby("family"):
        k = max(1, round(N_SETS * len(g) / len(df_e)))
        take += list(gen.choice(g["i"].to_numpy(), size=min(k, len(g)), replace=False))
    sample = [eligible[i] for i in sorted(take)]
    print(f"抽出 {len(sample)} セット（対照は最大 {max(CONTROL_COUNTS):,} 個）")

    # 対照は最大数を一度だけ引き、少ない条件はその先頭を使う。
    # 条件ごとに引き直すと、対照数の効果と乱数の効果が混ざる。
    nc_max = max(CONTROL_COUNTS)
    recs = []
    for j, (name, family, present) in enumerate(sample, 1):
        if j % 25 == 0:
            print(f"  {j}/{len(sample)} ...", flush=True)
        obs = mean_pairwise_rho_fast(ctx.S, ctx.idx(present))
        row = {"set": name, "family": family, "n_genes": len(present), "ic": obs}
        for kind, dec, pool in (("expr", expr_dec, expr_pool),
                                ("var", var_dec, var_pool)):
            dec_pos = np.array([dec[g] for g in present if g in dec], dtype=np.int64)
            vals = draw_null_rho(ctx.S, pool, dec_pos, nc_max, gen)
            for nc in CONTROL_COUNTS:
                res = empirical_null(obs, vals[:nc].tolist())
                row[f"{kind}_z_{nc}"] = res["null_z"]
                row[f"{kind}_pn_{nc}"] = res["null_p"]            # 正規近似
                row[f"{kind}_pe_{nc}"] = res["null_p_empirical"]  # 経験分布
                row[f"{kind}_skew_{nc}"] = res["null_skew"]
        recs.append(row)
    d = pd.DataFrame(recs)

    # 本文と同じ判定: 2 種類の対照の両方で BH-FDR を通ること
    out = []
    for nc in CONTROL_COUNTS:
        for how, tag in (("正規近似", "pn"), ("経験分布", "pe")):
            passes = np.ones(len(d), dtype=bool)
            for kind in ("expr", "var"):
                p = d[f"{kind}_{tag}_{nc}"].to_numpy(dtype=float)
                ok = np.isfinite(p)
                q = np.full(len(p), np.nan)
                q[ok] = multipletests(p[ok], alpha=FDR, method="fdr_bh")[1]
                passes &= np.nan_to_num(q, nan=1.0) < FDR
                d[f"{kind}_q{tag}_{nc}"] = q
            d[f"pass_{tag}_{nc}"] = passes
            out.append({
                "対照数": nc,
                "p 値の作り方": how,
                "経験 p の下限": round(1 / (nc + 1), 5),
                "合格セット数": int(passes.sum()),
                "合格率": round(float(passes.mean()), 4),
                "発現量マッチ z の中央値": round(float(d[f"expr_z_{nc}"].median()), 3),
                "帰無分布の歪度の中央値": round(float(d[f"expr_skew_{nc}"].median()), 3),
            })
    summary = pd.DataFrame(out)

    d.to_csv(TABLES / "control_count_sensitivity_sets.csv", index=False, encoding="utf-8")
    summary.to_csv(TABLES / "control_count_sensitivity.csv", index=False, encoding="utf-8")
    print(f"\n=== 対照数 x p 値の作り方（同一の {len(d)} セット）===")
    print(summary.to_string(index=False))

    # 読み方を出力に書いておく（表だけ見て誤読されないように）
    pn = summary[summary["p 値の作り方"] == "正規近似"].set_index("対照数")["合格率"]
    pe = summary[summary["p 値の作り方"] == "経験分布"].set_index("対照数")["合格率"]
    print("\n読み方")
    print(f"  正規近似の合格率は B=20 で {pn[20]:.3f}、B={nc_max:,} で {pn[nc_max]:.3f}。"
          "B を増やしても動かないなら、差はモンテカルロ誤差ではない")
    print(f"  経験分布の合格率は B=2,000 で {pe[2000]:.3f}、B={nc_max:,} で {pe[nc_max]:.3f}。"
          "ここが収束していれば主解析の B=10,000 で足りている")
    return 0


if __name__ == "__main__":
    sys.exit(main())
