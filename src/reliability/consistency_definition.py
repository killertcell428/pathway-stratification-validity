"""内部整合性の定義を Spearman から Pearson に替えて、結論が保つかを確かめる。

なぜ要るか
  本研究の内部整合性は「セット内の遺伝子ペアの個人間 Spearman 相関の平均」1 本である。
  二次指標として並べた Cronbach の α と折半法は独立な三角測量になっていない。
  α は平均ペア相関とサイズの決定論的関数（3.6 節が式を書いている）で、折半法も
  同じ標準化順位行列の上で計算される。**論文全体の内部整合性は事実上 1 つの数である。**

  しかもその 1 つが順位ベースなのに、整合性が妥当性を保証しようとしているスコア
  （z 平均と PLAGE）は線形スケールで動く。線形スケールの内部整合性が論文のどこにも無い。

  そして塞ぐコストがほぼゼロである。`ScoringContext` は行標準化済みの `Z` を
  すでに保持しているので、`mean_pairwise_rho(ctx.Z[idx])` がそのまま
  平均ペア Pearson 相関になる。**「1 行で振れるのに振っていない」状態を解消する。**

やること
  同じセット・同じ対照の引き（乱数種 2、run_evaluation と同一）で、
  Spearman 版と Pearson 版の両方を計算し、次の 3 つが保つかを見る。
    1. 「条件効果はあるが個人間の共変動が対照を超えない」割合
    2. ファミリー別の合格率の順位
    3. 個別セットの合否がどれだけ入れ替わるか

出力: results/tables/consistency_definition.csv
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import METADATA, TABLES, expr_path, gene_mean_path, load_config, rng
from ..scoring.methods import ScoringContext
from .metrics import null_rho_multi  # noqa: F401
from .metrics import empirical_null, matched_random_sets, mean_pairwise_rho
from .run_evaluation import load_all_sets


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    resting = cfg["conditions"]["resting"]

    expr = pd.read_parquet(expr_path(resting))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    rest_val = expr[val]
    ctx = ScoringContext(rest_val)
    print(f"検証側 {len(val)} 名 / 遺伝子 {rest_val.shape[0]:,}")

    gene_mean = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(rest_val.index)]
    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    by_dec: dict[int, list[str]] = defaultdict(list)
    for g, d in dec.items():
        by_dec[int(d)].append(g)

    pool_by_decile = {
        int(d): np.array([ctx.index[g] for g in gs if g in ctx.index], dtype=np.int64)
        for d, gs in by_dec.items()
    }
    assert not np.isnan(ctx.S).any() and not np.isnan(ctx.Z).any(),         "S または Z に NaN。行和によるまとめ計算の前提が崩れる"
    all_sets = load_all_sets(gs_cfg)
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]
    # 乱数種は run_evaluation と同じ 2。対照の引きを揃えないと定義の差と引きの差が混ざる。
    gen = rng(2)

    rows = []
    for name, (family, genes) in all_sets.items():
        present = ctx.present(genes)
        cov = len(present) / len(genes) if genes else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        idx = ctx.idx(present)
        # S は順位を標準化した行列、Z は値を標準化した行列。
        # 同じ関数に通すだけで Spearman 版と Pearson 版になる。
        ic_s = mean_pairwise_rho(ctx.S[idx])
        ic_p = mean_pairwise_rho(ctx.Z[idx])
        # 同じ対照セット群を S（順位）と Z（値）の両方に当てる。
        # 別々に引くと「定義の差」と「引きの差」が混ざる。
        dec_pos = np.array([dec[g] for g in present if g in dec], dtype=np.int64)
        nulls = null_rho_multi({"s": ctx.S, "z": ctx.Z}, pool_by_decile, dec_pos, nc, gen)
        ns = empirical_null(ic_s, nulls["s"].tolist())
        np_ = empirical_null(ic_p, nulls["z"].tolist())
        rows.append({
            "set": name, "family": family, "n_genes_present": len(present),
            "ic_spearman": ic_s, "ic_pearson": ic_p,
            "null_spearman": ns["null_mean"], "null_pearson": np_["null_mean"],
            "z_spearman": ns["null_z"], "z_pearson": np_["null_z"],
            "p_spearman": ns["null_p"], "p_pearson": np_["null_p"],
            "pe_spearman": ns["null_p_empirical"], "pe_pearson": np_["null_p_empirical"],
            "skew_spearman": ns["null_skew"], "skew_pearson": np_["null_skew"],
        })

    df = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    for tag in ("spearman", "pearson"):
        # BH-FDR は経験 p にかける（主解析と同じ土台にそろえる）
        ok = df[f"pe_{tag}"].notna()
        df[f"q_{tag}"] = np.nan
        df.loc[ok, f"q_{tag}"] = multipletests(df.loc[ok, f"pe_{tag}"], method="fdr_bh")[1]
        df[f"pass_{tag}"] = df[f"q_{tag}"].lt(0.05)
    df.to_csv(TABLES / "consistency_definition.csv", index=False, encoding="utf-8")

    print(f"\n=== 定義を替えたときの合格率（単一対照、{len(df)} セット）===")
    for tag in ("spearman", "pearson"):
        print(f"  {tag:9s} 合格 {100*df[f'pass_{tag}'].mean():5.1f}%  "
              f"整合性の中央値 {df[f'ic_{tag}'].median():.4f}  "
              f"対照の中央値 {df[f'null_{tag}'].median():.4f}")

    flip = (df.pass_spearman != df.pass_pearson)
    print(f"\n  個別セットの合否が入れ替わる: {int(flip.sum())} / {len(df)} "
          f"({100*flip.mean():.1f}%)")
    print(f"    Spearman のみ合格: {int((df.pass_spearman & ~df.pass_pearson).sum())}")
    print(f"    Pearson のみ合格: {int((~df.pass_spearman & df.pass_pearson).sum())}")

    print("\n=== ファミリー別の合格率 ===")
    fam = df.groupby("family").agg(
        n=("set", "size"),
        spearman=("pass_spearman", lambda s: round(100 * s.mean(), 1)),
        pearson=("pass_pearson", lambda s: round(100 * s.mean(), 1)))
    print(fam.to_string())
    ann = fam.drop(index=[i for i in ("data_derived", "anchor") if i in fam.index])
    r = spearmanr(ann.spearman, ann.pearson)
    print(f"  ファミリー順位の一致（注釈由来 {len(ann)} 件）: {r.statistic:.3f} (p = {r.pvalue:.3f})")
    print(f"\n-> {TABLES/'consistency_definition.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
