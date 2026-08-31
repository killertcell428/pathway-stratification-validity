"""遺伝子セットを反映型の尺度として解釈できるかを測る。

なぜ要るか
  本研究は「同じ潜在活性を測るセットなら構成遺伝子は個人間で共変動するはず」
  という前提に立って内部整合性を要求している。これは複数の指標が共通の潜在因子を
  反映する**反映型**の尺度には妥当だが、遺伝子が相補的・段階的・代償的に働く
  **形成型**の指標には当てはまらない（Bollen & Lennox 1991）。
  形成型なら遺伝子間相関が低くても合成スコアに外的妥当性がありうる。

  この区別を無視すると「すべての遺伝子セットに内部整合性を要求する」という
  擁護できない立場になる。そこで、反映型解釈がそもそも成立するセットが
  どれだけあるかを直接測る。

測るもの
  (1) 一次元性: 各セットの第 1 主成分が説明する分散の割合。
      反映型は単一の支配因子を前提とするので、これが低ければ前提が成り立たない。
  (2) 手法依存性: 一次元性が低いセットで、スコアリング手法を変えると
      個人の順位が変わるか。形成型指標には理論的に正当化された重みが要るが、
      遺伝子セットスコアは等重み（z 平均）か手法依存の暗黙の重み（順位・PC1）を
      使う。手法で順位が変わるなら、等重みの合成は形成型指標としても
      正当化されない。

  (1) と (2) が揃えば、反映型にも形成型にも当てはまらないセットが特定できる。
  そこでは内部整合性を要求することが不当ではなくなる。

  **ただし第 1 主成分の説明率は、それ自体がセットの遺伝子数に強く依存する。**
  遺伝子が少なければ 1 因子で説明できる割合は自動的に上がる（実測で
  pc1_frac と遺伝子数の順位相関は −0.807）。したがってファミリー別の
  pc1_frac の中央値をそのまま並べると、CORUM 複合体（遺伝子数中央値 4）が高く
  細胞種マーカー（105）が低いという順序が出るが、これはサイズの差を
  読んでいるにすぎない。そこでサイズと発現量十分位をそろえたランダムセットの
  pc1_frac も計算し、**超過**で比べる。

限界
  形成型の妥当性は本来、外的な基準との関連（criterion validity）で示す。
  本研究の表現型解析（3.10 節）は n = 42 で検出力が足りないため、
  形成型妥当性が「ない」ことは示せない。示せるのは重みの要件が満たされて
  いないことと、順位が手法に依存することである。

出力: results/tables/dimensionality.csv
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, TABLES, gene_mean_path, load_config, rng
from ..scoring.methods import _rank_rows, _standardize_rows
from .run_evaluation import load_all_sets

PC1_THRESHOLD = 0.30    # 反映型解釈の目安。単一因子が 3 割未満なら支配的とは言えない
N_PC1_CONTROL = 200     # pc1_frac の対照数。SVD を回すので 10,000 は要らない


def main() -> int:
    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    filt = gs_cfg["filters"]

    expr = pd.read_parquet(INTERIM / f"expr_{cfg['conditions']['resting']}.parquet")
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    X = expr[val]
    S = _standardize_rows(_rank_rows(X.to_numpy(dtype=np.float64)))
    index = {g: i for i, g in enumerate(X.index)}
    print(f"検証側 {len(val)} 名 / 遺伝子 {X.shape[0]:,}")

    metrics = pd.read_csv(TABLES / "gene_set_metrics.csv")
    sets = load_all_sets(gs_cfg)

    def pc1_of(idx: list[int]) -> float:
        A = S[idx]
        A = A - A.mean(axis=1, keepdims=True)
        sv = np.linalg.svd(A, compute_uv=False)
        sq = sv ** 2
        tot = float(sq.sum())
        return float(sq[0] / tot) if tot > 0 else float("nan")

    # 対照プール: 発現量十分位ごとの行番号
    gm = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gm = gm.loc[gm.index.intersection(X.index)]
    dec = pd.qcut(gm, 10, labels=False, duplicates="drop")
    pool: dict[int, list[int]] = {}
    for g, dd in dec.items():
        pool.setdefault(int(dd), []).append(index[g])
    gen = rng(31)

    rows = []
    for name, (family, genes) in sets.items():
        present = [g for g in genes if g in index]
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if len(present) / len(genes) < filt["min_coverage"] and family != "anchor":
            continue
        obs = pc1_of([index[g] for g in present])
        # サイズと発現量十分位をそろえた対照の pc1_frac
        decs = [int(dec[g]) for g in present if g in dec.index]
        null = []
        if len(decs) >= filt["min_genes"]:
            for _ in range(N_PC1_CONTROL):
                draw = [pool[dd][int(gen.integers(len(pool[dd])))] for dd in decs]
                v = pc1_of(draw)
                if v == v:
                    null.append(v)
        null_med = float(np.median(null)) if null else float("nan")
        p_emp = ((sum(1 for v in null if v >= obs) + 1) / (len(null) + 1)
                 if null else float("nan"))
        rows.append({"set": name, "family": family, "n_genes": len(present),
                     "pc1_frac": obs, "pc1_null_median": null_med,
                     "pc1_excess": obs - null_med, "pc1_p_empirical": p_emp})
    d = pd.DataFrame(rows).merge(
        metrics[["set", "method_agreement_min", "internal_consistency"]],
        on="set", how="left")
    d.to_csv(TABLES / "dimensionality.csv", index=False, encoding="utf-8")

    print(f"\n=== 一次元性（第 1 主成分の説明分散、{len(d)} セット）===")
    print(f"  中央値 {d.pc1_frac.median():.1%}"
          f"（四分位 {d.pc1_frac.quantile(.25):.1%}〜{d.pc1_frac.quantile(.75):.1%}）")
    for th in (0.3, 0.5):
        n = int((d.pc1_frac >= th).sum())
        print(f"  PC1 が {th:.0%} 以上: {n} / {len(d)} ({n/len(d):.1%})")

    lo = d[d.pc1_frac < PC1_THRESHOLD]
    hi = d[d.pc1_frac >= PC1_THRESHOLD]
    print(f"\n=== 手法依存性（最も不一致な手法対での順位相関の中央値）===")
    print(f"  PC1 < {PC1_THRESHOLD:.0%}（{len(lo)} セット）: "
          f"{lo.method_agreement_min.median():.3f}")
    print(f"  PC1 >= {PC1_THRESHOLD:.0%}（{len(hi)} セット）: "
          f"{hi.method_agreement_min.median():.3f}")

    print("\n=== ファミリー別の一次元性 ===")
    print(d.groupby("family").agg(
        n=("set", "size"), n_genes=("n_genes", "median"),
        pc1=("pc1_frac", "median"), pc1_null=("pc1_null_median", "median"),
        excess=("pc1_excess", "median")).round(3).to_string())
    print()
    print(f"  pc1_frac と遺伝子数の順位相関: "
          f"{d[['pc1_frac', 'n_genes']].corr(method='spearman').iloc[0, 1]:.3f}")
    print(f"  対照を上回るセット（経験 p < 0.05）: "
          f"{100 * d.pc1_p_empirical.lt(0.05).mean():.1f}%")
    print(f"\n-> {TABLES / 'dimensionality.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
