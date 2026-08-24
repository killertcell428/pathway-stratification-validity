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

from ..common import INTERIM, METADATA, TABLES, load_config
from ..scoring.methods import _rank_rows, _standardize_rows
from .run_evaluation import load_all_sets

PC1_THRESHOLD = 0.30    # 反映型解釈の目安。単一因子が 3 割未満なら支配的とは言えない


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

    rows = []
    for name, (family, genes) in sets.items():
        present = [g for g in genes if g in index]
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if len(present) / len(genes) < filt["min_coverage"] and family != "anchor":
            continue
        A = S[[index[g] for g in present]]
        A = A - A.mean(axis=1, keepdims=True)
        sv = np.linalg.svd(A, compute_uv=False)
        var_ratio = (sv ** 2) / float((sv ** 2).sum())
        rows.append({"set": name, "family": family, "n_genes": len(present),
                     "pc1_frac": float(var_ratio[0])})
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
    print(d.groupby("family").pc1_frac.agg(["size", "median"]).round(3).to_string())
    print(f"\n-> {TABLES / 'dimensionality.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
