"""発現フィルタ閾値（CPM）を振ったときに結論が変わらないかを確認する。

なぜ要るか。本研究は「近ゼロ発現の遺伝子を残すと個人間相関が水増しされる」と
主張している（実際、TPM 0.01 で切った初期版では 25,046 個の未発現遺伝子が残り、
第 1 主成分が平均発現量と rho = -0.973 で相関した）。その主張をする以上、
採用した閾値そのものが結論を作っていないことを示す義務がある。

比較するのは 3 水準。CPM >= 0.5 / 1.0（本文採用）/ 2.0 を、いずれも
「50% 以上の個人で満たす」条件で適用する。1.0 は edgeR filterByExpr の慣行。

前提: 各水準の gene_set_metrics{suffix}.csv が作られていること。
  T26_DATASET=gse81046 T26_MATRIX_SUFFIX=_cpm0.5 \\
    python -m src.preprocessing.build_rnaseq_matrix --reuse-full --min-value 0.5
  （同じ接尾辞で derive_modules → run_evaluation --suffix _cpm0.5）

出力: results/tables/gse81046/expression_filter_sensitivity.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from ..common import TABLES

# (ラベル, gene_set_metrics のファイル接尾辞)
LEVELS = [("CPM >= 0.5", "_cpm0.5"), ("CPM >= 1.0（本文）", ""), ("CPM >= 2.0", "_cpm2.0")]


def summarize(df: pd.DataFrame) -> dict:
    cond = df.delta_q < 0.05
    both = (df.null_q < 0.05) & (df.var_null_q < 0.05)
    return {
        "評価セット": len(df),
        "条件効果あり(%)": round(100 * cond.mean(), 1),
        "内部整合性あり(%)": round(100 * both.mean(), 1),
        "条件効果のみ(%)": round(100 * (cond & ~both).mean(), 1),
        "両方(%)": round(100 * (cond & both).mean(), 1),
        "整合性のみ(%)": round(100 * (~cond & both).mean(), 1),
        "どちらもなし(%)": round(100 * (~cond & ~both).mean(), 1),
        "整合性の中央値": round(df.internal_consistency.median(), 4),
        "対照の中央値": round(df.null_mean.median(), 4),
    }


def main() -> int:
    rows, family = {}, {}
    for label, suffix in LEVELS:
        path = TABLES / f"gene_set_metrics{suffix}.csv"
        if not path.exists():
            print(f"★ 見つからない: {path.name}（その水準の走行が済んでいない）")
            return 1
        df = pd.read_csv(path)
        rows[label] = summarize(df)
        both = (df.null_q < 0.05) & (df.var_null_q < 0.05)
        family[label] = (100 * df.assign(b=both).groupby("family").b.mean()).round(1)

    out = pd.DataFrame(rows).T
    print("=== 発現フィルタ閾値の感度分析（GSE81046, RNA-seq）===")
    print(out.to_string())

    fam = pd.DataFrame(family)
    print("\n=== ファミリー別 2 対照合格率(%) ===")
    print(fam.to_string())

    # 結論が保たれているかを機械判定する。目視で「だいたい同じ」と言わないため。
    print("\n=== 結論の保持チェック ===")
    parts = ["条件効果のみ(%)", "両方(%)", "整合性のみ(%)", "どちらもなし(%)"]
    ok = True
    for label, r in rows.items():
        c1 = r["条件効果あり(%)"] > r["内部整合性あり(%)"]
        c2 = max(parts, key=lambda k: r[k]) == "条件効果のみ(%)"
        print(f"  {label:20s} 条件効果 > 整合性: {'○' if c1 else '×'}"
              f" / 最大区画が「条件効果のみ」: {'○' if c2 else '×'}")
        ok = ok and c1 and c2
    order = [fam[label].sort_values(ascending=False).index.tolist() for label in fam]
    same_top = len({o[0] for o in order}) == 1
    print(f"  ファミリー順位の最上位が 3 水準で一致: {'○' if same_top else '×'}"
          f"（{sorted({o[0] for o in order})}）")
    print(f"\n{'結論は閾値に依存しない' if ok and same_top else '★ 結論が閾値に依存する'}")

    out.to_csv(TABLES / "expression_filter_sensitivity.csv", encoding="utf-8")
    fam.to_csv(TABLES / "expression_filter_sensitivity_by_family.csv", encoding="utf-8")
    print(f"-> {TABLES / 'expression_filter_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
