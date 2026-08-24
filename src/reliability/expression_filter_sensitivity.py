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
    rows, family, n_ctrl = {}, {}, {}
    for label, suffix in LEVELS:
        path = TABLES / f"gene_set_metrics{suffix}.csv"
        if not path.exists():
            print(f"★ 見つからない: {path.name}（その水準の走行が済んでいない）")
            return 1
        df = pd.read_csv(path)
        # 対照数がそろっていない腕を混ぜると、閾値の効果ではなく対照数の効果を測る。
        # 実測: 採用腕だけ 10,000 個にしたとき、採用腕 27.4% に対し両隣が
        # 38.6% と 35.9%（対照 20 個のまま）で、採用値が外れ値に見える表ができた。
        n_ctrl[label] = int(df["n_control"].median()) if "n_control" in df.columns else 20
        rows[label] = summarize(df)
        both = (df.null_q < 0.05) & (df.var_null_q < 0.05)
        family[label] = (100 * df.assign(b=both).groupby("family").b.mean()).round(1)

    if len(set(n_ctrl.values())) > 1:
        print("★ 水準間で対照数がそろっていない:", n_ctrl)
        print("  該当の水準を T26_MATRIX_SUFFIX を付けて run_evaluation で再走行すること")
        return 1
    print(f"（各水準の対照数: {next(iter(n_ctrl.values())):,} 個）")

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
    # 本文が主張しているのは「下位に来るのは複合体とレギュロン」である。
    # 最上位の入れ替わりは合格率が 1 セット未満の差で決まることがあり
    # （実測: CPM>=2.0 で signature 71.4%（35 件中 25）に対し
    # data_derived 71.0%（31 件中 22））、それで「閾値に依存する」と出すと
    # 狼少年になる。判定は下位 2 ファミリーの一致で行い、最上位は情報として出す。
    ann = {label: fam[label].drop(index=[i for i in ("anchor",) if i in fam[label].index])
           for label in fam}
    bottoms = {label: tuple(sorted(v.nsmallest(2).index)) for label, v in ann.items()}
    same_bottom = len(set(bottoms.values())) == 1
    print(f"  下位 2 ファミリーが 3 水準で一致: {'○' if same_bottom else '×'}"
          f"（{sorted(set(bottoms.values()))}）")
    tops = {label: v.idxmax() for label, v in ann.items()}
    if len(set(tops.values())) == 1:
        print(f"  最上位も 3 水準で一致（{next(iter(set(tops.values())))}）")
    else:
        print(f"  最上位は水準で入れ替わる（{tops}）。合格率の差を確認すること")
    print(f"\n{'結論は閾値に依存しない' if ok and same_bottom else '★ 結論が閾値に依存する'}")

    out.to_csv(TABLES / "expression_filter_sensitivity.csv", encoding="utf-8")
    fam.to_csv(TABLES / "expression_filter_sensitivity_by_family.csv", encoding="utf-8")
    print(f"-> {TABLES / 'expression_filter_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
