"""被覆率フィルタの感度分析をまとめる。

被覆率 60% 未満のセットを落とすと「発現している遺伝子が少ないセット」が除かれるため、
結果が楽観側に歪む可能性がある。閾値を振った実行結果を並べて、主要な結論が
閾値に依存しないかを確認する。

前提: run_evaluation を --min-coverage / --suffix 付きで走らせた結果があること。
  python -m src.reliability.run_evaluation --min-coverage 0.3 --suffix _cov0.3
出力: results/tables/coverage_sensitivity.csv
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from ..common import TABLES


def summarize(df: pd.DataFrame) -> dict:
    sig = df["delta_q"] < 0.05
    coh = (df["null_q"] < 0.05) & (df["var_null_q"] < 0.05)
    out = {
        "n_sets": len(df),
        "cond_effect": float(sig.mean()),
        "coherent": float(coh.mean()),
        "cond_only": float((sig & ~coh).mean()),
        "both": float((sig & coh).mean()),
        "ic_median": float(df["internal_consistency"].median()),
        "coverage_median": float(df["coverage"].median()),
    }
    for fam in ["data_derived", "celltype", "pathway", "signature", "complex", "regulon"]:
        d = df[df.family == fam]
        out[f"coh_{fam}"] = float(((d["null_q"] < 0.05) & (d["var_null_q"] < 0.05)).mean()) if len(d) else float("nan")
    return out


def label_for(stem: str) -> str | None:
    """gene_set_metrics のファイル名から被覆率の水準ラベルを決める。None なら対象外。

    以前は gene_set_metrics*.csv を全部拾い、`_cov` に一致しないものはすべて
    「0.6 (既定)」にしていた。そのため発現フィルタ分位の腕（_p40 / _p60）も
    既定ラベルになり、辞書代入で最後に読まれた _p60 が正本の行を上書きしていた
    （実測: 既定行が 2,195 セットではなく p60 の 1,713 セットになっていた）。
    正本は接尾辞なしの 1 件だけなので、名前で厳密に判定する。
    """
    m = re.search(r"_cov([0-9.]+)$", stem)
    if m:
        return f"{float(m.group(1)):.1f}"
    if stem == "gene_set_metrics":
        return "0.6 (既定)"
    return None


def main() -> int:
    rows = {}
    for path in sorted(TABLES.glob("gene_set_metrics*.csv")):
        label = label_for(path.stem)
        if label is None:
            continue
        rows[label] = summarize(pd.read_csv(path))
    assert "0.6 (既定)" in rows, "正本の gene_set_metrics.csv が読めていない"

    df = pd.DataFrame(rows).T.sort_index()
    df.to_csv(TABLES / "coverage_sensitivity.csv", encoding="utf-8")

    pct = [c for c in df.columns if c not in ("n_sets", "ic_median", "coverage_median")]
    show = df.copy()
    show[pct] = (show[pct] * 100).round(1)
    show["n_sets"] = show["n_sets"].astype(int)
    show[["ic_median", "coverage_median"]] = show[["ic_median", "coverage_median"]].round(3)
    print("被覆率閾値ごとの主要指標（% 表示）")
    print(show.to_string())

    spread = (df["cond_only"].max() - df["cond_only"].min()) * 100
    print(f"\n「条件効果のみ（層別化不可）」の閾値間の振れ幅: {spread:.1f} ポイント")
    print(f"  最小 {df['cond_only'].min()*100:.1f}% ({df['cond_only'].idxmin()}) / "
          f"最大 {df['cond_only'].max()*100:.1f}% ({df['cond_only'].idxmax()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
