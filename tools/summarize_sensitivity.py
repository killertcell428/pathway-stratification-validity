"""発現フィルタと正規化の感度分析をまとめる。

なぜ要るか
  2.2 節は「発現フィルタは解析の第一段である」と主張し、3.8 節は前処理で対照の水準が
  70 倍動くと示している。それなのに headline の 80.9% を出す主コホートのフィルタ
  （第 50 分位）は一度も振っておらず、正規化 3 通りも主成分の構造と対照の水準までしか
  通していなかった。**つまみは作ってあって回していない状態**を解消する。

読み方
  ここで見るのは絶対値ではなく、次の 3 つが水準をまたいで保つかどうかである。
    1. 条件効果は普遍か（各水準で 85% 以上か）
    2. 「条件効果はあるが個人間の共変動が対照を超えない」が最大の区画か
    3. ファミリー順位（complex と regulon が下位）が保たれるか

出力: results/tables/preprocessing_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"

# (ラベル, CSV への相対パス, 何を振ったか)
VARIANTS = [
    ("主コホート 第50分位（採用）", "gene_set_metrics.csv", "発現フィルタ"),
    ("主コホート 第40分位", "gene_set_metrics_p40.csv", "発現フィルタ"),
    ("主コホート 第60分位", "gene_set_metrics_p60.csv", "発現フィルタ"),
    ("GSE81046 TMM logCPM（採用）", "gse81046/gene_set_metrics.csv", "正規化"),
    ("GSE81046 log2(TPM+1)", "gse81046/gene_set_metrics_tpm.csv", "正規化"),
    ("GSE81046 TMM+分位正規化", "gse81046/gene_set_metrics_quantile.csv", "正規化"),
]


def control_count(d: pd.DataFrame) -> int | None:
    """その出力が対照を何個引いて作られたか。列がなければ旧版（20 個）。

    腕ごとに対照数が違うと、感度分析が「前処理の効果」ではなく
    「対照数の効果」を測ってしまう。実際に主解析だけ 10,000 個に増やしたとき、
    採用腕の合格率が 27.4% で両隣の腕が 38.6% と 35.9% になり、
    採用値が外れ値のように見える表ができた。ここで機械的に捕まえる。
    """
    if "n_control" not in d.columns:
        return None
    return int(d["n_control"].median())


def summarise(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    cond = d.delta_q < 0.05
    both = d.null_q.lt(0.05) & d.var_null_q.lt(0.05)
    fam = d.assign(passed=both).groupby("family").passed.mean() * 100
    ann = fam.drop(index=[i for i in ("data_derived", "anchor") if i in fam.index])
    return {
        "対照数": control_count(d),
        "評価セット": len(d),
        "条件効果あり(%)": round(100 * cond.mean(), 1),
        "個人間の共変動あり(%)": round(100 * both.mean(), 1),
        "条件効果のみ(%)": round(100 * (cond & ~both).mean(), 1),
        "最大の区画が条件効果のみ": bool((cond & ~both).mean() > max(
            both.mean(), (~cond & ~both).mean(), (cond & both).mean())),
        "complex(%)": round(float(ann.get("complex", float("nan"))), 1),
        "regulon(%)": round(float(ann.get("regulon", float("nan"))), 1),
        "下位2ファミリー": "+".join(sorted(ann.nsmallest(2).index)),
        "_fam": ann,
    }


def main() -> int:
    rows, fams = [], {}
    for label, rel, knob in VARIANTS:
        s = summarise(T / rel)
        if s is None:
            print(f"  未走行: {label}（{rel}）")
            continue
        fams[label] = s.pop("_fam")
        rows.append({"条件": label, "振った要素": knob, **s})
    if not rows:
        print("集計できる出力がない")
        return 1

    df = pd.DataFrame(rows)

    # 腕ごとの対照数がそろっているかを検査する。そろっていない表は
    # 「前処理の効果」ではなく「対照数の効果」を測ってしまう。
    for knob, g in df.groupby("振った要素"):
        counts = set(g["対照数"].dropna().astype(int))
        if len(g["対照数"].dropna()) < len(g):
            counts.add(20)  # 対照数の列がない出力は旧版（20 個）
        if len(counts) > 1:
            print(f"\n★ {knob} の腕で対照数がそろっていない: {sorted(counts)}")
            print("  そろえないと、前処理の効果と対照数の効果が混ざる。")
            print("  該当の腕を run_evaluation --suffix で再走行すること:")
            for _, r in g.iterrows():
                print(f"    {r['条件']:32s} 対照 {r['対照数']}")
            return 1

    df.to_csv(T / "preprocessing_sensitivity.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False))

    print("\n=== 採用値に対するファミリー順位の一致 ===")
    for knob, base in (("発現フィルタ", "主コホート 第50分位（採用）"),
                       ("正規化", "GSE81046 TMM logCPM（採用）")):
        if base not in fams:
            continue
        for label in [r["条件"] for r in rows if r["振った要素"] == knob and r["条件"] != base]:
            a, b = fams[base], fams[label]
            idx = list(a.index.intersection(b.index))
            r = spearmanr(a[idx], b[idx])
            print(f"  {label:26s} ρ = {r.statistic:.3f} (p = {r.pvalue:.3f}, {len(idx)} ファミリー)")

    print("\n=== 結論の三点が保つか ===")
    ok_cond = all(r["条件効果あり(%)"] >= 85 for r in rows)
    ok_region = all(r["最大の区画が条件効果のみ"] for r in rows)
    ok_bottom = len({r["下位2ファミリー"] for r in rows}) == 1
    print(f"  条件効果が 85% 以上: {'保つ' if ok_cond else '崩れる'}")
    print(f"  最大の区画が「条件効果のみ」: {'保つ' if ok_region else '崩れる'}")
    print(f"  下位 2 ファミリーが同一: {'保つ' if ok_bottom else '崩れる'}"
          f"（{sorted({r['下位2ファミリー'] for r in rows})}）")
    print(f"\n-> {T/'preprocessing_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
