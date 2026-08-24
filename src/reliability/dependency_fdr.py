"""BH-FDR の独立性仮定を外したときに判定がどれだけ動くかを測る（3.11 節）。

なぜ要るか
  本研究の検定は互いに独立ではない。遺伝子セットは遺伝子を共有しており、
  同じ遺伝子を含む 2 つのセットの共変動は正に相関する。Benjamini-Hochberg 法は
  検定統計量の独立、または正の回帰依存（PRDS）のもとで FDR を制御する。
  遺伝子共有による正の相関は PRDS に当てはまる場合が多いが、本研究の依存構造が
  PRDS を満たすことは示していない。「示していないなら、外した場合に何が起きるかを
  出しておく」ほうが、査読で仮定の妥当性を争うより短い。

  Benjamini-Yekutieli 法は依存構造を一切仮定せずに FDR を制御する。代償として
  閾値を Σ_{i=1..m} 1/i 倍だけ厳しくする。m = 2,195 なら約 8.2 倍である。
  つまりこれは「最悪ケース」の検査であり、ここで結論が生き残れば
  依存構造の議論そのものが不要になる。

  重要なのは、本研究の主要な主張が「合格が少ない」という向きであること。
  補正を厳しくすれば合格はさらに減るので、共変動・信頼性・表現型相関の側の
  結論は BY でも自動的に保たれる。逆向きに効くのは条件効果（90.8% が有意）
  だけなので、そこが崩れないかを実際に見る必要がある。

出力: results/tables/dependency_fdr.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..common import TABLES

FDR = 0.05

# (表示名, ファイル, p 値の列, 主張の向き)
# 向き = "少ないことが主張" なら BY で厳しくしても主張は保たれる。
# 向き = "多いことが主張" なら BY で崩れうるので、そこを見るのが本題。
TARGETS: list[tuple[str, str, str, str]] = [
    ("主コホート 条件効果", "gene_set_metrics.csv", "delta_p", "多い"),
    ("主コホート 共変動（発現量そろえ対照）", "gene_set_metrics.csv",
     "null_p_empirical", "少ない"),
    ("主コホート 共変動（分散そろえ対照）", "gene_set_metrics.csv",
     "var_null_p_empirical", "少ない"),
    ("GSE81046 条件効果", "gse81046/gene_set_metrics.csv", "delta_p", "多い"),
    ("GSE81046 共変動（発現量そろえ対照）", "gse81046/gene_set_metrics.csv",
     "null_p_empirical", "少ない"),
    ("GSE81046 共変動（分散そろえ対照）", "gse81046/gene_set_metrics.csv",
     "var_null_p_empirical", "少ない"),
    ("反復測定 ICC", "retest_metrics.csv", "icc_p_empirical", "少ない"),
    ("反復測定 内部整合性", "retest_metrics.csv", "ic_p_empirical", "少ない"),
    ("反復測定 再測定順位相関", "retest_metrics.csv",
     "retest_rho_p_empirical", "少ない"),
    ("表現型相関（接種直前）", "phenotype_metrics.csv",
     "abs_rho_p_empirical", "少ない"),
    ("表現型相関（接種 7 日前）", "phenotype_metrics.csv",
     "abs_rho_p_empirical_day-7", "少ない"),
]


def _rows() -> list[dict]:
    out = []
    for label, fname, col, direction in TARGETS:
        path = TABLES / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if col not in df.columns:
            print(f"  [飛ばす] {label}: 列 {col} がない")
            continue
        p = df[col]
        ok = p.notna()
        m = int(ok.sum())
        if m < 2:
            continue
        q_bh = multipletests(p[ok], alpha=FDR, method="fdr_bh")[1]
        q_by = multipletests(p[ok], alpha=FDR, method="fdr_by")[1]
        n_bh = int((q_bh < FDR).sum())
        n_by = int((q_by < FDR).sum())
        # BY の代償は閾値の Σ 1/i 倍。理屈どおりの値かを併記して読者が検算できる形にする。
        penalty = float(np.sum(1.0 / np.arange(1, m + 1)))
        out.append({
            "対象": label,
            "主張の向き": direction,
            "検定数": m,
            "BH 通過": n_bh,
            "BY 通過": n_by,
            "BH 通過率": n_bh / m,
            "BY 通過率": n_by / m,
            "差（ポイント）": 100 * (n_bh - n_by) / m,
            "BY の閾値の厳しさ（倍）": penalty,
        })
    return out


def _dissociation() -> list[dict]:
    """乖離の割合そのものを BH と BY で作り直す。

    論文の中心の数値は「条件効果はあるが共変動の対照を通らない」割合である。
    条件効果は BY で減り、共変動も BY で減るので、乖離の割合は両側から動く。
    どちらに動くかは机上では決まらないので実測する。
    """
    out = []
    for tag, fname in (("主コホート", "gene_set_metrics.csv"),
                       ("GSE81046", "gse81046/gene_set_metrics.csv")):
        path = TABLES / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        m = len(df)
        got = {}
        for method in ("fdr_bh", "fdr_by"):
            q = {}
            for col in ("delta_p", "null_p_empirical", "var_null_p_empirical"):
                p = df[col]
                ok = p.notna()
                qq = pd.Series(np.nan, index=df.index)
                qq[ok] = multipletests(p[ok], alpha=FDR, method=method)[1]
                q[col] = qq
            cond = q["delta_p"] < FDR
            coh = (q["null_p_empirical"] < FDR) & (q["var_null_p_empirical"] < FDR)
            got[method] = {
                "条件効果あり(%)": 100 * float(cond.mean()),
                "共変動あり(%)": 100 * float(coh.mean()),
                "条件効果のみ(%)": 100 * float((cond & ~coh).mean()),
            }
        for k in got["fdr_bh"]:
            out.append({"コホート": tag, "量": k, "検定数": m,
                        "BH": got["fdr_bh"][k], "BY": got["fdr_by"][k],
                        "差（ポイント）": got["fdr_by"][k] - got["fdr_bh"][k]})
    return out


def main() -> int:
    print("[1/2] 各判定を BH と BY で作り直す")
    rows = _rows()
    if not rows:
        print("  対象がない")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "dependency_fdr.csv", index=False, encoding="utf-8")
    for r in rows:
        print(f"  {r['対象']:38s} m={r['検定数']:5d}  "
              f"BH {r['BH 通過']:5d} ({100 * r['BH 通過率']:5.2f}%)  "
              f"BY {r['BY 通過']:5d} ({100 * r['BY 通過率']:5.2f}%)  "
              f"厳しさ {r['BY の閾値の厳しさ（倍）']:.2f} 倍")

    print("[2/2] 乖離の割合を作り直す")
    dis = pd.DataFrame(_dissociation())
    dis.to_csv(TABLES / "dependency_fdr_regions.csv", index=False, encoding="utf-8")
    for _, r in dis.iterrows():
        print(f"  {r['コホート']:9s} {r['量']:18s} "
              f"BH {r['BH']:5.1f}  BY {r['BY']:5.1f}  "
              f"差 {r['差（ポイント）']:+5.1f} ポイント")
    print("→ results/tables/dependency_fdr.csv, dependency_fdr_regions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
