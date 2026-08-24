"""反復測定信頼性の基準を通ったセットの内訳と非独立性を数える（3.10 節・2.2 節）。

なぜ要るか
  3.10 節と 2.2 節は「基準を通ったセットは互いに独立ではない」ことを、
  ファミリー内訳・Jaccard 係数・重複除去後の実質遺伝子数で述べている。
  ところがこの計算はコードとして残っておらず、原稿にだけ数値があった。
  対照を 20 個から 10,000 個に増やして通過セットが 15 件から変わったため、
  再計算できる形にしておかないと原稿の数値を追随させられない。

  同時に 3.10 節の「本研究自身の予測も外れた」の数値（接種直前と 7 日前で
  相関の大きさが何分の 1 になるか、対照超過 z が 2 を超える件数）も出す。
  ここは通過セットの集合が変われば全部動く。

出力: results/tables/passing_set_overlap.csv（1 行の要約）
      results/tables/passing_set_overlap_sets.csv（通過セットの一覧）
"""

from __future__ import annotations

import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from ..common import INTERIM, TABLES, load_config
from ..scoring.methods import _standardize_rows
from .metrics import pooled_set_score
from .retest_check import RETEST
from .run_evaluation import load_all_sets


REFERENCE_MARKERS = ["T Cells", "T Memory Cells", "T Cells Naive", "B Cells",
                     "NK Cells", "Monocytes", "Neutrophils", "Platelets"]


def _composition_correlation(passing, all_sets, genes) -> pd.DataFrame:
    """細胞種マーカー以外の通過セットと、血球組成マーカーとの順位相関。

    ラベルが経路やモジュールでも、遺伝子内容が血球系なら組成マーカーと相関する。
    絶対値の最大が大きければ「そのセットは実質的に組成を測っている」と言える。
    """
    b = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[1]}.parquet")
    Z = pd.DataFrame(_standardize_rows(b.to_numpy(dtype=np.float64)),
                     index=b.index, columns=b.columns)

    def score(gs: set[str] | list[str]) -> np.ndarray | None:
        idx = [g for g in gs if g in Z.index]
        if len(idx) < 2:
            return None
        return pooled_set_score(Z.loc[idx].to_numpy())

    refs = {}
    for name, (family, gl) in all_sets.items():
        if family != "celltype":
            continue
        label = name.split("|", 1)[1]
        if label in REFERENCE_MARKERS:
            s = score(gl)
            if s is not None:
                refs[label] = s
    if not refs:
        return pd.DataFrame()

    rows = []
    for _, r in passing.iterrows():
        if r["family"] == "celltype":
            continue
        s = score(genes.get(r["set"], []))
        if s is None:
            continue
        cors = {k: float(stats.spearmanr(s, v)[0]) for k, v in refs.items()}
        best = max(cors, key=lambda k: abs(cors[k]))
        rows.append({"set": r["set"], "family": r["family"], "icc": r["icc"],
                     "最大相関の相手": best, "最大相関": round(cors[best], 3),
                     **{f"rho_{k}": round(v, 3) for k, v in cors.items()}})
    return pd.DataFrame(rows)


def main() -> int:
    gs_cfg = load_config("gene_sets")
    alpha = load_config("analysis")["multiple_testing"]["alpha"]

    retest = pd.read_csv(TABLES / "retest_metrics.csv")
    pheno = pd.read_csv(TABLES / "phenotype_metrics.csv")
    passing = retest[retest["icc_q"] < alpha].copy()
    if passing.empty:
        print("反復測定の基準を通ったセットが 0 件")
        return 1

    all_sets = load_all_sets(gs_cfg)
    # 原稿が引く遺伝子数は「注釈された遺伝子」ではなく、この解析で実際に使えた
    # 遺伝子（コホートで検出できた分）である。retest_metrics の n_genes_present と
    # 突き合わせて、名寄せがずれていないか確認する。
    genes = {}
    for name in passing["set"]:
        if name not in all_sets:
            print(f"  [warn] {name} が遺伝子セット定義に見つからない")
            continue
        genes[name] = set(all_sets[name][1])

    fam = passing["family"].value_counts()
    total = sum(len(v) for v in genes.values())
    union = len(set().union(*genes.values())) if genes else 0
    jac = [
        len(genes[a] & genes[b]) / len(genes[a] | genes[b])
        for a, b in combinations(genes, 2)
        if genes[a] | genes[b]
    ]

    # 3.10 節: 接種直前と 7 日前で表現型相関がどれだけ落ちるか
    ph = pheno.set_index("set")
    hit = [s for s in passing["set"] if s in ph.index]
    rest = [s for s in ph.index if s not in set(passing["set"])]
    abs0 = ph.loc[hit, "rho_day0"].abs()
    abs0_rest = ph.loc[rest, "rho_day0"].abs()
    abs7 = ph.loc[hit, "rho_day-7"].abs()
    z0 = ph.loc[hit, "abs_rho_z"]
    z7 = ph.loc[hit, "abs_rho_z_day-7"]
    same_sign = int((np.sign(ph.loc[hit, "rho_day0"]) == np.sign(ph.loc[hit, "rho_day-7"])).sum())
    mw = stats.mannwhitneyu(abs0, abs0_rest, alternative="greater")

    # 細胞種マーカーとラベルされていない通過セットが、実は血球組成を測っていないかを
    # 確かめる。3.7 節の「整合性はラベルではなく遺伝子内容に対応する」が正しければ、
    # ラベルが経路でも、内容が血球系なら組成マーカーと強く相関するはずである。
    # ここで相関しなければ 3.9 節の「通過セットは組成に対応する」は成り立たない。
    comp = _composition_correlation(passing, all_sets, genes)

    rec = {
        "通過セット数": len(passing),
        "評価セット数": len(retest),
        "通過率(%)": round(100 * len(passing) / len(retest), 2),
        "細胞種マーカー数": int(fam.get("celltype", 0)),
        "反応経路数": int(fam.get("pathway", 0)),
        "その他数": int(len(passing) - fam.get("celltype", 0) - fam.get("pathway", 0)),
        "Jaccard中央値": round(float(np.median(jac)), 3) if jac else np.nan,
        "Jaccard最大": round(float(np.max(jac)), 3) if jac else np.nan,
        "合計遺伝子数": total,
        "重複除去後": union,
        "実質割合(%)": round(100 * union / total, 1) if total else np.nan,
        "接種直前_絶対相関_中央値": round(float(abs0.median()), 3),
        "残り_絶対相関_中央値": round(float(abs0_rest.median()), 3),
        "残りのセット数": len(rest),
        "MannWhitney_p": float(mw.pvalue),
        "7日前_絶対相関_中央値": round(float(abs7.median()), 3),
        "接種直前_z_中央値": round(float(z0.median()), 2),
        "7日前_z_中央値": round(float(z7.median()), 2),
        "接種直前_z2超え": int((z0 > 2).sum()),
        "7日前_z2超え": int((z7 > 2).sum()),
        "符号一致": same_sign,
        "減衰倍率": round(float(abs0.median() / abs7.median()), 1) if abs7.median() else np.nan,
        "非マーカー通過セット数": len(comp),
        "非マーカーの組成相関の絶対値の最小": (
            round(float(comp["最大相関"].abs().min()), 3) if len(comp) else np.nan),
    }
    pd.DataFrame([rec]).to_csv(TABLES / "passing_set_overlap.csv", index=False,
                              encoding="utf-8")

    detail = passing[["set", "family", "n_genes_present", "icc", "icc_null_mean",
                      "icc_p_empirical", "icc_q"]].copy()
    detail = detail.sort_values("icc", ascending=False)
    detail.to_csv(TABLES / "passing_set_overlap_sets.csv", index=False, encoding="utf-8")
    if len(comp):
        comp.to_csv(TABLES / "passing_set_composition.csv", index=False, encoding="utf-8")

    print(f"反復測定の基準を通ったセット {len(passing)} / {len(retest)} 件")
    print(f"  ファミリー内訳: {dict(fam)}")
    print(f"  Jaccard 中央値 {rec['Jaccard中央値']} / 最大 {rec['Jaccard最大']}")
    print(f"  合計 {total} 遺伝子 -> 重複除去後 {union} 遺伝子（実質 {rec['実質割合(%)']}%）")
    print(f"  接種直前の |rho| 中央値 {rec['接種直前_絶対相関_中央値']}"
          f"（残り {rec['残りのセット数']} 件は {rec['残り_絶対相関_中央値']}, "
          f"Mann-Whitney p = {rec['MannWhitney_p']:.2g}）")
    print(f"  7 日前では {rec['7日前_絶対相関_中央値']} に落ちる（{rec['減衰倍率']} 分の 1）")
    print(f"  対照超過 z の中央値 {rec['接種直前_z_中央値']} -> {rec['7日前_z_中央値']}、"
          f"z>2 は {rec['接種直前_z2超え']}/{len(passing)} -> {rec['7日前_z2超え']}/{len(passing)}")
    print(f"  符号は {rec['符号一致']}/{len(passing)} で保たれる")
    if len(comp):
        print(f"\n細胞種マーカー以外の通過セット {len(comp)} 件と血球組成マーカーの相関:")
        print(comp[["set", "family", "icc", "最大相関の相手", "最大相関"]].to_string(index=False))
        print("  ラベルが経路でも内容が血球系なら組成マーカーと相関する。"
              "絶対値が大きければ実質的に組成を測っている。")
    print(f"\n通過セット上位:")
    print(detail.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
