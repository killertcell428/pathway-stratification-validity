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
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..common import INTERIM, TABLES, gene_mean_path, load_config, rng
from ..scoring.methods import _standardize_rows
from .metrics import pooled_set_score
from .retest_check import RETEST
from .run_evaluation import load_all_sets


REFERENCE_MARKERS = ["T Cells", "T Memory Cells", "T Cells Naive", "B Cells",
                     "NK Cells", "Monocytes", "Neutrophils", "Platelets"]


N_COMP_CONTROL = 1_000   # 組成相関の対照数。17 件 x 1,000 なので 10,000 は要らない


def _composition_correlation(passing, all_sets, genes, n_control=N_COMP_CONTROL) -> pd.DataFrame:
    """通過セットと血球組成マーカーの順位相関を、対照と比べて測る。

    ラベルが経路やモジュールでも、遺伝子内容が血球系なら組成マーカーと相関する。
    絶対値の最大が大きければ「そのセットは実質的に組成を測っている」と言える。

    **ただし絶対値だけでは足りない。**転写全体が血球組成の軸に載っているなら、
    どんな遺伝子を集めても組成マーカーと相関する。そこでサイズと平均発現量の十分位を
    そろえたランダムセットにも同じ「マーカーとの最大絶対相関」を計算し、
    観測値がその分布のどこに来るかを経験 p 値で示す。ここで超過が出なければ、
    「通過セットは組成を測っている」とは言えず、「この行列では何を集めても組成に
    載る」と言うべきことになる。

    **さらに、対照と比べても循環が残る。**通過 17 件のうち 14 件は細胞種マーカーで、
    代理変数に使うマーカー集合と遺伝子を共有する。共有していれば、相関の一部は
    「同じ遺伝子を両側に入れた」ことの帰結であって、組成との関連の証拠にならない。
    そこで 2 通りで測る。

      raw    : そのまま（重複を許す）
      disjoint : 評価セットからマーカー遺伝子の和集合を、各マーカー集合から評価セットの
                 遺伝子を、**双方から**除いて測り直す。対照も同じプール（マーカー遺伝子を
                 除いた集合）から引き直す

    disjoint で超過が残れば、相関は共有遺伝子だけでは説明されない。
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

    # 対照プール: 発現量十分位ごとに、この行列で使える遺伝子を集める
    gm = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gm = gm.loc[gm.index.intersection(Z.index)]
    dec = pd.qcut(gm, 10, labels=False, duplicates="drop")
    pool: dict[int, list[str]] = defaultdict(list)
    for g, d in dec.items():
        pool[int(d)].append(g)
    gen = rng(23)

    def max_abs_corr(gl) -> float | None:
        s = score(gl)
        if s is None:
            return None
        return max(abs(float(stats.spearmanr(s, v)[0])) for v in refs.values())

    # マーカー集合の遺伝子（和集合）。disjoint 版で評価セットから除く対象。
    marker_genes: set[str] = set()
    marker_members: dict[str, list[str]] = {}
    for name, (family, gl) in all_sets.items():
        if family != "celltype":
            continue
        label = name.split("|", 1)[1]
        if label in REFERENCE_MARKERS:
            marker_members[label] = [g for g in gl if g in Z.index]
            marker_genes |= set(marker_members[label])
    pool_disj: dict[int, list[str]] = {
        d: [g for g in gs if g not in marker_genes] for d, gs in pool.items()
    }
    gen_disj = rng(29)

    def max_abs_corr_disjoint(gl) -> tuple[float, str] | None:
        """評価セットとマーカー集合から共有遺伝子を双方除いて、最大絶対相関を返す。"""
        s_genes = [g for g in gl if g not in marker_genes]
        s = score(s_genes)
        if s is None:
            return None
        best_v, best_k = None, ""
        for label, m_genes in marker_members.items():
            m = score([g for g in m_genes if g not in set(gl)])
            if m is None:
                continue
            v = abs(float(stats.spearmanr(s, m)[0]))
            if best_v is None or v > best_v:
                best_v, best_k = v, label
        return (best_v, best_k) if best_v is not None else None

    rows = []
    for _, r in passing.iterrows():
        present = [g for g in genes.get(r["set"], []) if g in Z.index and g in dec.index]
        if len(present) < 2:
            continue
        s = score(present)
        cors = {k: float(stats.spearmanr(s, v)[0]) for k, v in refs.items()}
        best = max(cors, key=lambda k: abs(cors[k]))
        obs = abs(cors[best])
        # --- disjoint 版 ---
        n_shared = len(set(present) & marker_genes)
        dj = max_abs_corr_disjoint(present)
        if dj is None:
            dj_obs, dj_ref, dj_null_med, dj_p = float("nan"), "", float("nan"), float("nan")
        else:
            dj_obs, dj_ref = dj
            decs_d = [int(dec[g]) for g in present if g not in marker_genes]
            dj_null = []
            for _ in range(n_control):
                draw = [pool_disj[d][int(gen_disj.integers(len(pool_disj[d])))]
                        for d in decs_d if pool_disj[d]]
                v = max_abs_corr_disjoint(draw)
                if v is not None:
                    dj_null.append(v[0])
            dj_null_med = float(np.median(dj_null)) if dj_null else float("nan")
            dj_p = ((sum(1 for v in dj_null if v >= dj_obs) + 1) / (len(dj_null) + 1)
                    if dj_null else float("nan"))
        # 同じサイズ・同じ発現量十分位の構成でランダムセットを引き、同じ量を測る
        decs = [int(dec[g]) for g in present]
        null = []
        for _ in range(n_control):
            draw = [pool[d][int(gen.integers(len(pool[d])))] for d in decs]
            v = max_abs_corr(draw)
            if v is not None:
                null.append(v)
        null_med = float(np.median(null)) if null else float("nan")
        p_emp = (sum(1 for v in null if v >= obs) + 1) / (len(null) + 1) if null else float("nan")
        rows.append({"set": r["set"], "family": r["family"], "icc": r["icc"],
                     "最大相関の相手": best, "最大相関": round(cors[best], 3),
                     "最大絶対相関": round(obs, 3),
                     "対照の最大絶対相関 中央値": round(null_med, 3),
                     "超過": round(obs - null_med, 3),
                     "経験p": round(p_emp, 4),
                     "マーカーと共有する遺伝子数": n_shared,
                     "重複除去後の最大絶対相関": round(dj_obs, 3),
                     "重複除去後の相手": dj_ref,
                     "重複除去後の対照 中央値": round(dj_null_med, 3),
                     "重複除去後の超過": round(dj_obs - dj_null_med, 3),
                     "重複除去後の経験p": round(dj_p, 4),
                     **{f"rho_{k}": round(v, 3) for k, v in cors.items()}})
    out = pd.DataFrame(rows)
    if not out.empty:
        for src, dst in (("経験p", "q"), ("重複除去後の経験p", "重複除去後のq")):
            ok = out[src].notna()
            out[dst] = np.nan
            if ok.sum() > 1:
                out.loc[ok, dst] = multipletests(out.loc[ok, src], method="fdr_bh")[1]
    return out


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
