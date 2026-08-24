"""「縦断的に安定で個人間で変動する」選抜が対照を超えるかを、実データで確かめる。

なぜ要るか
  本稿の緒言は、Scheid ら [12] の遺伝子選抜について次のように書いている。

    縦断的に安定で個人間で変動する遺伝子群が、その遺伝子に固有の情報を
    担っているのか、それとも任意の遺伝子群に残る軸を反映しているだけなのかは、
    その設計では判定できない

  「判定できない」で止めると推測のままになる。彼ら自身が公表した選抜群
  （補足 Table S2 の 44 群）を本稿の対照設計に載せれば、判定できる。

何を測るか
  GSE47353 の接種前 2 時点（同一個人・同一条件、7 日間隔）で、
  44 群それぞれについて個人間の共変動と反復測定信頼性（ICC）を計算し、
  サイズと発現量をそろえたランダム対照と比べる。手続きは retest_check と同一。

  本流のパイプラインには足さない。ファミリーを 7 つに増やすと本文の
  「6 ファミリー」やファミリー別の表がすべて動くため、独立の検査として走らせる。

結果の読み方（両方向あり得る）
  対照を超えない  → 対照なしの選抜が何を拾うかの実例になる。緒言の主張が実測になる
  対照を超える    → 緒言の主張を弱める必要がある。その場合はそう書く

注意
  彼らの群は CD4+ T 細胞で選抜され、ここで採点するのは末梢血単核球である。
  コホートをまたいだ適用であり、公表シグネチャを別コホートで採点するのと
  同じ性質の制約がある。この制約は結果の解釈に明記する。

出力: results/tables/stability_selected.csv
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from ..common import INTERIM, RAW, TABLES, load_config, rng
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import _rank_rows, _standardize_rows
from .batch_check import residualize
from .metrics import draw_index_matrix, null_batch  # noqa: F401
from .metrics import (empirical_null, icc_two_way, matched_random_sets,
                      mean_pairwise_rho, pooled_set_score)
from .run_evaluation import load_all_sets

SUPPLEMENT = RAW / "NIHMS930345-supplement-2.xlsx"
PROBEMAP = RAW / "gencode.v36.annotation.gtf.gene.probemap"
SHEETS = ["1st_Cohort_Baseline", "1st_Cohort_Stimulated",
          "2nd_Cohort_Baseline", "2nd_Cohort_Stimulated"]
RETEST = ("day-7", "day0")      # どちらも接種前。retest_check と同じ対

# 除去する雑音軸。GSE47353 は公開メタデータにチップも処理日もないため、
# 技術要因は抜けない（4.3 節に記載の限界）。代わりに細胞組成を抜く。
# 3.9 節で ICC が対照を超えた 15 セットの上位が好中球・血小板・単球だったので、
# ここで検証すべき軸はまさにこれである。
COMPOSITION_SETS = ["Neutrophils", "T Cells Naive", "B Cells",
                    "NK Cells", "Platelets", "Monocytes"]


def composition_design(x: np.ndarray, genes, index: dict[str, int]) -> np.ndarray:
    """細胞種マーカーの z 平均を並べた設計行列を作る（組成の代理）。

    deconvolution を使わない理由は 2.6 節と同じ。参照プロファイルの選択が
    結果を左右し、このコホートに適切な参照がない。
    """
    gmt = RAW / "gene_sets" / "celltype__PanglaoDB_Augmented_2021.gmt"
    sets = parse_gmt(gmt.read_text(encoding="utf-8")) if gmt.exists() else {}
    z = _standardize_rows(x)
    cols = []
    used = []
    for name in COMPOSITION_SETS:
        idx = [index[g] for g in sets.get(name, []) if g in index]
        if len(idx) >= 10:
            cols.append(np.nanmean(z[idx], axis=0))
            used.append(name)
    print(f"  組成の代理に使えた細胞種マーカー: {used}")
    return np.column_stack(cols) if cols else np.zeros((x.shape[1], 0))


def load_groups() -> dict[str, list[str]]:
    """補足 Table S2 の選抜群を読み、Ensembl ID を記号に変換する。

    本稿の行列は HUGO 記号で索引されているため変換が必要。
    gencode v36 の対応表（TCGA 用に取得済み）を版番号を落として使う。
    """
    pm = pd.read_csv(PROBEMAP, sep="\t")
    e2s = dict(zip(pm["id"].str.split(".").str[0], pm["gene"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = pd.read_excel(SUPPLEMENT, sheet_name=SHEETS)

    out: dict[str, list[str]] = {}
    for sheet, df in book.items():
        for col in df.columns:
            ids = [str(v) for v in df[col].dropna() if str(v).startswith("ENSG")]
            if not ids:
                continue
            syms = sorted({e2s[i] for i in ids if i in e2s})
            out[f"{sheet}|{col}"] = syms
    return out


def main() -> int:
    if not SUPPLEMENT.exists():
        print(f"補足ファイルがない: {SUPPLEMENT}")
        return 1

    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]
    gen = rng(53)

    groups = load_groups()
    print(f"[1/3] 選抜群を読む: {len(groups)} 群")

    a = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[0]}.parquet")
    b = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[1]}.parquet")
    genes = a.index.intersection(b.index)
    donors = [d for d in a.columns if d in set(b.columns)]
    a, b = a.loc[genes, donors], b.loc[genes, donors]
    print(f"  反復測定の対: {len(donors)} 名 / 遺伝子 {len(genes):,}"
          f"（{RETEST[0]} と {RETEST[1]}、どちらも接種前）")

    both = np.concatenate([a.to_numpy(dtype=np.float64),
                           b.to_numpy(dtype=np.float64)], axis=1)
    n = a.shape[1]
    index = {g: i for i, g in enumerate(genes)}

    # 組成の設計行列は 2 時点を連結した行列から作る（組成は検体ごとに変わる）
    design = composition_design(both, genes, index)
    both_res = residualize(both, design) if design.shape[1] else both

    def standardise(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = mat.mean(axis=1, keepdims=True)
        sd = mat.std(axis=1, ddof=0, keepdims=True)
        sd[sd == 0] = np.nan
        zz = (mat - mu) / sd
        return zz[:, :n], zz[:, n:], _standardize_rows(_rank_rows(mat[:, n:]))

    variants = {"除去前": standardise(both)}
    if design.shape[1]:
        variants["組成除去後"] = standardise(both_res)

    # 循環性の検証に使うマーカー遺伝子の和集合。
    # 群がマーカー遺伝子そのものを含んでいると、マーカースコアを回帰で抜く操作が
    # 「その遺伝子を抜く」ことに近づく。重複を除いても低下が残るかを確かめる。
    gmt = RAW / "gene_sets" / "celltype__PanglaoDB_Augmented_2021.gmt"
    marker_sets = parse_gmt(gmt.read_text(encoding="utf-8")) if gmt.exists() else {}
    marker_union: set[str] = set()
    for nm in COMPOSITION_SETS:
        marker_union |= set(marker_sets.get(nm, []))

    gene_mean = pd.read_csv(INTERIM / "valid_gene_expression.csv",
                            index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(genes)]
    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    by_dec: dict[int, list[str]] = defaultdict(list)
    for g, d in dec.items():
        by_dec[int(d)].append(g)
    pool_by_decile = {
        int(d): np.array([index[g] for g in gs if g in index], dtype=np.int64)
        for d, gs in by_dec.items()
    }
    print("[2/3] 群ごとに個人間の共変動と ICC を対照と比べる（除去前後）")
    rows, skipped = [], defaultdict(int)
    for variant, (z_a, z_b, S) in variants.items():
     # 行和によるまとめ計算は NaN があると使えない（順位も z も分散を持つので通常は出ない）
     assert not (np.isnan(S).any() or np.isnan(z_a).any() or np.isnan(z_b).any()), \
         f"{variant}: S / z に NaN。まとめ計算の前提が崩れる"
     for drop_marker in (False, True):
      tag = variant + ("・マーカー重複除外" if drop_marker else "")
      for name, syms in sorted(groups.items()):
          present = [g for g in syms if g in index]
          if drop_marker:
              present = [g for g in present if g not in marker_union]
          coverage = len(present) / len(syms) if syms else 0.0
          if len(present) < filt["min_genes"]:
              skipped["min_genes"] += 1
              continue
          if len(present) > filt["max_genes"]:
              skipped["max_genes"] += 1
              continue
          if coverage < filt["min_coverage"]:
              skipped["coverage"] += 1
              continue

          idx = np.array([index[g] for g in present])
          ic = mean_pairwise_rho(S[idx])
          icc = icc_two_way(np.column_stack([pooled_set_score(z_a[idx]),
                                             pooled_set_score(z_b[idx])]))

          # 対照 nc 個（既定 10,000）。1 件ずつ ICC を呼ぶと終わらないので
          # (対照, 個人) の作業配列にまとめて計算する。
          dec_pos = np.array([dec[g] for g in present if g in dec], dtype=np.int64)
          null_ic, null_icc = [], []
          filled = 0
          while filled < nc:
              take = min(2_000, nc - filled)
              m = draw_index_matrix(pool_by_decile, dec_pos, take, gen)
              x, yv, _ = null_batch(m, z_a, z_b, S)
              null_ic.append(x)
              null_icc.append(yv)
              filled += take
          n_ic = empirical_null(ic, np.concatenate(null_ic).tolist())
          n_icc = empirical_null(icc, np.concatenate(null_icc).tolist())

          rows.append({
              "variant": tag,
              "group": name, "n_genes": len(syms), "n_present": len(present),
              "coverage": round(coverage, 3),
              "ic": ic, "ic_null_mean": n_ic["null_mean"], "ic_z": n_ic["null_z"],
              "ic_p": n_ic["null_p"], "ic_p_empirical": n_ic["null_p_empirical"],
              "icc_p_empirical": n_icc["null_p_empirical"],
              "icc": icc, "icc_null_mean": n_icc["null_mean"],
              "icc_z": n_icc["null_z"], "icc_p": n_icc["null_p"],
          })
    df = pd.DataFrame(rows)
    if skipped:
        print("  除外:", dict(skipped))

    from statsmodels.stats.multitest import multipletests
    # FDR は条件（除去前 / 除去後）ごとに独立にかける。
    # 両条件をまとめて補正すると、同じ 18 群を 2 回見ているだけなのに検定数が倍になり、
    # 基準が不当に厳しくなる（実測で ICC の合格率が 88.9% から 61.1% に下がっていた）。
    # BH-FDR は経験 p にかける（主解析と同じ土台にそろえる）
    for col, out_col in (("ic_p_empirical", "ic_q"), ("icc_p_empirical", "icc_q")):
        df[out_col] = np.nan
        for variant, g in df.groupby("variant"):
            ok = g[col].notna()
            if ok.sum():
                df.loc[g.index[ok], out_col] = multipletests(
                    g.loc[ok, col], method="fdr_bh")[1]
    df.to_csv(TABLES / "stability_selected.csv", index=False, encoding="utf-8")

    print("[3/3] 判定")
    for variant, d in df.groupby("variant", sort=False):
        print(f"  --- {variant}（評価できた群 {len(d)} 件）---")
        print(f"    個人間の共変動 中央値 {d.ic.median():.3f}"
              f"（対照 {d.ic_null_mean.median():.3f}）"
              f" / 対照超え {int((d.ic_q < 0.05).sum())}/{len(d)}"
              f" ({(d.ic_q < 0.05).mean():.1%})")
        print(f"    ICC 中央値 {d.icc.median():.3f}"
              f"（対照 {d.icc_null_mean.median():.3f}）"
              f" / 対照超え {int((d.icc_q < 0.05).sum())}/{len(d)}"
              f" ({(d.icc_q < 0.05).mean():.1%})")
    if df.variant.nunique() > 1:
        piv = df.pivot_table(index="group", columns="variant", values=["ic", "icc"])
        for m, jp in (("ic", "個人間の共変動"), ("icc", "ICC")):
            before, after = piv[(m, "除去前")], piv[(m, "組成除去後")]
            ok = before.notna() & after.notna() & (before != 0)
            chg = ((after[ok] - before[ok]) / before[ok] * 100).median()
            print(f"  {jp}の変化（中央値）: {chg:+.0f}%")
    print(f"\n-> {TABLES / 'stability_selected.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
