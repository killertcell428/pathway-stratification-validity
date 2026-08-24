"""GTEx v8 全血を gene x individual 行列にする（WP1 の帰属再現用）。

T26_DATASET=gtex_blood で実行する前提。

GTEx を使う理由は共変量の豊富さにある。GSE35846 は技術と生物学の両方を持つが
189 名・マイクロアレイだった。GTEx は RNA-seq で約 755 検体、しかも
  技術   RIN 値(SMRIN) / 虚血時間(SMTSISCH) / バッチ(SMGEBTCH, SMNABTCH)
  生物学 年齢 / 性別 / 死因分類(DTHHRDY)
を公開している。虚血時間は他コホートにない軸で、「死後・採取後の時間経過」という
生物学とも技術ともつかない要因を評価できる。

処理は GSE81046 と同じ経路に揃える（TMM log-CPM、CPM >= 1 を過半数）。
1.6GB / 0.9GB の GCT は全部展開せず、全血の列だけをストリームで抜く。

出力
  data/interim/gtex_blood/expr_blood.parquet   genes x samples
  data/metadata/gtex_blood/covariates.csv      検体ごとの技術・生物学共変量
"""

from __future__ import annotations

import os

import gzip
import sys

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, RAW, load_config
from .build_rnaseq_matrix import tmm_factors

READS = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz"
SAMPLE_ATTR = "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
SUBJECT_PHENO = "GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"
# 解析する組織。既存の T26_DATASET と同じ発想で環境変数から切り替える。
# 組織ごとにコードを書き換えると「結果の違いはデータの違いにのみ由来する」
# という WP1 の前提が崩れるため、切り替えは入出力の名前空間と設定ファイルだけで行う。
TISSUE = os.environ.get("T26_GTEX_TISSUE", "Whole Blood")

# 使う共変量。SMRIN=RNA integrity、SMTSISCH=虚血時間、SMGEBTCH=発現バッチ、
# SMNABTCH=核酸抽出バッチ、SMCENTER=採取施設
TECH_COLS = ["SMRIN", "SMTSISCH", "SMGEBTCH", "SMNABTCH", "SMCENTER"]
BIO_COLS = ["AGE", "SEX", "DTHHRDY"]


def load_covariates() -> pd.DataFrame:
    attr = pd.read_csv(RAW / SAMPLE_ATTR, sep="\t", low_memory=False)
    attr = attr[attr["SMTSD"] == TISSUE].copy()
    attr["subject"] = attr["SAMPID"].str.split("-").str[:2].str.join("-")
    pheno = pd.read_csv(RAW / SUBJECT_PHENO, sep="\t")
    cov = attr.merge(pheno, left_on="subject", right_on="SUBJID", how="left")
    keep = ["SAMPID", "subject"] + [c for c in TECH_COLS + BIO_COLS if c in cov.columns]
    cov = cov[keep]
    print(f"  {TISSUE}: {len(cov)} 検体 / {cov.subject.nunique()} 個人")
    for c in TECH_COLS + BIO_COLS:
        if c in cov.columns:
            n = cov[c].notna().sum()
            uniq = cov[c].nunique()
            print(f"    {c:10s} 欠損なし {n:4d}  水準/範囲 {uniq}")
    return cov


def stream_counts(sample_ids: set[str]) -> pd.DataFrame:
    """GCT から対象検体の列だけを抜く。行名は Description（symbol）にする。"""
    path = RAW / READS
    if not path.exists():
        raise FileNotFoundError(f"{path} がない")
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()
        f.readline()
        header = f.readline().rstrip("\n").split("\t")
        idx = [i for i, h in enumerate(header) if h in sample_ids]
        cols = [header[i] for i in idx]
        print(f"  GCT の列 {len(header)-2:,} 本のうち {len(idx)} 本が対象")
        syms, rows = [], []
        for n, line in enumerate(f, 1):
            p = line.rstrip("\n").split("\t")
            syms.append(p[1].strip().upper())
            rows.append([p[i] for i in idx])
            if n % 20000 == 0:
                print(f"    {n:,} 行読了 ...")
    df = pd.DataFrame(np.array(rows, dtype=np.float32), index=syms, columns=cols)
    df.index.name = "gene"
    print(f"  読み込み: {df.shape[0]:,} genes x {df.shape[1]} samples")
    return df


def main() -> int:
    cfg = load_config("analysis")
    print("[1/4] 共変量を読む")
    cov = load_covariates()

    print("[2/4] カウント行列から全血の列を抜く")
    counts = stream_counts(set(cov["SAMPID"]))
    cov = cov[cov["SAMPID"].isin(counts.columns)].reset_index(drop=True)

    print("[3/4] TMM log-CPM に変換して発現フィルタをかける")
    # 同一 symbol に複数行がある場合は合計する（GTEx は Ensembl 単位の行を持つ）
    counts = counts.groupby(level=0).sum()
    c = counts.to_numpy(dtype=np.float64)
    f = tmm_factors(c)
    eff = c.sum(axis=0) * f
    expr = pd.DataFrame(np.log2(c / eff * 1e6 + 1.0).astype(np.float32),
                        index=counts.index, columns=counts.columns)
    print(f"  TMM 係数: 中央値 {np.median(f):.3f}、範囲 {f.min():.3f}〜{f.max():.3f}")

    spec = cfg["preprocessing"]["expression_filter"]
    lin = np.power(2.0, expr.to_numpy(dtype=np.float64)) - 1.0
    frac = (lin >= float(spec["min_value"])).mean(axis=1)
    expressed = expr.index[frac >= float(spec["min_fraction"])]
    print(f"  発現フィルタ: CPM >= {spec['min_value']} を {spec['min_fraction']:.0%} 以上で "
          f"{len(expressed):,}/{len(expr):,} 遺伝子")
    expr = expr.loc[expressed]

    print("[4/4] 書き出す")
    expr.to_parquet(INTERIM / "expr_blood.parquet")
    cov.to_csv(METADATA / "covariates.csv", index=False, encoding="utf-8")
    print(f"  expr_blood.parquet: {expr.shape[0]:,} genes x {expr.shape[1]} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
