"""検証コホート GSE47353 を、時点別の gene x individual 行列に変換する。

このコホートは 3 つの穴を埋める。
  (1) 反復測定信頼性: day-7 と day0 はどちらも接種前なので同一個人・同一条件の対になる
  (2) 別プラットフォーム・別細胞での外部再現性: Affymetrix Gene 1.0 ST / PBMC
  (3) 表現型予測力: ワクチン応答クラス（mn_adjmfc_class）

出力
  data/interim/valid_expr_<tp>.parquet     genes x individuals（tp = day-7, day0 ほか）
  data/metadata/valid_samples.csv          GSM・個人 ID・時点・応答クラス
  data/interim/valid_gene_expression.csv   day0 の平均発現（対照のマッチング用）
"""

from __future__ import annotations

import gzip
import re
import sys

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, RAW, load_config

SERIES = "GSE47353_series_matrix.txt.gz"
ANNOT = "GPL6244.annot.gz"


def parse_sample_metadata(path) -> pd.DataFrame:
    """series matrix のヘッダから GSM・個人 ID・時点・応答クラスを取る。"""
    gsms: list[str] = []
    fields: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_geo_accession"):
                gsms = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                key = vals[0].split(":")[0].strip() if vals else ""
                if key:
                    fields.setdefault(key, vals)
    df = pd.DataFrame({"gsm": gsms})
    for key, vals in fields.items():
        df[key] = [re.sub(rf"^{re.escape(key)}:\s*", "", v).strip() for v in vals]
    df["timepoint"] = df["sample collection time"].str.extract(r"\((day[^)]+)\)")
    df = df.rename(columns={"individual id": "individual", "mn_adjmfc_class": "response_class"})
    return df[["gsm", "individual", "timepoint", "response_class", "gender", "age"]]


def parse_expression(path, gsms: list[str]) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
        df = pd.read_csv(f, sep="\t", index_col=0, comment="!", low_memory=False)
    df.index = df.index.astype(str).str.strip('"')
    df.columns = [str(c).strip('"') for c in df.columns]
    keep = [c for c in df.columns if c in set(gsms)]
    return df[keep].astype(np.float32)


def parse_annotation(path) -> pd.Series:
    """GPL の annot から probe -> HUGO 記号を取る。複数記号の行は捨てる。"""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            if line.startswith("ID\t"):
                break
            skip += 1
    df = pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False,
                     compression="gzip", encoding_errors="replace")
    col = next((c for c in df.columns if c.lower().startswith("gene symbol")), None)
    if col is None:
        raise RuntimeError(f"遺伝子記号の列が見つからない: {list(df.columns)[:8]}")
    s = df[["ID", col]].dropna()
    s[col] = s[col].astype(str).str.strip().str.upper()
    s = s[~s[col].str.contains(r"///|^$")]          # 複数遺伝子に当たるプローブは除く
    return s.drop_duplicates("ID").set_index(df.columns[0])[col]


def main() -> int:
    cfg = load_config("analysis")
    series, annot = RAW / SERIES, RAW / ANNOT
    if not series.exists():
        raise FileNotFoundError(f"{series} がない。pixi run download を実行する")

    print("[1/4] サンプル情報を読む")
    meta = parse_sample_metadata(series)
    meta.to_csv(METADATA / "valid_samples.csv", index=False, encoding="utf-8")
    print(f"  {len(meta)} サンプル / {meta.individual.nunique()} 名")
    print("  時点:", dict(meta.timepoint.value_counts()))

    print("[2/4] 発現行列とアノテーションを読む")
    expr = parse_expression(series, meta.gsm.tolist())
    print(f"  {expr.shape[0]} probes x {expr.shape[1]} samples  値域 {expr.to_numpy().min():.2f}〜{expr.to_numpy().max():.2f}")
    probe2sym = parse_annotation(annot)
    print(f"  記号が付いたプローブ: {len(probe2sym)}")

    print("[3/4] 遺伝子レベルに畳んで発現フィルタをかける")
    common = expr.index.intersection(probe2sym.index.astype(str))
    sym = probe2sym.loc[common]
    day0 = meta.loc[meta.timepoint == "day0", "gsm"]
    day0 = [g for g in day0 if g in expr.columns]
    mean_day0 = expr.loc[common, day0].mean(axis=1)
    order = pd.DataFrame({"symbol": sym.values, "mean": mean_day0.values}, index=common)
    rep = order.sort_values(["symbol", "mean"], ascending=[True, False]).groupby("symbol").head(1)
    probe_of_gene = pd.Series(rep.index.values, index=rep["symbol"].values)
    print(f"  代表プローブ: {len(probe_of_gene)} 遺伝子")

    gene_expr = expr.loc[probe_of_gene.values]
    gene_expr.index = pd.Index(probe_of_gene.index, name="gene")
    spec = cfg["preprocessing"]["expression_filter"]
    gene_mean_day0 = gene_expr[day0].mean(axis=1)
    cutoff = float(np.percentile(gene_mean_day0.values, spec["value"]))
    expressed = gene_mean_day0.index[gene_mean_day0.values > cutoff]
    print(f"  発現フィルタ: day0 平均 > {cutoff:.3f} で {len(expressed)}/{len(gene_mean_day0)} 遺伝子")
    gene_expr = gene_expr.loc[expressed]
    gene_mean_day0.loc[expressed].rename("mean_expression").to_csv(
        INTERIM / "valid_gene_expression.csv"
    )

    print("[4/4] 時点別に書き出す（列は個人 ID）")
    for tp, grp in meta.dropna(subset=["timepoint"]).groupby("timepoint"):
        cols = [g for g in grp.gsm if g in gene_expr.columns]
        if not cols:
            continue
        sub = gene_expr[cols].copy()
        sub.columns = grp.set_index("gsm").loc[cols, "individual"].astype(str).values
        sub = sub.loc[:, ~sub.columns.duplicated()]
        path = INTERIM / f"valid_expr_{tp}.parquet"
        sub.astype(np.float32).to_parquet(path)
        print(f"  {path.name}: {sub.shape[0]} genes x {sub.shape[1]} individuals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
