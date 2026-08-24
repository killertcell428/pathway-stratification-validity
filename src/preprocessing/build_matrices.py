"""E-MTAB-2232 の処理済み行列を、条件別の gene x individual 行列に変換する。

やること
  1. ADF (A-MEXP-2210) で ILMN プローブ -> HUGO 遺伝子記号を引く
  2. 1 遺伝子に複数プローブある場合、安静時条件で平均発現が最大のプローブを代表にし、
     その選択を全条件で共通に使う
  3. 安静時条件の平均発現の分位で発現フィルタをかける
     （未発現遺伝子を残すと「共発現しない」という主張が検出限界の産物になる）
  4. 条件別の parquet と、遺伝子平均発現テーブル、個人の探索/検証分割を書き出す

出力
  data/interim/expr_<condition>.parquet   genes x individuals
  data/interim/gene_expression_naive.csv  遺伝子の平均発現（ランダム対照のマッチング用）
  data/metadata/samples.json              条件別の個人数と重なり
  data/metadata/donor_split.json          探索/検証の個人分割
"""

from __future__ import annotations

import os

import json
import sys
import zipfile

import numpy as np
import pandas as pd

from ..common import expr_path, gene_mean_path, INTERIM, METADATA, RAW, load_config, rng

# 処理済み zip 内のファイル名の接頭辞 -> 条件キー
PREFIX_TO_CONDITION = {"CD14": "naive", "LPS2": "lps2", "LPS24": "lps24", "IFN": "ifn24"}


def load_probe_to_symbol() -> pd.Series:
    """ADF から Experimental プローブの HUGO 記号を引く。"""
    adf = RAW / "A-MEXP-2210.adf.txt"
    if not adf.exists():
        raise FileNotFoundError(f"{adf} がない。pixi run download を先に実行する")

    with adf.open(encoding="utf-8", errors="replace") as f:
        skip = 0
        for line in f:
            if line.startswith("Reporter Name\t"):
                break
            skip += 1

    df = pd.read_csv(adf, sep="\t", skiprows=skip, low_memory=False)
    keep = df["Reporter Group[role]"].eq("Experimental")
    sym = (
        df.loc[keep, ["Reporter Name", "Reporter Database Entry[hugo]"]]
        .dropna()
        .rename(columns={"Reporter Name": "probe", "Reporter Database Entry[hugo]": "symbol"})
    )
    sym["symbol"] = sym["symbol"].astype(str).str.strip().str.upper()
    sym = sym[sym["symbol"].ne("") & ~sym["symbol"].str.startswith("LOC")]
    return sym.drop_duplicates("probe").set_index("probe")["symbol"]


def read_condition_matrices() -> dict[str, pd.DataFrame]:
    """processed zip から条件別の probe x individual 行列を読む。"""
    out: dict[str, pd.DataFrame] = {}
    for zip_path in sorted(RAW.glob("E-MTAB-2232.processed.*.zip")):
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                prefix = name.split(".")[0]
                cond = PREFIX_TO_CONDITION.get(prefix)
                if cond is None:
                    print(f"  [warn] 条件を判定できないファイルを飛ばす: {name}")
                    continue
                with z.open(name) as f:
                    header = f.readline().decode("utf-8").rstrip("\n").split("\t")
                # dtype はプローブ ID 列を除いて指定する（一括指定すると ID を float に
                # 変換しようとして落ちる）
                dtypes = {c: np.float32 for c in header[1:]}
                dtypes[header[0]] = str
                with z.open(name) as f:
                    df = pd.read_csv(f, sep="\t", index_col=0, dtype=dtypes)
                df.columns = [str(c).strip() for c in df.columns]
                out[cond] = df
                print(f"  {cond:6s} <- {name}  {df.shape[0]} probes x {df.shape[1]} individuals")
    return out


def collapse_to_genes(
    mats: dict[str, pd.DataFrame], probe2sym: pd.Series, resting: str
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """代表プローブを安静時条件で決め、全条件に同じ選択を適用する。"""
    rest = mats[resting]
    common = rest.index.intersection(probe2sym.index)
    sym = probe2sym.loc[common]
    mean_rest = rest.loc[common].mean(axis=1)

    order = pd.DataFrame({"symbol": sym, "mean": mean_rest}).sort_values(
        ["symbol", "mean"], ascending=[True, False]
    )
    representative = order.groupby("symbol", sort=True).head(1)
    probe_of_gene = pd.Series(representative.index.values, index=representative["symbol"].values)
    print(f"  代表プローブ: {len(probe_of_gene)} 遺伝子（元 {len(common)} プローブ）")

    gene_mats: dict[str, pd.DataFrame] = {}
    for cond, mat in mats.items():
        probes = [p for p in probe_of_gene.values if p in mat.index]
        sub = mat.loc[probes].copy()
        sub.index = probe_of_gene.index[[i for i, p in enumerate(probe_of_gene.values) if p in mat.index]]
        sub.index.name = "gene"
        gene_mats[cond] = sub

    gene_mean_rest = gene_mats[resting].mean(axis=1).rename("mean_expression")
    return gene_mats, gene_mean_rest


def apply_expression_filter(
    gene_mats: dict[str, pd.DataFrame], gene_mean_rest: pd.Series, cfg: dict
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    spec = cfg["preprocessing"]["expression_filter"]
    if spec["method"] != "percentile":
        raise ValueError(f"未実装の発現フィルタ: {spec['method']}")
    # 感度分析用に環境変数で分位を上書きできる。既定は設定ファイルの値。
    # T26_DATASET / T26_MATRIX_SUFFIX と同じ名前空間の考え方に揃えてある。
    value = float(os.environ.get("T26_EXPR_PERCENTILE", spec["value"]))
    cutoff = float(np.percentile(gene_mean_rest.values, value))
    expressed = gene_mean_rest.index[gene_mean_rest.values > cutoff]
    print(
        f"  発現フィルタ: 安静時平均 > {cutoff:.3f} (第{value:g}分位) で "
        f"{len(expressed)}/{len(gene_mean_rest)} 遺伝子を残す"
    )
    return ({c: m.loc[m.index.intersection(expressed)] for c, m in gene_mats.items()},
            gene_mean_rest.loc[expressed])


def main() -> int:
    cfg = load_config("analysis")
    resting = cfg["conditions"]["resting"]

    print("[1/4] ADF を読む")
    probe2sym = load_probe_to_symbol()
    print(f"  {len(probe2sym)} プローブに記号が付いた")

    print("[2/4] 条件別行列を読む")
    mats = read_condition_matrices()
    if resting not in mats:
        raise RuntimeError(f"安静時条件 {resting} の行列がない")

    print("[3/4] 遺伝子レベルに畳んで発現フィルタをかける")
    gene_mats, gene_mean = collapse_to_genes(mats, probe2sym, resting)
    gene_mats, gene_mean = apply_expression_filter(gene_mats, gene_mean, cfg)

    print("[4/4] 書き出す")
    meta = {}
    for cond, mat in gene_mats.items():
        # expr_path は T26_MATRIX_SUFFIX を見る。直書きすると、感度分析の走行が
        # 正本の行列を上書きする（RNA-seq 側の build_rnaseq_matrix は接尾辞対応済みで、
        # ここだけ非対称になっていた）。
        path = expr_path(cond)
        mat.astype(np.float32).to_parquet(path)
        meta[cond] = {"n_genes": int(mat.shape[0]), "n_individuals": int(mat.shape[1])}
        print(f"  {path.name}: {mat.shape[0]} genes x {mat.shape[1]} individuals")

    gene_mean.to_csv(gene_mean_path())

    ids = {c: set(m.columns) for c, m in gene_mats.items()}
    meta["overlap_with_resting"] = {
        c: len(ids[c] & ids[resting]) for c in gene_mats if c != resting
    }
    (METADATA / "samples.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    donors = sorted(ids[resting], key=lambda x: int(x) if str(x).isdigit() else 10**9)
    if len(donors) < cfg["preprocessing"]["min_individuals"]:
        raise RuntimeError(f"個人数が少なすぎる: {len(donors)}")
    perm = rng(1).permutation(len(donors))
    n_disc = int(round(len(donors) * cfg["donor_split"]["discovery_fraction"]))
    split = {
        "discovery": sorted(donors[i] for i in perm[:n_disc]),
        "validation": sorted(donors[i] for i in perm[n_disc:]),
    }
    (METADATA / "donor_split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  個人分割: 探索 {len(split['discovery'])} / 検証 {len(split['validation'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
