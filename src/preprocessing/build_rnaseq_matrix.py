"""GSE81046（RSEM gene-level RNA-seq）を条件別の gene x individual 行列にする。

T26_DATASET=gse81046 で実行する前提。出力先は名前空間つきの
data/interim/gse81046/ と data/metadata/gse81046/ になり、以降の
derive_modules / run_evaluation / attribution / retest は同じコードのまま動く。

処理の設計（主コホートと同じ規則を RNA-seq に翻訳する）
  1. RAW.tar から *genes.results.txt.gz だけを展開せずストリームで読む
  2. TPM 列を取り、log2(TPM + 1) に変換する
  3. Ensembl ID → HUGO symbol は GTEx GCT のヘッダ 2 列（Name / Description）から
     作った対応表を使う（バージョン番号は落として突合する）
  4. 同一 symbol に複数 Ensembl ID がある場合は、安静時（NI）平均発現が最大の ID を
     代表にする（主コホートの「代表プローブ」と同じ規則）
  5. 発現フィルタ: NI 平均の中位数超え（同じ規則）
  6. 個人分割: 探索/検証 50:50（同じ seed）

ファイル名の規約: GSMxxxxxxx_{個人}_{条件}{時間}_mRNA.genes.results.txt.gz
  例 GSM2141236_AF11_L2_mRNA.genes.results.txt.gz → 個人 AF11, 条件 L
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile

import numpy as np
import pandas as pd

from ..common import (INTERIM, MATRIX_SUFFIX, METADATA, RAW, expr_path,
                      gene_mean_path, load_config, rng)

TAR = "GSE81046_RAW.tar"
GTEX_GCT = "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
NAME_RE = re.compile(r"GSM\d+_(?P<ind>[A-Z]+\d+)_(?P<cond>NI|L|S)\d*_mRNA\.genes\.results\.txt\.gz$")


def gene_map_from_gtex() -> pd.Series:
    """GTEx GCT の先頭 2 列（Name=ENSG.version, Description=symbol）から対応表を作る。

    行列本体は読まず、各行の先頭 2 フィールドだけを取る（1.6GB を 1 パス）。
    """
    path = RAW / GTEX_GCT
    if not path.exists():
        raise FileNotFoundError(f"{path} がない。pixi run download を先に実行する")
    pairs: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        f.readline()          # #1.2
        f.readline()          # 次元
        f.readline()          # ヘッダ
        for line in f:
            name, _, rest = line.partition("\t")
            desc, _, _ = rest.partition("\t")
            ensg = name.split(".")[0]
            sym = desc.strip().upper()
            if ensg and sym:
                pairs[ensg] = sym
    s = pd.Series(pairs)
    print(f"  Ensembl→symbol 対応: {len(s):,} 件（GTEx GCT 由来）")
    return s


def tmm_factors(counts: np.ndarray, logratio_trim: float = 0.3,
                sum_trim: float = 0.05) -> np.ndarray:
    """TMM 正規化係数（Robinson & Oshlack 2010）。edgeR calcNormFactors と同じ手順。

    counts は 遺伝子 x サンプル。参照サンプルは上四分位が平均に最も近いものを選ぶ。
    各サンプルについて参照との log 比 M を、M で上下 30%・A で上下 5% トリムした
    のち逆分散重みで平均し、2^M を係数とする。最後に幾何平均 1 に正規化する。
    RNA-seq の共発現解析で TPM をそのまま使うと、発現が何遺伝子に集中しているかの
    個人差が全遺伝子に共通軸として乗るため、この工程を主経路に置く。
    """
    lib = counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpm = counts / lib * 1e6
    uq = np.array([np.percentile(cpm[cpm[:, j] > 0, j], 75) if (cpm[:, j] > 0).any() else 0.0
                   for j in range(counts.shape[1])])
    ref = int(np.argmin(np.abs(uq - uq.mean())))

    f = np.ones(counts.shape[1])
    r_cnt, r_lib = counts[:, ref], lib[ref]
    for j in range(counts.shape[1]):
        o_cnt, o_lib = counts[:, j], lib[j]
        ok = (o_cnt > 0) & (r_cnt > 0)
        if ok.sum() < 10:
            continue
        o, r = o_cnt[ok], r_cnt[ok]
        m = np.log2((o / o_lib) / (r / r_lib))
        a = 0.5 * np.log2((o / o_lib) * (r / r_lib))
        w = (o_lib - o) / (o_lib * o) + (r_lib - r) / (r_lib * r)   # 近似分散の逆数の逆
        keep = np.isfinite(m) & np.isfinite(a) & (w > 0)
        m, a, w = m[keep], a[keep], w[keep]
        if m.size < 10:
            continue
        lo_m, hi_m = np.quantile(m, [logratio_trim, 1 - logratio_trim])
        lo_a, hi_a = np.quantile(a, [sum_trim, 1 - sum_trim])
        sel = (m >= lo_m) & (m <= hi_m) & (a >= lo_a) & (a <= hi_a)
        if sel.sum() < 10:
            continue
        f[j] = 2 ** (np.sum(m[sel] / w[sel]) / np.sum(1.0 / w[sel]))
    return f / np.exp(np.mean(np.log(f)))


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """サンプル（列）間で分布を揃える。

    各列を順位に置き換え、全列の順位ごとの平均値を共通の参照分布として割り当てる。
    同順位は平均で処理する。TPM が揃えるのはライブラリサイズだけで分布の形は揃わない
    ため、共発現を測る前にこれを入れないと「発現が何遺伝子に集中しているか」の
    個人差が全遺伝子に共通の軸として乗る。
    """
    x = df.to_numpy(dtype=np.float64)
    order = np.argsort(x, axis=0, kind="stable")
    ranked = np.take_along_axis(x, order, axis=0)
    reference = ranked.mean(axis=1)                      # 順位ごとの平均 = 参照分布
    out = np.empty_like(x)
    rows = np.arange(x.shape[0])
    for j in range(x.shape[1]):
        # 同値は平均順位に割り当てる（rankdata "average" 相当）
        col = x[:, j]
        r = np.empty(len(col))
        r[order[:, j]] = rows
        # 同値グループを平均化
        s = np.sort(col)
        uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        starts = csum - counts
        grp_mean = np.array([reference[a:b].mean() for a, b in zip(starts, csum)])
        out[:, j] = grp_mean[np.searchsorted(uniq, col)]
    return pd.DataFrame(out.astype(np.float32), index=df.index, columns=df.columns)


def read_tar_tpm() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """RAW.tar から TPM 行列（ENSG x サンプル）とサンプル表を作る。"""
    path = RAW / TAR
    if not path.exists():
        raise FileNotFoundError(f"{path} がない。pixi run download を先に実行する")

    cols: dict[str, pd.Series] = {}
    counts: dict[str, pd.Series] = {}
    meta = []
    with tarfile.open(path) as tar:
        for member in tar:
            m = NAME_RE.search(member.name)
            if not m:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            with gzip.open(io.BytesIO(fobj.read()), "rt", encoding="utf-8") as f:
                df = pd.read_csv(f, sep="\t", usecols=["gene_id", "TPM", "expected_count"])
            key = f"{m['ind']}|{m['cond']}"
            idx = df["gene_id"].str.split(".").str[0].values
            cols[key] = pd.Series(df["TPM"].to_numpy(dtype=np.float32), index=idx)
            counts[key] = pd.Series(df["expected_count"].to_numpy(dtype=np.float32), index=idx)
            meta.append({"sample": key, "individual": m["ind"], "condition": m["cond"]})
            if len(meta) % 100 == 0:
                print(f"  {len(meta)} サンプル読了 ...")
    expr = pd.DataFrame(cols)
    cnt = pd.DataFrame(counts)
    info = pd.DataFrame(meta)
    print(f"  行列: {expr.shape[0]:,} genes x {expr.shape[1]} samples")
    print("  条件内訳:", info.condition.value_counts().to_dict())
    print(f"  個人数: {info.individual.nunique()}")
    return expr, cnt, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-value", type=float, default=None,
                    help="発現フィルタの閾値（CPM）を config より優先して使う。感度分析用")
    ap.add_argument("--reuse-full", action="store_true",
                    help="フィルタ前の expr_full.parquet を再利用し、tar の読み直しを省く")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = load_config("analysis")
    resting = cfg["conditions"]["resting"]
    all_conds = [resting] + list(cfg["conditions"]["perturbed"])
    quant = cfg["preprocessing"].get("quantification", "tpm")
    full_path = INTERIM / "expr_full.parquet"

    # 閾値だけを変えて回し直すとき、tar の再読み込み（数分）と TMM の再計算は
    # 結果が同一なので省く。フィルタ前の行列は前回の走行で保存してある。
    if args.reuse_full and full_path.exists():
        print(f"[1/5] フィルタ前の行列を再利用: {full_path.name}")
        expr = pd.read_parquet(full_path)
        ni_cols = [c for c in expr.columns if c.endswith(f"|{resting}")]
        info = pd.read_csv(METADATA / "samples.csv")
        print(f"  {expr.shape[0]:,} genes x {expr.shape[1]} samples"
              f"（安静時 {len(ni_cols)} 列）")
        return _finish(cfg, args, expr, ni_cols, info, quant, all_conds, resting)

    print("[1/5] RAW.tar から TPM と expected_count を読む")
    expr, cnt, info = read_tar_tpm()

    print("[2/5] 定量値を作って symbol に対応づける")
    if quant == "tmm_logcpm":
        # 分野標準（edgeR）。expected_count に TMM 係数を掛けて log2(CPM+1) にする。
        c = cnt.loc[cnt.index.intersection(expr.index)].to_numpy(dtype=np.float64)
        f = tmm_factors(c)
        eff = c.sum(axis=0) * f
        expr = pd.DataFrame(
            np.log2(c / eff * 1e6 + 1.0).astype(np.float32),
            index=cnt.index.intersection(expr.index), columns=cnt.columns,
        )
        print(f"  TMM 正規化係数: 中央値 {np.median(f):.3f}、範囲 {f.min():.3f}〜{f.max():.3f}")
    elif quant == "tpm":
        expr = np.log2(expr + 1.0)
    else:
        raise ValueError(f"未実装の定量: {quant}")
    gmap = gene_map_from_gtex()
    expr = expr.loc[expr.index.intersection(gmap.index)]
    symbols = gmap.loc[expr.index]

    print("[3/5] 代表 Ensembl ID を安静時平均で選び、symbol に畳む")
    ni_cols = [c for c in expr.columns if c.endswith(f"|{resting}")]
    mean_ni = expr[ni_cols].mean(axis=1)
    order = pd.DataFrame({"symbol": symbols.values, "mean": mean_ni.values}, index=expr.index)
    rep = order.sort_values(["symbol", "mean"], ascending=[True, False]).groupby("symbol").head(1)
    expr = expr.loc[rep.index]
    expr.index = pd.Index(rep["symbol"].values, name="gene")

    # フィルタ前の全遺伝子行列を残す（閾値の感度分析で tar を読み直さないため）
    expr.astype(np.float32).to_parquet(full_path)
    print(f"  フィルタ前を保存: {full_path.name}（{expr.shape[0]:,} genes）")

    return _finish(cfg, args, expr, ni_cols, info, quant, all_conds, resting)


def _finish(cfg, args, expr, ni_cols, info, quant, all_conds, resting) -> int:
    """発現フィルタから下流（条件別の書き出しと個人分割）。

    tar の読み直しを省いた再走行でもここから同じ処理を通せるように切り出した。
    """
    spec = dict(cfg["preprocessing"]["expression_filter"])
    if args.min_value is not None:
        spec["min_value"] = args.min_value
        print(f"  発現フィルタの閾値を上書き: min_value = {args.min_value}")
    mean_ni = expr[ni_cols].mean(axis=1)
    if spec["method"] == "detected":
        # RNA-seq 用。単位は quantification に追随する（tmm_logcpm なら CPM、tpm なら TPM）。
        unit = "CPM" if quant == "tmm_logcpm" else "TPM"
        lin_ni = np.power(2.0, expr[ni_cols].to_numpy(dtype=np.float64)) - 1.0
        frac = (lin_ni >= float(spec["min_value"])).mean(axis=1)
        expressed = expr.index[frac >= float(spec["min_fraction"])]
        print(f"  発現フィルタ: {unit} >= {spec['min_value']} を {spec['min_fraction']:.0%} 以上の "
              f"個人で満たす遺伝子 {len(expressed):,}/{len(expr):,}")
    elif spec["method"] == "percentile":
        cutoff = float(np.percentile(mean_ni.values, spec["value"]))
        expressed = mean_ni.index[mean_ni.values > cutoff]
        print(f"  発現フィルタ: NI 平均 > {cutoff:.3f} (第{spec['value']}分位) で "
              f"{len(expressed):,}/{len(mean_ni):,} 遺伝子")
    else:
        raise ValueError(f"未実装の発現フィルタ: {spec['method']}")
    expr = expr.loc[expressed]

    norm = cfg["preprocessing"].get("cross_sample_normalization")
    # YAML 側で「なし」を none / null / 空文字のどれで書いても通す。TMM を主経路に
    # した時点で既定は「なし」であり、ここで転ぶと前処理を最初から回せない。
    if norm in (None, False, "", "none", "null", "None"):
        print("  サンプル間正規化なし（TMM 係数のみ）")
    elif norm == "quantile":
        expr = quantile_normalize(expr)
        print("  サンプル間分位正規化を適用（全サンプルを共通の分布に揃える）")
    elif norm:
        raise ValueError(f"未実装のサンプル間正規化: {norm}")

    print("[4/5] 条件別に書き出す（列は個人 ID）")
    for cond in all_conds:
        cc = [c for c in expr.columns if c.endswith(f"|{cond}")]
        sub = expr[cc].copy()
        sub.columns = [c.split("|")[0] for c in cc]
        sub = sub.loc[:, ~sub.columns.duplicated()]
        out = expr_path(cond)
        sub.astype(np.float32).to_parquet(out)
        print(f"  {out.name}: {sub.shape[0]:,} genes x {sub.shape[1]} individuals")

    mean_ni.loc[expressed].rename("mean_expression").to_csv(gene_mean_path())

    print("[5/5] 個人を探索/検証に分割する（主コホートと同じ seed）")
    ni = pd.read_parquet(expr_path(resting))
    donors = sorted(ni.columns)
    if len(donors) < cfg["preprocessing"]["min_individuals"]:
        raise RuntimeError(f"個人数が少なすぎる: {len(donors)}")
    perm = rng(1).permutation(len(donors))
    n_disc = int(round(len(donors) * cfg["donor_split"]["discovery_fraction"]))
    split = {
        "discovery": sorted(donors[i] for i in perm[:n_disc]),
        "validation": sorted(donors[i] for i in perm[n_disc:]),
    }
    # 感度分析の走行では正本の分割を書き換えない。遺伝子フィルタは個人を落とさない
    # ので中身は同一になるが、正本を上書きしうる経路は残さない。
    if MATRIX_SUFFIX:
        print(f"  接尾辞 {MATRIX_SUFFIX} の走行なので donor_split.json は書き換えない")
    else:
        (METADATA / "donor_split.json").write_text(
            json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        info.to_csv(METADATA / "samples.csv", index=False, encoding="utf-8")
    print(f"  探索 {len(split['discovery'])} / 検証 {len(split['validation'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
