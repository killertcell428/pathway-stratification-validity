"""定量・正規化の選択が主成分構造と対照の水準に与える影響を測る（原稿 §3.8 の表）。

主張は「内部整合性の絶対値は正規化の選択で桁が変わるので、絶対値の報告は情報を
持たない」である。これを言うには、同じデータ・同じ遺伝子集合に対して正規化だけを
差し替えた比較が要る。

遺伝子集合は本文採用の 11,623 遺伝子に固定する。フィルタを変えると遺伝子集合が
変わり、正規化の効果とフィルタの効果が混ざるため。フィルタ側の議論は §2.2 と
expression_filter_sensitivity.py が担う。

測る量（いずれも安静時・検証側の個人）
  第 1 主成分の寄与率
  第 1 主成分と「その検体の平均発現量」「その検体の検出遺伝子割合」の Spearman 相関
    → 生物学ではなく「発現が何遺伝子に集中しているか」を拾っていないかの検査
  ランダム遺伝子集合の内部整合性の中央値
    → 対照の水準そのものが正規化で動くことの提示

出力: results/tables/gse81046/normalization_comparison.csv
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import METADATA, TABLES, expr_path, load_config, rng
from ..preprocessing.build_rnaseq_matrix import (gene_map_from_gtex,
                                                 quantile_normalize,
                                                 read_tar_tpm, tmm_factors)
from ..scoring.methods import _standardize_rows

N_RANDOM_SETS = 200
RANDOM_SET_SIZE = 50


def collapse_to_symbols(mat: pd.DataFrame, ni_cols: list[str],
                        gmap: pd.Series) -> pd.DataFrame:
    """代表 Ensembl ID を安静時平均で選び、symbol に畳む（本編と同じ規則）。"""
    mat = mat.loc[mat.index.intersection(gmap.index)]
    mean_ni = mat[ni_cols].mean(axis=1)
    order = pd.DataFrame({"symbol": gmap.loc[mat.index].values, "mean": mean_ni.values},
                         index=mat.index)
    rep = order.sort_values(["symbol", "mean"], ascending=[True, False]) \
               .groupby("symbol").head(1)
    out = mat.loc[rep.index]
    out.index = pd.Index(rep["symbol"].values, name="gene")
    return out


def mean_pairwise_rho(block: np.ndarray) -> float:
    """遺伝子ペアの個人間 Spearman 相関の平均（metrics 実装と同じ定義）。"""
    if block.shape[0] < 2:
        return np.nan
    r = spearmanr(block, axis=1).statistic
    r = np.atleast_2d(r)
    iu = np.triu_indices(r.shape[0], k=1)
    return float(np.nanmean(r[iu]))


def measure(expr: pd.DataFrame, detected_frac: pd.Series, gen) -> dict:
    x = expr.to_numpy(dtype=np.float64)
    z = np.nan_to_num(_standardize_rows(x))
    _, s, vt = np.linalg.svd(z - z.mean(axis=1, keepdims=True), full_matrices=False)
    var_ratio = float((s[0] ** 2) / (s ** 2).sum())
    pc1 = vt[0]

    mean_expr = x.mean(axis=0)
    det = detected_frac.reindex(expr.columns).to_numpy(dtype=np.float64)
    # 主成分の符号は数値計算上不定で、変種ごとに反転しうる。そのまま並べると
    # 符号の違いが「相関が消えた」ように見えてしまうので、平均発現量と正相関する
    # 向きに固定してから両方の相関を出す。
    rho_mean = float(spearmanr(pc1, mean_expr).statistic)
    if rho_mean < 0:
        pc1, rho_mean = -pc1, -rho_mean
    rho_det = float(spearmanr(pc1, det).statistic)

    idx = np.arange(x.shape[0])
    nulls = [mean_pairwise_rho(z[gen.choice(idx, RANDOM_SET_SIZE, replace=False)])
             for _ in range(N_RANDOM_SETS)]
    return {
        "第1主成分の寄与率": round(100 * var_ratio, 1),
        "平均発現量との相関": round(rho_mean, 3),
        "検出率との相関": round(rho_det, 3),
        "検出率との相関の絶対値": round(abs(rho_det), 3),
        "ランダム対照の整合性(中央値)": round(float(np.nanmedian(nulls)), 4),
        "ランダム対照の整合性(5-95%)": (
            f"{np.nanpercentile(nulls, 5):.3f}〜{np.nanpercentile(nulls, 95):.3f}"),
    }


def main() -> int:
    cfg = load_config("analysis")
    resting = cfg["conditions"]["resting"]
    gen = rng(41)

    # 本文が使っている遺伝子集合と個人を固定する
    universe = pd.read_parquet(expr_path(resting)).index
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    keep = set(split["validation"])
    print(f"遺伝子集合を本文と同一に固定: {len(universe):,} 遺伝子 / 検証側 {len(keep)} 名")

    print("RAW.tar を 1 回読む（3 変種を同じ入力から作る）")
    tpm, cnt, _ = read_tar_tpm()
    gmap = gene_map_from_gtex()
    ni_all = [c for c in tpm.columns if c.endswith(f"|{resting}")]
    ni_cols = sorted(c for c in ni_all if c.split("|")[0] in keep)

    # 検体ごとの検出遺伝子割合は全 feature に対する性質なので、畳み込み前に測る
    detected_frac = (cnt[ni_cols] > 0).mean(axis=0)
    detected_frac.index = ni_cols

    variants: dict[str, pd.DataFrame] = {}

    log_tpm = np.log2(tpm + 1.0)
    a = collapse_to_symbols(log_tpm, ni_all, gmap)
    variants["log2(TPM+1)"] = a

    b = quantile_normalize(a[ni_cols])
    variants["log2(TPM+1) + 分位正規化"] = b

    common = cnt.index.intersection(tpm.index)
    c = cnt.loc[common].to_numpy(dtype=np.float64)
    f = tmm_factors(c)
    eff = c.sum(axis=0) * f
    logcpm = pd.DataFrame(np.log2(c / eff * 1e6 + 1.0), index=common, columns=cnt.columns)
    variants["TMM log-CPM（本研究の採用）"] = collapse_to_symbols(logcpm, ni_all, gmap)

    rows = {}
    for name, mat in variants.items():
        cols = [c for c in ni_cols if c in mat.columns]
        sub = mat.reindex(universe).dropna(how="all")[cols]
        print(f"\n--- {name} ---")
        print(f"  {sub.shape[0]:,} genes x {sub.shape[1]} 個人")
        rows[name] = measure(sub, detected_frac, gen)
        for k, v in rows[name].items():
            print(f"  {k}: {v}")

    out = pd.DataFrame(rows).T
    out.to_csv(TABLES / "normalization_comparison.csv", encoding="utf-8")
    print("\n=== 表 4 に入る値 ===")
    print(out.to_string())
    print(f"\n-> {TABLES / 'normalization_comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
