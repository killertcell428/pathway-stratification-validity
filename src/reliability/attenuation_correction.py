"""減衰補正を実際に当てて、判定が動くかを測る（古典テスト理論の第 3 の柱）。

なぜ要るか
  観測される遺伝子間相関は、遺伝子ごとの測定信頼性で縮んでいる。
  古典テスト理論では r_true = r_obs / sqrt(rel_i * rel_j) で戻す（Spearman の減衰補正）。
  したがって本研究が報告する個人間の共変動は真値の下限であり、
  「共変動が小さい」という結論はこの縮みの産物ではないかという反論が立つ。
  限界として述べるだけでは反論を吸収できないので、実際に補正して判定が動くか測る。

  補正できるのは遺伝子レベルの信頼性が測れるコホートだけである。
  同一個人・同一条件の反復測定を持つのは GSE47353（接種前 2 時点、56 名）だけなので、
  ここで測る。主コホートには反復測定がないため補正できない（限界として残る）。

何を測るか
  1. 遺伝子ごとの信頼性（2 時点間の順位相関）
  2. 注釈セットの補正後の共変動
  3. **同じ対照にも同じ補正を当てたときの共変動**
  4. 補正前後で合格判定がどれだけ動くか

  ここが要点である。対照は構成遺伝子ごとの平均発現量分位でそろえてあり、
  信頼性は発現量と相関するので、補正は両側にほぼ同じだけかかる。
  相対比較として報告している以上、補正は結論を動かさないはずである。
  「はず」で済ませずに測る。

計算
  r_ij = (S_i . S_j) / n なので、T_i = S_i / sqrt(rel_i) と置けば
  補正後の平均ペア相関 = (||sum_i T_i||^2 / n - sum_i 1/rel_i) / (g(g-1))
  になる。O(gn) のまま計算できる（対角が 1 でなく 1/rel_i になるだけ）。

出力: results/tables/attenuation_correction.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..common import INTERIM, TABLES, load_config, rng
from ..scoring.methods import _rank_rows, _standardize_rows
from .metrics import draw_index_matrix, empirical_null
from .retest_check import RETEST
from .run_evaluation import load_all_sets

CHUNK = 2_000
# 信頼性がこれ以下の遺伝子は補正から外す。1/rel が発散して補正後の相関が
# 1 を超える（減衰補正の既知の病理）。除外数は必ず報告する。
MIN_RELIABILITY = 0.10


def disattenuated_mean_rho(S: np.ndarray, inv_sqrt_rel: np.ndarray,
                           inv_rel: np.ndarray, idx: np.ndarray) -> float:
    """減衰補正後の平均ペア相関。T_i = S_i / sqrt(rel_i) の行和で計算する。"""
    g = idx.size
    if g < 2:
        return np.nan
    acc = (S[idx] * inv_sqrt_rel[idx, None]).sum(axis=0)
    diag = inv_rel[idx].sum()
    return (float(acc @ acc) / S.shape[1] - diag) / (g * (g - 1))


def disattenuated_batch(S: np.ndarray, inv_sqrt_rel: np.ndarray,
                        inv_rel: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(対照, 位置) の行番号行列に対する補正後の平均ペア相関。"""
    b, g = idx.shape
    acc = np.zeros((b, S.shape[1]), dtype=np.float64)
    for j in range(g):
        rows = idx[:, j]
        acc += S[rows] * inv_sqrt_rel[rows, None]
    diag = inv_rel[idx].sum(axis=1)
    return (np.einsum("bn,bn->b", acc, acc) / S.shape[1] - diag) / (g * (g - 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-control", type=int, default=2_000,
                    help="対照数。補正の有無で判定がどう動くかを見るだけなので主解析より少なくてよい")
    args = ap.parse_args(argv)

    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    alpha = cfg["multiple_testing"]["alpha"]
    gen = rng(29)

    a = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[0]}.parquet")
    b = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[1]}.parquet")
    genes = a.index.intersection(b.index)
    donors = [d for d in a.columns if d in set(b.columns)]
    a, b = a.loc[genes, donors], b.loc[genes, donors]
    n = len(donors)
    print(f"反復測定の対: {n} 名 / 遺伝子 {len(genes)}")

    # 遺伝子ごとの信頼性。共変動の定義（順位ベース）に合わせて順位で取る。
    Ra = _standardize_rows(_rank_rows(a.to_numpy(dtype=np.float64)))
    Rb = _standardize_rows(_rank_rows(b.to_numpy(dtype=np.float64)))
    rel = (Ra * Rb).mean(axis=1)
    print(f"遺伝子レベル信頼性（2 時点間の順位相関）: 中央値 {np.median(rel):.3f}"
          f" / 四分位 {np.percentile(rel, 25):.3f}-{np.percentile(rel, 75):.3f}")
    usable = rel > MIN_RELIABILITY
    print(f"  信頼性 {MIN_RELIABILITY} 以下で補正から外した遺伝子: "
          f"{int((~usable).sum())} 件 ({100 * (~usable).mean():.1f}%)")

    # 共変動は day0 の順位行列で測る（retest_check と同じ）
    S = Rb
    inv_sqrt_rel = np.where(usable, 1.0 / np.sqrt(np.clip(rel, MIN_RELIABILITY, None)), 0.0)
    inv_rel = np.where(usable, 1.0 / np.clip(rel, MIN_RELIABILITY, None), 0.0)

    gene_mean = pd.read_csv(INTERIM / "valid_gene_expression.csv",
                            index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(genes)]
    print(f"  信頼性と平均発現量の順位相関: "
          f"{stats.spearmanr(rel[[genes.get_loc(g) for g in gene_mean.index]], gene_mean)[0]:.3f}"
          f"（対照を発現量分位でそろえる根拠）")

    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    index = {g: i for i, g in enumerate(genes)}
    pools: dict[int, list[int]] = defaultdict(list)
    for g, d in dec.items():
        pools[int(d)].append(index[g])
    pool_by_decile = {d: np.array(v, dtype=np.int64) for d, v in pools.items()}

    all_sets = load_all_sets(gs_cfg)
    filt = gs_cfg["filters"]
    rows = []
    for i, (name, (family, gl)) in enumerate(all_sets.items(), 1):
        if i % 500 == 0:
            print(f"  {i}/{len(all_sets)} ...", flush=True)
        present = [g for g in gl if g in index]
        cov = len(present) / len(gl) if gl else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        # 補正できるのは信頼性が測れた遺伝子だけ。落ちた分は記録する。
        pos = [g for g in present if g in dec and usable[index[g]]]
        if len(pos) < filt["min_genes"]:
            continue
        idx = np.array([index[g] for g in pos], dtype=np.int64)
        dec_pos = np.array([dec[g] for g in pos], dtype=np.int64)

        raw = disattenuated_mean_rho(S, np.ones_like(inv_sqrt_rel),
                                     np.ones_like(inv_rel), idx)
        cor = disattenuated_mean_rho(S, inv_sqrt_rel, inv_rel, idx)

        null_raw, null_cor = [], []
        filled = 0
        while filled < args.n_control:
            take = min(CHUNK, args.n_control - filled)
            m = draw_index_matrix(pool_by_decile, dec_pos, take, gen)
            null_raw.append(disattenuated_batch(S, np.ones_like(inv_sqrt_rel),
                                                np.ones_like(inv_rel), m))
            null_cor.append(disattenuated_batch(S, inv_sqrt_rel, inv_rel, m))
            filled += take
        nr = empirical_null(raw, np.concatenate(null_raw).tolist())
        nc = empirical_null(cor, np.concatenate(null_cor).tolist())
        rows.append({
            "set": name, "family": family, "n_genes": len(pos),
            "n_dropped": len(present) - len(pos),
            "ic_raw": raw, "ic_corrected": cor,
            "null_raw": nr["null_mean"], "null_corrected": nc["null_mean"],
            "p_raw": nr["null_p_empirical"], "p_corrected": nc["null_p_empirical"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("評価できたセットが 0 件")
        return 1
    for col, out in (("p_raw", "q_raw"), ("p_corrected", "q_corrected")):
        ok = df[col].notna()
        df[out] = np.nan
        df.loc[ok, out] = multipletests(df.loc[ok, col], method="fdr_bh")[1]
    df["pass_raw"] = df.q_raw < alpha
    df["pass_corrected"] = df.q_corrected < alpha
    df.to_csv(TABLES / "attenuation_correction.csv", index=False, encoding="utf-8")

    print(f"\n評価 {len(df)} セット（対照 {args.n_control:,} 個）")
    print(f"  補正前: 共変動の中央値 {df.ic_raw.median():.4f} / 対照 {df.null_raw.median():.4f}"
          f" / 合格 {100 * df.pass_raw.mean():.1f}%")
    print(f"  補正後: 共変動の中央値 {df.ic_corrected.median():.4f} / 対照 {df.null_corrected.median():.4f}"
          f" / 合格 {100 * df.pass_corrected.mean():.1f}%")
    print(f"  補正で共変動は {df.ic_corrected.median() / df.ic_raw.median():.2f} 倍、"
          f"対照は {df.null_corrected.median() / df.null_raw.median():.2f} 倍になる")
    flip = int((df.pass_raw != df.pass_corrected).sum())
    print(f"  判定が変わったセット {flip} / {len(df)} 件 "
          f"（補正で合格 {int((~df.pass_raw & df.pass_corrected).sum())} / "
          f"不合格 {int((df.pass_raw & ~df.pass_corrected).sum())}）")
    print(f"  補正から外れた遺伝子の中央値 {int(df.n_dropped.median())} 件 / セット")
    print("\nファミミリー別の合格率(%)")
    fam = (100 * df.groupby("family")[["pass_raw", "pass_corrected"]].mean()).round(1)
    print(fam.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
