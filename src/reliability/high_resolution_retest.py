"""反復測定信頼性の対照を 10,000 個に増やし、経験 p 値のまま BH-FDR を通す。

なぜ要るか
  high_resolution_null.py と同じ理由である。retest_check.py も対照 20 個の
  平均と SD から正規近似の p を作っていた。表 5 の「対照を有意に上回った注釈セット
  15 / 2,093」はその p 値に BH-FDR をかけた数なので、正規近似を捨てないと
  この数字も土台が弱いままになる。ICC と順位相関の帰無分布は 0 付近に集まって
  右に歪むので、正規近似の当てはまりは内部整合性より悪い可能性がある。

計算量をどう下げたか
  対照 1 個ごとに Python で ICC を呼ぶと 10,000 個 x 2,093 セットで到底終わらない。
  (対照, 個人) の作業配列に位置ごとの行を足し込み、ICC・Spearman・平均ペア相関を
  対照方向にまとめて計算する。ICC(2,1) は測定値を定数倍しても不変なので、
  遺伝子数で割らずに和のまま渡してよい（tests で確認している）。

出力
  results/tables/high_resolution_retest.csv
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
from .metrics import (draw_index_matrix, icc_two_way, mean_pairwise_rho,
                      null_batch, pooled_set_score,
                      rank_consistency)
from .retest_check import RETEST
from .run_evaluation import load_all_sets

CHUNK = 2_000


def summarize(observed: float, vals: np.ndarray, prefix: str) -> dict[str, float]:
    vals = vals[~np.isnan(vals)]
    if vals.size < 5 or np.isnan(observed):
        return {f"{prefix}_null_mean": np.nan, f"{prefix}_null_sd": np.nan,
                f"{prefix}_null_skew": np.nan, f"{prefix}_z": np.nan,
                f"{prefix}_exceed": np.nan, f"{prefix}_p_empirical": np.nan,
                f"{prefix}_n_control": 0}
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    exceed = int(np.sum(vals >= observed))
    return {
        f"{prefix}_null_mean": mu,
        f"{prefix}_null_sd": sd,
        f"{prefix}_null_skew": float(((vals - mu) ** 3).mean() / sd**3) if sd > 0 else np.nan,
        f"{prefix}_z": (observed - mu) / sd if sd > 0 else np.nan,
        f"{prefix}_exceed": exceed,
        f"{prefix}_p_empirical": float((exceed + 1) / (vals.size + 1)),
        f"{prefix}_n_control": int(vals.size),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-control", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=None, help="先頭 N セットだけ（検査用）")
    args = ap.parse_args(argv)

    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    gen = rng(13)

    a = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[0]}.parquet")
    b = pd.read_parquet(INTERIM / f"valid_expr_{RETEST[1]}.parquet")
    genes = a.index.intersection(b.index)
    donors = [d for d in a.columns if d in set(b.columns)]
    a, b = a.loc[genes, donors], b.loc[genes, donors]
    print(f"反復測定の対: {len(donors)} 名 / 遺伝子 {len(genes)}")

    both = np.concatenate([a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)], axis=1)
    mu = both.mean(axis=1, keepdims=True)
    sd = both.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = np.nan
    z = (both - mu) / sd
    n = a.shape[1]
    z_a, z_b = np.ascontiguousarray(z[:, :n]), np.ascontiguousarray(z[:, n:])
    S = _standardize_rows(_rank_rows(b.to_numpy(dtype=np.float64)))
    if np.isnan(z).any() or np.isnan(S).any():
        print("  z または S に NaN がある。まとめ計算の前提が崩れる")
        return 1

    gene_mean = pd.read_csv(INTERIM / "valid_gene_expression.csv", index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(genes)]
    dec = pd.qcut(gene_mean, 10, labels=False, duplicates="drop").to_dict()
    index = {g: i for i, g in enumerate(genes)}
    pools: dict[int, list[int]] = defaultdict(list)
    for g, d in dec.items():
        pools[int(d)].append(index[g])
    pool_by_decile = {d: np.array(v, dtype=np.int64) for d, v in pools.items()}

    all_sets = load_all_sets(gs_cfg)
    filt = gs_cfg["filters"]
    rows = []
    names = list(all_sets)
    if args.limit:
        names = names[: args.limit]
    for i, name in enumerate(names, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(names)} ...", flush=True)
        family, gene_list = all_sets[name]
        present = [g for g in gene_list if g in index]
        cov = len(present) / len(gene_list) if gene_list else 0.0
        if not (filt["min_genes"] <= len(present) <= filt["max_genes"]):
            continue
        if cov < filt["min_coverage"] and family != "anchor":
            continue
        pos = [g for g in present if g in dec]
        if len(pos) < 2:
            continue
        idx = np.array([index[g] for g in present], dtype=np.int64)
        dec_pos = np.array([dec[g] for g in pos], dtype=np.int64)

        s_a, s_b = pooled_set_score(z_a[idx]), pooled_set_score(z_b[idx])
        obs_rho = rank_consistency(s_a, s_b)["rho"]
        obs_icc = icc_two_way(np.column_stack([s_a, s_b]))
        obs_ic = mean_pairwise_rho(S[idx])

        ic_v, icc_v, rho_v = [], [], []
        filled = 0
        while filled < args.n_control:
            take = min(CHUNK, args.n_control - filled)
            m = draw_index_matrix(pool_by_decile, dec_pos, take, gen)
            x, y, r = null_batch(m, z_a, z_b, S)
            ic_v.append(x)
            icc_v.append(y)
            rho_v.append(r)
            filled += take

        rows.append(
            {
                "set": name, "family": family, "n_genes_present": len(present),
                "ic_day0": obs_ic, "icc": obs_icc, "retest_rho": obs_rho,
                **summarize(obs_ic, np.concatenate(ic_v), "hr_ic"),
                **summarize(obs_icc, np.concatenate(icc_v), "hr_icc"),
                **summarize(obs_rho, np.concatenate(rho_v), "hr_rho"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  評価できたセットが 0 件")
        return 1

    stored = TABLES / "retest_metrics.csv"
    if stored.exists():
        old = pd.read_csv(stored).set_index("set")
        common = df["set"].isin(old.index)
        gap = float(
            (df.loc[common, "icc"].to_numpy()
             - old.loc[df.loc[common, "set"], "icc"].to_numpy()).__abs__().max()
        )
        print(f"  観測 ICC の最大乖離 {gap:.2e}（既存出力と同じ文脈かの検査）")
        if gap > 1e-9:
            print("  既存の retest_metrics と一致しない")
            return 1
        for col, out in (("icc_q", "icc_q_stored"), ("ic_q", "ic_q_stored"),
                         ("retest_rho_q", "retest_rho_q_stored")):
            if col in old.columns:
                df[out] = old.loc[df["set"], col].to_numpy()

    method = cfg["multiple_testing"]["method"]
    alpha = cfg["multiple_testing"]["alpha"]
    for col, out in (("hr_ic_p_empirical", "hr_ic_q"),
                     ("hr_icc_p_empirical", "hr_icc_q"),
                     ("hr_rho_p_empirical", "hr_rho_q")):
        ok = df[col].notna()
        df[out] = np.nan
        df.loc[ok, out] = multipletests(df.loc[ok, col], method=method)[1]

    df.to_csv(TABLES / "high_resolution_retest.csv", index=False, encoding="utf-8")
    print(f"\n評価 {len(df)} セット -> high_resolution_retest.csv")
    for label, hr, old_col in (
        ("内部整合性 (day0)", "hr_ic_q", "ic_q_stored"),
        ("ICC", "hr_icc_q", "icc_q_stored"),
        ("反復間 Spearman", "hr_rho_q", "retest_rho_q_stored"),
    ):
        n_hr = int((df[hr] < alpha).sum())
        line = f"  {label:18s} 対照 {args.n_control} 個・経験 p: {n_hr} / {len(df)} 件"
        if old_col in df.columns:
            n_old = int((df[old_col] < alpha).sum())
            line += f"（対照 20 個・正規近似では {n_old} 件）"
        print(line)
    print(f"  帰無分布の歪度の中位数  ic {df.hr_ic_null_skew.median():.3f} / "
          f"icc {df.hr_icc_null_skew.median():.3f} / rho {df.hr_rho_null_skew.median():.3f}")

    blood = df[(df["hr_icc_q"] < alpha)]
    if len(blood):
        print("\n  ICC が対照を上回ったセット:")
        for _, r in blood.sort_values("hr_icc_p_empirical").head(30).iterrows():
            print(f"    {r['set'][:62]:62s} ICC {r['icc']:.3f} 対照 {r['hr_icc_null_mean']:.3f} "
                  f"p {r['hr_icc_p_empirical']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
