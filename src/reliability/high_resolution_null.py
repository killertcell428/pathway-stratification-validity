"""対照セットを段階的に増やし、経験 p 値のまま BH-FDR を通す（正規近似を捨てる）。

なぜ要るか
  主解析は対照 20 個で走っていた。経験 p は 1/(B+1) より小さくならないので、
  20 個では最小 0.048 にしかならず、数千セットに BH-FDR をかけると全滅する。
  そのため 20 個の平均と SD から正規近似の p を作って FDR にかけていた。
  しかし平均ペア Spearman の帰無分布は上下に有界で右裾が歪んでおり、
  正規近似で極端な z を確率に変換する根拠は弱い。合格率（11.1% など）が
  その p 値から作られている以上、ここは論文の中心結果の土台になっている。

  そこで対照を 1,000 個から始め、裾にいるセットだけ 10,000 個、100,000 個まで
  追加する（adaptive permutation）。経験 p の下限が BH の閾値より十分下がるので、
  正規近似を一切使わずに合格・不合格を決められる。

計算量をどう下げたか
  S は行ごとに順位を取って標準化した行列なので、各行の分散は ddof=0 で厳密に 1。
  よって c = S S^T / n の対角は 1 で、
      平均ペア相関 = (c.sum() - g) / (g(g-1)) = (||sum_i S_i||^2 / n - g) / (g(g-1))
  が成り立つ。O(g^2 n) の行列積が O(gn) の行和になる。
  同値性は tests/test_high_resolution_null.py で 1e-11 未満まで確認している。

  この式が使えるのは S に NaN がない場合だけである（順位は必ず分散を持つので
  実際には NaN が出ない）。念のため実行時に検査している。

出力
  results/tables/high_resolution_null{suffix}.csv   1 行 = 1 遺伝子セット
  results/tables/high_resolution_summary{suffix}.csv ファミリー別・全体の合格率の比較
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..common import (METADATA, TABLES, expr_path, gene_mean_path, load_config,
                      rng)
from ..scoring.methods import ScoringContext
from .metrics import draw_null_rho, mean_pairwise_rho_fast
from .run_evaluation import load_all_sets

# 段階的に増やす対照数。裾にいるセットだけ次の段に進む。
STAGES = (1_000, 10_000, 100_000)
# 現段階で「観測値以上の対照」がこの数以下なら、経験 p の下限が効いている
# 可能性があるので次の段に進める。5 は Besag & Clifford の逐次モンテカルロで
# 使われる打ち切り数と同じ発想（裾の分解能だけを買い足す）。
ESCALATE_IF_EXCEED_AT_MOST = 5
# 一度に確保する対照のかたまり。(CHUNK, n) の作業配列を使うのでメモリはこれで決まる。


def build_pools(
    values: pd.Series, gene_index: dict[str, int]
) -> tuple[dict[str, int], dict[int, np.ndarray]]:
    """分位ごとの抽出プールを、遺伝子名ではなく S の行番号で持つ。"""
    deciles = pd.qcut(values, 10, labels=False, duplicates="drop")
    decile_of_gene = {g: int(d) for g, d in deciles.items()}
    rows: dict[int, list[int]] = defaultdict(list)
    for g, d in decile_of_gene.items():
        rows[d].append(gene_index[g])
    return decile_of_gene, {d: np.array(v, dtype=int) for d, v in rows.items()}


def adaptive_null(
    S: np.ndarray,
    observed: float,
    pool_by_decile: dict[int, np.ndarray],
    decile_of_position: np.ndarray,
    generator: np.random.Generator,
    stages: tuple[int, ...] = STAGES,
) -> dict[str, float]:
    """段階的に対照を増やし、経験 p と分布の形を返す。"""
    drawn: list[np.ndarray] = []
    total = 0
    for target in stages:
        need = target - total
        if need > 0:
            drawn.append(
                draw_null_rho(S, pool_by_decile, decile_of_position, need, generator)
            )
            total = target
        vals = np.concatenate(drawn)
        exceed = int(np.sum(vals >= observed))
        if exceed > ESCALATE_IF_EXCEED_AT_MOST:
            break
    vals = np.concatenate(drawn)
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    return {
        "n_control": int(vals.size),
        "null_mean": mu,
        "null_sd": sd,
        "null_median": float(np.median(vals)),
        "null_p975": float(np.quantile(vals, 0.975)),
        "null_skew": float(((vals - mu) ** 3).mean() / sd**3) if sd > 0 else np.nan,
        "null_z": (observed - mu) / sd if sd > 0 else np.nan,
        "exceed": exceed,
        "p_empirical": float((exceed + 1) / (vals.size + 1)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="", help="読み書きするテーブルの接尾辞")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="被覆率の下限を config より優先して上書きする（感度分析用）")
    ap.add_argument("--stages", default=",".join(str(s) for s in STAGES),
                    help="対照数の段（カンマ区切り）")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N セットだけ（検査用）")
    args = ap.parse_args(argv)
    stages = tuple(int(s) for s in args.stages.split(","))

    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    if args.min_coverage is not None:
        gs_cfg["filters"]["min_coverage"] = args.min_coverage
    resting = cfg["conditions"]["resting"]

    src = TABLES / f"gene_set_metrics{args.suffix}.csv"
    if not src.exists():
        print(f"  {src.name} がない。先に主解析を走らせる")
        return 1
    base = pd.read_csv(src).set_index("set")

    print("[1/4] 行列と個人分割を読む")
    expr = pd.read_parquet(expr_path(resting))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    rest_val = expr[val]
    ctx = ScoringContext(rest_val)
    S = ctx.S
    if np.isnan(S).any():
        print("  S に NaN がある。行和による高速化は使えない")
        return 1
    print(f"  遺伝子 {S.shape[0]} x 個人 {S.shape[1]}")

    print("[2/4] 分位プールを作る（発現量・個人間分散の 2 系統）")
    gene_mean = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(rest_val.index)]
    dec_expr, pool_expr = build_pools(gene_mean, ctx.index)
    gene_var = rest_val.loc[gene_mean.index].var(axis=1)
    dec_var, pool_var = build_pools(gene_var, ctx.index)

    print("[3/4] セットごとに段階的な対照を引く")
    all_sets = load_all_sets(gs_cfg)
    gen = rng(2)
    rows = []
    targets = [s for s in base.index if s in all_sets]
    if args.limit:
        targets = targets[: args.limit]
    for i, name in enumerate(targets, 1):
        if i % 200 == 0:
            print(f"  {i}/{len(targets)} ...", flush=True)
        genes = all_sets[name][1]
        present = ctx.present(genes)
        # 主解析と同じ位置集合を使う。分位が付かない遺伝子は対照を引けないので落とす
        # （主解析の matched_random_sets も同じ位置を飛ばしている）。
        pos_e = [g for g in present if g in dec_expr]
        pos_v = [g for g in present if g in dec_var]
        if len(pos_e) < 2:
            continue
        idx_all = ctx.idx(present)
        observed = mean_pairwise_rho_fast(S, idx_all)

        e = adaptive_null(
            S, observed, pool_expr,
            np.array([dec_expr[g] for g in pos_e]), gen, stages,
        )
        v = adaptive_null(
            S, observed, pool_var,
            np.array([dec_var[g] for g in pos_v]), gen, stages,
        )
        rows.append(
            {
                "set": name,
                "family": all_sets[name][0],
                "n_genes_present": len(present),
                "internal_consistency": observed,
                "internal_consistency_stored": float(base.at[name, "internal_consistency"]),
                **{f"hr_{k}": val_ for k, val_ in e.items()},
                **{f"hr_var_{k}": val_ for k, val_ in v.items()},
                "null_q_stored": float(base.at[name, "null_q"]),
                "null_p_stored": float(base.at[name, "null_p"]),
                "null_sd_stored": float(base.at[name, "null_sd"]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  評価できたセットが 0 件")
        return 1

    # 主解析の観測値と一致しているか。ここがずれていたら文脈の作り方が違う。
    gap = float((df["internal_consistency"] - df["internal_consistency_stored"]).abs().max())
    print(f"  観測値の最大乖離 {gap:.2e}（主解析と同じ文脈で計算できているかの検査）")
    if gap > 1e-9:
        print("  主解析の内部整合性と一致しない。設定の読み込みを見直す")
        return 1

    print("[4/4] 経験 p に BH-FDR をかける")
    method = cfg["multiple_testing"]["method"]
    for col, out in (("hr_p_empirical", "hr_q"), ("hr_var_p_empirical", "hr_var_q")):
        ok = df[col].notna()
        df[out] = np.nan
        df.loc[ok, out] = multipletests(df.loc[ok, col], method=method)[1]

    df.to_csv(TABLES / f"high_resolution_null{args.suffix}.csv", index=False,
              encoding="utf-8")

    alpha = cfg["multiple_testing"]["alpha"]
    df["pass_hr"] = df["hr_q"] < alpha
    df["pass_stored"] = df["null_q_stored"] < alpha
    summary = (
        df.groupby("family")
        .agg(
            n_sets=("set", "size"),
            pass_frac_stored=("pass_stored", "mean"),
            pass_frac_hr=("pass_hr", "mean"),
            pass_frac_hr_var=("hr_var_q", lambda s: float((s < alpha).mean())),
            median_n_control=("hr_n_control", "median"),
            median_null_skew=("hr_null_skew", "median"),
            median_sd_ratio=("hr_null_sd", "median"),
        )
        .round(4)
    )
    overall = pd.DataFrame(
        {
            "n_sets": [len(df)],
            "pass_frac_stored": [df["pass_stored"].mean()],
            "pass_frac_hr": [df["pass_hr"].mean()],
            "pass_frac_hr_var": [float((df["hr_var_q"] < alpha).mean())],
            "median_n_control": [df["hr_n_control"].median()],
            "median_null_skew": [df["hr_null_skew"].median()],
            "median_sd_ratio": [df["hr_null_sd"].median()],
        },
        index=["__all__"],
    ).round(4)
    summary = pd.concat([summary, overall])
    summary.to_csv(TABLES / f"high_resolution_summary{args.suffix}.csv", encoding="utf-8")
    print(summary.to_string())
    print()
    print(f"  対照 20 個・正規近似での合格率 {df['pass_stored'].mean():.4f}")
    print(f"  段階的対照・経験 p での合格率   {df['pass_hr'].mean():.4f}")
    print(f"  判定が変わったセット {int((df['pass_hr'] != df['pass_stored']).sum())} 件"
          f"（合格化 {int((df['pass_hr'] & ~df['pass_stored']).sum())} / "
          f"不合格化 {int((~df['pass_hr'] & df['pass_stored']).sum())}）")
    print(f"  帰無分布の歪度の中位数 {df['hr_null_skew'].median():.4f}"
          f"（正規なら 0。正規近似の妥当性の目安）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
