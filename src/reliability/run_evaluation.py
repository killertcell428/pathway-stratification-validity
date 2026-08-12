"""遺伝子セット x 評価軸のテーブルを作る（本研究の中心的な出力）。

評価は検証側の個人だけで行う。データ由来モジュールは探索側で作られているので、
全ファミリーが同じ検証個人で評価される（同条件での比較になる）。

出力
  results/tables/gene_set_metrics.csv  1 行 = 1 遺伝子セット
  results/tables/family_summary.csv    ファミリー別の要約
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..common import (METADATA, RAW, TABLES, expr_path, gene_mean_path,
                      load_config, modules_path, rng)
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import METHODS, ScoringContext, score_set
from .metrics import (
    alpha_from_mean_rho,
    condition_effect,
    cronbach_alpha,
    direction_concordance,
    empirical_null,
    matched_random_sets,
    mean_pairwise_rho,
    pooled_set_score,
    rank_consistency,
    split_half_reliability,
)


def load_all_sets(gs_cfg: dict) -> dict[str, tuple[str, list[str]]]:
    """{set_name: (family, genes)} を作る。"""
    sets: dict[str, tuple[str, list[str]]] = {}
    for family, spec in gs_cfg["families"].items():
        lib = spec.get("library")
        if lib:
            path = RAW / "gene_sets" / f"{family}__{lib}.gmt"
            if not path.exists():
                print(f"  [warn] {path.name} がない。pixi run download を実行する")
                continue
            parsed = parse_gmt(path.read_text(encoding="utf-8"))
        else:
            path = modules_path()
            if not path.exists():
                print(f"  [warn] {path.name} がない。pixi run modules を実行する")
                continue
            parsed = parse_gmt(path.read_text(encoding="utf-8"))
        for name, genes in parsed.items():
            sets[f"{family}|{name}"] = (family, genes)
        print(f"  {family:12s} {len(parsed)} sets")

    for name, spec in gs_cfg.get("anchor_sets", {}).items():
        genes = spec.get("genes")
        if genes:
            sets[f"anchor|{name}"] = ("anchor", [g.upper() for g in genes])
    return sets


def pooled_z(rest: pd.DataFrame, pert: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """条件をまたいだ共通基準で z 化する（条件間差を残すため）。"""
    genes = rest.index.intersection(pert.index)
    donors = [d for d in rest.columns if d in set(pert.columns)]
    a, b = rest.loc[genes, donors], pert.loc[genes, donors]
    both = pd.concat([a, b], axis=1).to_numpy(dtype=np.float64)
    mu = both.mean(axis=1, keepdims=True)
    sd = both.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = np.nan
    z = (both - mu) / sd
    n = a.shape[1]
    return (
        pd.DataFrame(z[:, :n], index=genes, columns=donors),
        pd.DataFrame(z[:, n:], index=genes, columns=donors),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="被覆率の下限を config より優先して上書きする（感度分析用）")
    ap.add_argument("--suffix", default="", help="出力ファイル名に付ける接尾辞")
    args = ap.parse_args(argv)

    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    if args.min_coverage is not None:
        print(f"[感度分析] min_coverage を {gs_cfg['filters']['min_coverage']} -> {args.min_coverage} に上書き")
        gs_cfg["filters"]["min_coverage"] = args.min_coverage
    resting = cfg["conditions"]["resting"]
    pair = cfg["conditions"]["cross_perturbation_pair"]
    main_pert = pair[0]

    print("[1/5] 行列と個人分割を読む")
    expr = {}
    for cond in [resting] + list(cfg["conditions"]["perturbed"]):
        p = expr_path(cond)
        if p.exists():
            expr[cond] = pd.read_parquet(p)
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr[resting].columns]
    print(f"  評価に使う個人: {len(val)} 名（検証側）")

    rest_val = expr[resting][val]
    ctx_rest = ScoringContext(rest_val)
    ctx_pert = ScoringContext(expr[main_pert][[d for d in val if d in expr[main_pert].columns]])

    z_rest, z_pert = pooled_z(
        expr[resting][val], expr[main_pert][[d for d in val if d in expr[main_pert].columns]]
    )

    cross_donors = [
        d for d in val if d in expr[pair[0]].columns and d in expr[pair[1]].columns
    ]
    ctx_cross = {
        c: ScoringContext(expr[c][cross_donors]) for c in pair if c in expr
    }
    print(f"  摂動をまたいだ比較に使える個人: {len(cross_donors)} 名 ({pair[0]} & {pair[1]})")

    print("[2/5] 遺伝子セットを読む")
    all_sets = load_all_sets(gs_cfg)
    print(f"  合計 {len(all_sets)} sets")

    print("[3/5] 発現量分位で陰性対照の抽出プールを作る")
    gene_mean = pd.read_csv(gene_mean_path(), index_col=0)["mean_expression"]
    gene_mean = gene_mean.loc[gene_mean.index.intersection(rest_val.index)]
    deciles = pd.qcut(gene_mean, 10, labels=False, duplicates="drop")
    decile_of_gene = deciles.to_dict()
    genes_by_decile: dict[int, list[str]] = defaultdict(list)
    for g, d in decile_of_gene.items():
        genes_by_decile[int(d)].append(g)

    # 第 2 の対照: 個人間分散の分位をそろえる。データ由来モジュールは高分散遺伝子から
    # 作られるため、発現量マッチだけでは「分散が大きいほど相関が出やすい」交絡が残る。
    gene_var = rest_val.loc[gene_mean.index].var(axis=1)
    var_deciles = pd.qcut(gene_var, 10, labels=False, duplicates="drop")
    var_decile_of_gene = var_deciles.to_dict()
    genes_by_var_decile: dict[int, list[str]] = defaultdict(list)
    for g, d in var_decile_of_gene.items():
        genes_by_var_decile[int(d)].append(g)

    print("[4/5] セットごとに評価する")
    filt = gs_cfg["filters"]
    nc = cfg["metrics"]["negative_control"]["n_sets_per_size"]
    repeats = cfg["metrics"]["internal_consistency"]["split_half_repeats"]
    gen = rng(2)

    rows = []
    skipped = defaultdict(int)
    for i, (name, (family, genes)) in enumerate(all_sets.items(), 1):
        if i % 500 == 0:
            print(f"  {i}/{len(all_sets)} ...")
        present = ctx_rest.present(genes)
        coverage = len(present) / len(genes) if genes else 0.0
        if len(present) < filt["min_genes"]:
            skipped["min_genes"] += 1
            continue
        if len(present) > filt["max_genes"]:
            skipped["max_genes"] += 1
            continue
        if coverage < filt["min_coverage"] and family != "anchor":
            skipped["coverage"] += 1
            continue

        idx = ctx_rest.idx(present)
        S_sub = ctx_rest.S[idx]
        ic = mean_pairwise_rho(S_sub)
        alpha = cronbach_alpha(S_sub)
        sh, sh_sb = split_half_reliability(S_sub, repeats, gen)
        ic_pert = mean_pairwise_rho(ctx_pert.S[ctx_pert.idx(present)])

        # 陰性対照: 同じサイズ・同じ発現分位から引いたランダムセット
        null_ic, null_sh = [], []
        for rs in matched_random_sets(present, decile_of_gene, genes_by_decile, nc, gen):
            S_null = ctx_rest.S[ctx_rest.idx(rs)]
            null_ic.append(mean_pairwise_rho(S_null))
            null_sh.append(split_half_reliability(S_null, 10, gen)[0])
        null = empirical_null(ic, null_ic)
        null_sh_mean = float(np.nanmean(null_sh)) if null_sh else np.nan

        null_var_ic = [
            mean_pairwise_rho(ctx_rest.S[ctx_rest.idx(rs)])
            for rs in matched_random_sets(
                present, var_decile_of_gene, genes_by_var_decile, nc, gen
            )
        ]
        null_var = empirical_null(ic, null_var_ic)

        # 条件効果（共通基準の z のまま平均する。条件内で再標準化しない）
        z_genes = [g for g in present if g in z_rest.index]
        eff = condition_effect(
            pooled_set_score(z_rest.loc[z_genes].to_numpy()),
            pooled_set_score(z_pert.loc[z_genes].to_numpy()),
        )
        conc = direction_concordance(
            expr[resting].loc[[g for g in present if g in expr[resting].index], z_rest.columns].to_numpy(),
            expr[main_pert].loc[[g for g in present if g in expr[main_pert].index], z_rest.columns].to_numpy(),
        )

        # 摂動をまたいだ順位の一貫性
        if len(ctx_cross) == 2:
            a = score_set(ctx_cross[pair[0]], present, "zmean")
            b = score_set(ctx_cross[pair[1]], present, "zmean")
            cross = rank_consistency(a, b)
        else:
            cross = {"rho": np.nan, "rho_p": np.nan, "tertile_swap": np.nan}

        # 手法間一致（同じ個人を 4 手法で順位づけしたときの一致度）
        scores = {m: score_set(ctx_rest, present, m) for m in METHODS}
        pairs = []
        for j, m1 in enumerate(METHODS):
            for m2 in METHODS[j + 1 :]:
                s1, s2 = scores[m1], scores[m2]
                ok = ~(np.isnan(s1) | np.isnan(s2))
                if ok.sum() > 10 and s1[ok].std() > 0 and s2[ok].std() > 0:
                    pairs.append(rank_consistency(s1, s2)["rho"])
        method_rho = float(np.nanmean(pairs)) if pairs else np.nan
        method_rho_min = float(np.nanmin(pairs)) if pairs else np.nan

        rows.append(
            {
                "set": name,
                "family": family,
                "n_genes_annotated": len(genes),
                "n_genes_present": len(present),
                "coverage": coverage,
                "mean_expression": float(gene_mean.reindex(present).mean()),
                "internal_consistency": ic,
                "internal_consistency_perturbed": ic_pert,
                "cronbach_alpha": alpha,
                "alpha_null_expected": alpha_from_mean_rho(null["null_mean"], len(present)),
                "split_half": sh,
                "split_half_sb": sh_sb,
                "split_half_null": null_sh_mean,
                **null,
                **{f"var_{k}": v for k, v in null_var.items()},
                **eff,
                **conc,
                "cross_perturbation_rho": cross["rho"],
                "cross_perturbation_p": cross["rho_p"],
                "cross_perturbation_tertile_swap": cross["tertile_swap"],
                "method_agreement_mean": method_rho,
                "method_agreement_min": method_rho_min,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  評価できたセットが 0 件。フィルタ設定を見直す")
        return 1
    print(f"  評価済み {len(df)} sets / 除外 {dict(skipped)}")

    print("[5/5] 多重比較補正して書き出す")
    for col, out in (
        ("delta_p", "delta_q"),
        ("direction_p", "direction_q"),
        ("null_p", "null_q"),
        ("var_null_p", "var_null_q"),
    ):
        ok = df[col].notna()
        df[out] = np.nan
        if ok.sum() > 1:
            df.loc[ok, out] = multipletests(
                df.loc[ok, col], method=cfg["multiple_testing"]["method"]
            )[1]

    df.to_csv(TABLES / f"gene_set_metrics{args.suffix}.csv", index=False, encoding="utf-8")

    summary = (
        df.groupby("family")
        .agg(
            n_sets=("set", "size"),
            median_size=("n_genes_present", "median"),
            ic_median=("internal_consistency", "median"),
            ic_null_median=("null_mean", "median"),
            ic_above_null_frac=("null_q", lambda s: float((s < 0.05).mean())),
            ic_above_varnull_frac=("var_null_q", lambda s: float((s < 0.05).mean())),
            varnull_median=("var_null_mean", "median"),
            split_half_median=("split_half", "median"),
            split_half_null_median=("split_half_null", "median"),
            alpha_median=("cronbach_alpha", "median"),
            alpha_null_median=("alpha_null_expected", "median"),
            cond_effect_sig_frac=("delta_q", lambda s: float((s < 0.05).mean())),
            cond_effect_d_median=("cohens_d", lambda s: float(np.nanmedian(np.abs(s)))),
            direction_sig_frac=("direction_q", lambda s: float((s < 0.05).mean())),
            cross_rho_median=("cross_perturbation_rho", "median"),
            method_agreement_median=("method_agreement_mean", "median"),
        )
        .round(4)
    )
    summary.to_csv(TABLES / f"family_summary{args.suffix}.csv", encoding="utf-8")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
