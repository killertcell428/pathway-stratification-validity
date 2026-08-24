"""ssGSEA を手法間一致度の解析に加える（3.5 節の穴を塞ぐ）。

なぜ要るか
  3.5 節は「整合性の低いセットではスコアリング手法を替えると個人順位が変わる」を示し、
  4.2 節末尾の処方（手法を替えて結論が変わるならセットの側を疑え）の唯一の根拠になっている。
  ところがその根拠は自前実装 4 手法の中でしか測っておらず、実際に最も広く使われている
  ssGSEA と GSVA が入っていない。「precision medicine の single-sample pathway score を
  問題視する論文なのに最も普及した 2 手法が無い」は査読で必ず来る。

  塞ぎ方は安い。**手法間一致度は注釈セットのみで計算でき、対照セットを必要としない。**
  2.4 節が自前実装を正当化した理由（セットごとに数千の対照を同一処理で通す必要）は
  一致度の解析には当てはまらない。そこで実物の ssGSEA（gseapy 1.3.1、Barbie ら 2009 の
  実装）を注釈セットに対して走らせ、既存 4 手法に足して一致度を測り直す。

  GSVA も同じ理由で入れられる。ただし Bioconductor の R パッケージなので、
  主環境（conda-forge のみ・Python だけ）には入らない。envs/gsva に
  linux-64（WSL）専用の別環境を切り出し、受け渡しを
  「発現行列と GMT を入力、スコア行列を出力」のファイル境界に限定した。
  **抽出（どの 427 セットを使うか）はこのスクリプトに一本化する。**
  R 側で抽出をやり直すと ssGSEA と別のセットになり、比較の意味がなくなる。

  使い方
    1. python -m src.reliability.ssgsea_agreement --export-for-gsva <DIR>
    2. WSL 側で: cd ~/gsva-env && pixi run gsva <DIR>/expr.tsv <DIR>/sets.gmt <DIR>/scores.tsv
    3. python -m src.reliability.ssgsea_agreement --gsva-scores <DIR>/scores.tsv

出力: results/tables/ssgsea_agreement.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pathlib import Path

from ..common import METADATA, TABLES, expr_path, load_config, rng
from ..scoring.methods import METHODS, ScoringContext, score_set
from .run_evaluation import load_all_sets

N_SAMPLE = 400          # 層化抽出するセット数。ssGSEA は 1 セットあたりのコストが高い
COHERENCE_BINS = [-1.0, 0.02, 0.10, 0.30, 1.0]
BIN_LABELS = ["<0.02", "0.02-0.10", "0.10-0.30", ">=0.30"]


def _wsl_path(p: Path) -> str:
    r"""Windows のパスを WSL から見えるパスに直す（C:\dir -> /mnt/c/dir）。"""
    s = str(p.resolve()).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def run_ssgsea(expr: pd.DataFrame, sets: dict[str, list[str]]) -> pd.DataFrame:
    """実物の ssGSEA（gseapy）でセット x 個人のスコア行列を作る。"""
    import gseapy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gseapy.ssgsea(data=expr, gene_sets=sets, outdir=None,
                            sample_norm_method="rank", min_size=2, max_size=10 ** 6,
                            permutation_num=0, no_plot=True, threads=4, verbose=False)
    df = res.res2d
    # gseapy は long 形式（Name = 検体、Term = セット、NES/ES = スコア）で返す
    col = "NES" if "NES" in df.columns else "ES"
    wide = df.pivot(index="Term", columns="Name", values=col).astype(float)
    return wide.reindex(columns=[c for c in expr.columns if c in wide.columns])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-for-gsva", metavar="DIR",
                    help="GSVA に渡す発現行列と GMT を書き出して終了する")
    ap.add_argument("--gsva-scores", metavar="TSV",
                    help="GSVA のスコア行列を読み、6 手法目として一致度に加える")
    args = ap.parse_args(argv)

    cfg = load_config("analysis")
    gs_cfg = load_config("gene_sets")
    resting = cfg["conditions"]["resting"]

    expr = pd.read_parquet(expr_path(resting))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    rest_val = expr[val]
    ctx = ScoringContext(rest_val)
    print(f"検証側 {len(val)} 名 / 遺伝子 {rest_val.shape[0]:,}")

    # 既存の評価結果から整合性を取り、その水準で層化抽出する
    metrics = pd.read_csv(TABLES / "gene_set_metrics.csv")
    metrics["bin"] = pd.cut(metrics.internal_consistency, bins=COHERENCE_BINS,
                            labels=BIN_LABELS)
    all_sets = load_all_sets(gs_cfg)
    gen = rng(97)
    take = []
    for b, grp in metrics.groupby("bin", observed=True):
        n = min(len(grp), max(40, int(round(N_SAMPLE * len(grp) / len(metrics)))))
        take += list(grp.sample(n=n, random_state=int(gen.integers(10 ** 6))).set)
    sample = [s for s in take if s in all_sets]
    sets = {s: [g for g in all_sets[s][1] if g in ctx.index] for s in sample}
    sets = {k: v for k, v in sets.items() if len(v) >= 2}
    print(f"層化抽出 {len(sets)} セット（整合性の水準で層化）")

    if args.export_for_gsva:
        # GSVA は別環境（WSL / R）で走らせるので、入力をファイルで渡す。
        # 抽出はここで確定させ、R 側では一切選び直さない。
        out = Path(args.export_for_gsva)
        out.mkdir(parents=True, exist_ok=True)
        rest_val.to_csv(out / "expr.tsv", sep="\t", encoding="utf-8")
        with (out / "sets.gmt").open("w", encoding="utf-8") as fh:
            for s, g in sets.items():
                fh.write(f"{s}\tT26\t" + "\t".join(g) + "\n")
        print(f"  発現行列 -> {out / 'expr.tsv'}（{rest_val.shape[0]:,} x {rest_val.shape[1]}）")
        print(f"  遺伝子セット -> {out / 'sets.gmt'}（{len(sets)} 件）")
        print("\nWSL 側で次を実行する:")
        print("  cd ~/gsva-env && pixi run gsva \\")
        print(f"    '{_wsl_path(out / 'expr.tsv')}' \\")
        print(f"    '{_wsl_path(out / 'sets.gmt')}' \\")
        print(f"    '{_wsl_path(out / 'scores.tsv')}'")
        return 0

    print("ssGSEA を走らせる（gseapy）...")
    ss = run_ssgsea(rest_val, sets)
    print(f"  ssGSEA スコア: {ss.shape[0]} セット x {ss.shape[1]} 名")

    print("既存 4 手法のスコアを計算する ...")
    scores = {m: {s: score_set(ctx, g, m) for s, g in sets.items()} for m in METHODS}
    scores["ssgsea"] = {s: ss.loc[s].reindex(ctx.samples).to_numpy()
                        for s in sets if s in ss.index}

    if args.gsva_scores:
        gv = pd.read_csv(args.gsva_scores, sep="\t", index_col=0)
        # R 側で列名が変換されることがあるので、検体名の対応を明示的に取る
        gv.columns = [c.replace(".", "-") if c not in ctx.samples else c for c in gv.columns]
        missing = [s for s in sets if s not in gv.index]
        if missing:
            print(f"  [warn] GSVA スコアに無いセット {len(missing)} 件")
        scores["gsva"] = {s: gv.loc[s].reindex(ctx.samples).to_numpy()
                          for s in sets if s in gv.index}
        print(f"  GSVA スコア: {gv.shape[0]} セット x {gv.shape[1]} 名"
              f"（GSVA {Path(args.gsva_scores).parent.name} 由来）")

    methods = list(METHODS) + ["ssgsea"] + (["gsva"] if args.gsva_scores else [])
    ic = metrics.set_index("set").internal_consistency
    rows = []
    for s in sets:
        vals = {}
        for m in methods:
            v = scores[m].get(s)
            if v is not None and np.isfinite(v).sum() >= 3:
                vals[m] = v
        if len(vals) < 2:
            continue
        pairs = {}
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                if a in vals and b in vals:
                    ok = np.isfinite(vals[a]) & np.isfinite(vals[b])
                    if ok.sum() >= 3:
                        pairs[f"{a}|{b}"] = spearmanr(vals[a][ok], vals[b][ok]).statistic
        if not pairs:
            continue
        # 手法の集合ごとに切り分ける。列名と中身がずれると本文の数値が静かに変わる。
        # 以前は agreement_mean_5 に list(pairs.values()) を入れていたため、
        # GSVA を足した途端に「5 手法」列が 6 手法の値になっていた。
        inhouse = [v for k, v in pairs.items()
                   if "ssgsea" not in k and "gsva" not in k]
        five = [v for k, v in pairs.items() if "gsva" not in k]      # 自前 4 + ssGSEA
        with_ss = [v for k, v in pairs.items() if "ssgsea" in k]
        with_gv = [v for k, v in pairs.items() if "gsva" in k]
        rec = {
            "set": s, "family": all_sets[s][0], "n_genes": len(sets[s]),
            "internal_consistency": float(ic.get(s, np.nan)),
            "agreement_mean_4": float(np.nanmean(inhouse)) if inhouse else np.nan,
            "agreement_min_4": float(np.nanmin(inhouse)) if inhouse else np.nan,
            "agreement_mean_5": float(np.nanmean(five)) if five else np.nan,
            "agreement_min_5": float(np.nanmin(five)) if five else np.nan,
            "agreement_mean_ssgsea": float(np.nanmean(with_ss)) if with_ss else np.nan,
            "agreement_min_ssgsea": float(np.nanmin(with_ss)) if with_ss else np.nan,
        }
        if with_gv:
            rec.update({
                "agreement_mean_6": float(np.nanmean(list(pairs.values()))),
                "agreement_min_6": float(np.nanmin(list(pairs.values()))),
                "agreement_mean_gsva": float(np.nanmean(with_gv)),
                "agreement_min_gsva": float(np.nanmin(with_gv)),
            })
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df.internal_consistency, bins=COHERENCE_BINS, labels=BIN_LABELS)
    df.to_csv(TABLES / "ssgsea_agreement.csv", index=False, encoding="utf-8")

    print(f"\n=== 整合性の水準別の手法間一致度（{len(df)} セット）===")
    aggs = dict(
        n=("set", "size"),
        mean_4=("agreement_mean_4", "median"), min_4=("agreement_min_4", "median"),
        mean_5=("agreement_mean_5", "median"), min_5=("agreement_min_5", "median"),
        ss_mean=("agreement_mean_ssgsea", "median"),
        ss_min=("agreement_min_ssgsea", "median"))
    if "agreement_mean_6" in df.columns:
        aggs.update(mean_6=("agreement_mean_6", "median"),
                    min_6=("agreement_min_6", "median"),
                    gv_mean=("agreement_mean_gsva", "median"),
                    gv_min=("agreement_min_gsva", "median"))
    g = df.groupby("bin", observed=True).agg(**aggs)
    print(g.round(3).to_string())
    print("\n  列の意味: _4 は自前実装 4 手法のみ、_5 は ssGSEA を加えた 5 手法、")
    print("            _6 は GSVA も加えた 6 手法、ss_ / gv_ はその手法を含む対のみ")

    ok = df.internal_consistency.notna()
    cols = [("agreement_mean_4", "4 手法"), ("agreement_mean_5", "5 手法"),
            ("agreement_mean_ssgsea", "ssGSEA を含む対")]
    if "agreement_mean_6" in df.columns:
        cols += [("agreement_mean_6", "6 手法"),
                 ("agreement_mean_gsva", "GSVA を含む対")]
    for col, lab in cols:
        r = spearmanr(df.loc[ok, "internal_consistency"], df.loc[ok, col],
                      nan_policy="omit")
        print(f"  ρ(整合性, {lab}) = {r.statistic:.3f} (p = {r.pvalue:.2e})")
    print(f"\n-> {TABLES/'ssgsea_agreement.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
