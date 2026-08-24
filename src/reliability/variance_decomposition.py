"""4 コホートで、主成分の説明分散を要因群の固有分と共有分に分ける。

何を確かめるか
  既存の帰属解析は、要因を 1 つずつ主成分に当てた一元配置 R^2 を分類ごとに足している。
  要因が相関していれば同じ分散が複数の要因に数えられるので、
  「組成は技術の 2.5 倍」「個人属性は組成の 12 分の 1」という比は保証されない。
  重回帰の全部分集合から固有分・共有分を復元し、その比が保つかを見る。

3 つの量を並べて出す（どれと比べているかを取り違えないため）
  paper_sum   要因ごとの一元配置の超過 R^2 を群内で足したもの。**論文の値の定義**
  oneway_group 群の要因をまとめて 1 本の重回帰に入れた超過 R^2。
              paper_sum との差は、群の**内側**での二重計上にあたる
  shapley     群の間で共有された分散を、投入順序の全並べ替えで等分したもの。
              paper_sum との差が、論文の比に乗っていた二重計上の総量

符号化について
  分解には重回帰が必要で、水準数が多い要因（SMNABTCH は 335 水準）を素で入れると
  設計が飽和する。そこでカテゴリ変数は上位 9 水準 + その他に丸める。
  丸めが効果を削っていないかを見るため、paper_sum は**丸める前**（論文と同じ符号化）と
  **丸めた後**の両方を出す。

出力
  results/tables/<cohort>/variance_decomposition.csv        主成分 x 群
  results/tables/<cohort>/variance_decomposition_shared.csv commonality の共有成分
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from ..common import DATA, RESULTS, ROOT, load_config, rng
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import _standardize_rows
from .commonality import code_factor, decompose, dummy_block, r2

N_PC = 5
NPERM = 200

G35_TECH = ["plate id", "chip_position", "batch effect_date", "rna integrity score"]
G35_BIO = ["gender", "age", "percentage of body fat", "ethnicity"]
G35_COMP = ["Neutrophils", "T Cells Naive", "B Cells", "NK Cells", "Platelets", "Monocytes"]
# 連続として扱う共変量。ここに無いものはすべてカテゴリ。既存解析の CONTINUOUS と揃える。
G35_CONTINUOUS = {"rna integrity score", "age", "percentage of body fat"}
GTEX_CONTINUOUS = {"SMRIN", "SMTSISCH"}


def principal_components(expr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """遺伝子を標準化順位に直してから SVD する（既存解析と同じ手順）。"""
    z = np.nan_to_num(_standardize_rows(expr.to_numpy(dtype=np.float64)))
    _, sv, vt = np.linalg.svd(z - z.mean(axis=1, keepdims=True), full_matrices=False)
    return (sv ** 2) / float((sv ** 2).sum()), vt


def oneway_excess(y: np.ndarray, labels: np.ndarray, gen, nperm: int) -> float:
    """1 要因の超過 R^2。既存解析の r2_with_permutation と同じ量。"""
    x = dummy_block(labels)
    obs = r2(y, x)
    perm = np.array([r2(y, x[gen.permutation(len(y))]) for _ in range(nperm)])
    return obs - float(np.nanmean(perm))


def marker_scores(expr: pd.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    """細胞種マーカーの z 平均。組成の代理。"""
    gmt = DATA / "raw" / "gene_sets" / "celltype__PanglaoDB_Augmented_2021.gmt"
    if not gmt.exists():
        return {}
    sets = parse_gmt(gmt.read_text(encoding="utf-8"))
    z = np.nan_to_num(_standardize_rows(expr.to_numpy(dtype=np.float64)))
    index = {g: i for i, g in enumerate(expr.index)}
    out = {}
    for n in names:
        genes = [g for g in sets.get(n, []) if g in index]
        if len(genes) >= 5:
            out[n] = z[np.array([index[g] for g in genes])].mean(axis=0)
    return out


def raw_labels(v, continuous: bool) -> np.ndarray:
    """論文と同じ符号化（水準を丸めない）。連続変数だけ 5 分位に落とす。"""
    return code_factor(v, continuous, max_levels=10 ** 6)


# --- コホートごとの入力作り ---
# 返す groups は {群名: [(要因名, 丸めた後のラベル, 丸める前のラベル), ...]}

def load_main() -> tuple:
    from .batch_check import load_sdrf_technical
    cfg = load_config("analysis")
    expr = pd.read_parquet(DATA / "interim" / f"expr_{cfg['conditions']['resting']}.parquet")
    split = json.loads((DATA / "metadata" / "donor_split.json").read_text(encoding="utf-8"))
    val = [d for d in split["validation"] if d in expr.columns]
    tech = load_sdrf_technical()
    have = [d for d in val if d in tech.index]
    meta = tech.loc[have]
    var_ratio, vt = principal_components(expr[have])
    # ここは分類ではなく要因ごとに切る。分類で切ると技術 1 群になってしまうが、
    # 3.7 節の主張はチップ単独についてのものなので、チップ固有分が見たい。
    # チップは 59 水準ある。丸めると 3.7 節の 0.899 と対応しなくなるので丸めない。
    groups = {f: [(f, raw_labels(meta[f].to_numpy(), False),
                   raw_labels(meta[f].to_numpy(), False))]
              for f in ("chip", "position", "batch")}
    note = (f"検体 {len(have)} / チップ {meta.chip.nunique()} 枚・位置 "
            f"{meta.position.nunique()} 種・処理バッチ {meta.batch.nunique()} 群。"
            "要因ごとに分解し、チップは水準を丸めていない")
    return var_ratio, vt, groups, note


def load_gse35846() -> tuple:
    expr = pd.read_parquet(DATA / "interim" / "tech_expr_wholeblood.parquet")
    meta = pd.read_csv(DATA / "metadata" / "tech_samples.csv", index_col=0)
    meta = meta.loc[[s for s in expr.columns if s in meta.index]]
    expr = expr[meta.index.tolist()]
    var_ratio, vt = principal_components(expr)
    comp = marker_scores(expr, G35_COMP)

    def cols(names):
        out = []
        for c in names:
            cont = c in G35_CONTINUOUS
            v = meta[c].to_numpy()
            out.append((c, code_factor(v, cont), raw_labels(v, cont)))
        return out

    groups = {
        "技術": cols(G35_TECH),
        "生物学": cols(G35_BIO),
        # 組成の代理はマーカーの z 平均なので常に連続
        "細胞組成": [(k, code_factor(v, True), raw_labels(v, True)) for k, v in comp.items()],
    }
    note = f"検体 {expr.shape[1]} / 組成の代理 {len(comp)} 種"
    return var_ratio, vt, groups, note


def load_gtex(ns: str) -> tuple:
    att = yaml.safe_load((ROOT / "config" / f"analysis_{ns}.yml").read_text(
        encoding="utf-8"))["attribution"]
    expr = pd.read_parquet(DATA / "interim" / ns / "expr_blood.parquet")
    cov = pd.read_csv(DATA / "metadata" / ns / "covariates.csv").set_index("SAMPID")
    cov = cov.reindex(expr.columns)
    var_ratio, vt = principal_components(expr)
    comp = marker_scores(expr, [s.split("|", 1)[1] for s in att["composition_sets"]])

    def cols(names):
        keep = [c for c in names if c in cov.columns and cov[c].notna().sum() > len(cov) * 0.5]
        out = []
        for c in keep:
            cont = c in GTEX_CONTINUOUS
            v = cov[c].to_numpy()
            out.append((c, code_factor(v, cont), raw_labels(v, cont)))
        return out

    groups = {
        "技術": cols(att["technical"]),
        "生物学": cols(att["biological"]),
        att["composition_label"]: [(k, code_factor(v, True), raw_labels(v, True))
                                   for k, v in comp.items()],
    }
    note = (f"検体 {expr.shape[1]} / 組成の代理 {len(comp)} 種 / "
            f"技術 {len(groups['技術'])} 要因・生物学 {len(groups['生物学'])} 要因")
    return var_ratio, vt, groups, note


# 原稿 3.7 節が実際に主張しているのは分類の比ではなく、要因の比である。
#   「安定した個人特性と言える年齢と性別は、合わせて 0.061 にすぎず、死因分類の 5 分の 1 に満たない」
#   「安定した個人特性であるべき年齢と性別は、ここでも最大の雑音軸より 1 桁小さく」
# これを検証するには、年齢・性別と死因分類・虚血時間を別の群に置いた分割が要る。
# 分類（技術/生物学/組成）で切ると、死因分類が年齢・性別と同じ「生物学」に入ってしまい、
# 主張そのものを見られない。要因を全部個別にすると 2^14 部分集合で計算が終わらないので、
# 主張に必要な 4 群にまとめる。
CLAIM_GROUPS = {
    "安定した個人属性": ["AGE", "SEX"],
    "採取時の状況": ["DTHHRDY", "SMTSISCH"],
    "その他の技術": ["SMRIN", "SMGEBTCH", "SMNABTCH", "SMCENTER"],
    # 組成は composition_sets から動的に入れる
}


def load_gtex_claim(ns: str) -> tuple:
    """原稿の主張に合わせた 4 群（安定属性 / 採取状況 / その他技術 / 組成）で分ける。"""
    att = yaml.safe_load((ROOT / "config" / f"analysis_{ns}.yml").read_text(
        encoding="utf-8"))["attribution"]
    expr = pd.read_parquet(DATA / "interim" / ns / "expr_blood.parquet")
    cov = pd.read_csv(DATA / "metadata" / ns / "covariates.csv").set_index("SAMPID")
    cov = cov.reindex(expr.columns)
    var_ratio, vt = principal_components(expr)
    comp = marker_scores(expr, [s.split("|", 1)[1] for s in att["composition_sets"]])

    groups = {}
    for g, names in CLAIM_GROUPS.items():
        keep = [c for c in names if c in cov.columns and cov[c].notna().sum() > len(cov) * 0.5]
        if not keep:
            continue
        groups[g] = [(c, code_factor(cov[c].to_numpy(), c in GTEX_CONTINUOUS),
                      raw_labels(cov[c].to_numpy(), c in GTEX_CONTINUOUS)) for c in keep]
    groups[att["composition_label"]] = [(k, code_factor(v, True), raw_labels(v, True))
                                        for k, v in comp.items()]
    note = (f"検体 {expr.shape[1]} / 主張別 4 群。"
            "年齢・性別を死因分類・虚血時間から分けて置いている")
    return var_ratio, vt, groups, note


COHORTS = {
    "main": (load_main, ""),
    "gse35846": (load_gse35846, ""),
    "gtex_blood": (lambda: load_gtex("gtex_blood"), "gtex_blood"),
    "gtex_muscle": (lambda: load_gtex("gtex_muscle"), "gtex_muscle"),
    "gtex_blood_claim": (lambda: load_gtex_claim("gtex_blood"), "gtex_blood"),
    "gtex_muscle_claim": (lambda: load_gtex_claim("gtex_muscle"), "gtex_muscle"),
}


def run(name: str, nperm: int = NPERM) -> pd.DataFrame:
    loader, out_ns = COHORTS[name]
    print(f"\n{'=' * 74}\n{name}")
    var_ratio, vt, groups, note = loader()
    keys = list(groups)
    blocks = {k: [dummy_block(lab) for _, lab, _ in v] for k, v in groups.items()}
    n = vt.shape[1]
    print(f"  {note}")
    for k, v in blocks.items():
        raw_cols = sum(len(np.unique(r)) - 1 for _, _, r in groups[k])
        print(f"  {k:8s} {len(v)} 要因 / 設計 {sum(m.shape[1] for m in v):3d} 列"
              f"（丸める前なら {raw_cols} 列）")
    n_cols_full = sum(m.shape[1] for v in blocks.values() for m in v)
    print(f"  設計の合計 {n_cols_full} 列 / 検体 {n} 名（列÷検体 = {n_cols_full/n:.2f}）")
    gen = rng(53)

    rows, shared_rows = [], []
    for k in range(min(N_PC, vt.shape[0])):
        pc = vt[k]
        res = decompose(pc, blocks, gen, nperm)
        full = frozenset(keys)
        for g in keys:
            paper = sum(oneway_excess(pc, r, gen, nperm) for _, _, r in groups[g])
            paper_c = sum(oneway_excess(pc, c, gen, nperm) for _, c, _ in groups[g])
            rows.append({
                "pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]), "group": g,
                "paper_sum": paper, "paper_sum_capped": paper_c,
                "oneway_group": res["oneway"][g],
                "unique": res["commonality"][frozenset([g])],
                "shapley": res["shapley"][g],
            })
        for t, v in res["commonality"].items():
            if len(t) > 1:
                shared_rows.append({"pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]),
                                    "shared_between": " x ".join(sorted(t)), "value": v})
        print(f"  PC{k+1} ({var_ratio[k]:5.1%})  全要因の超過 {res['total']:+.3f}"
              f"  観測 R²={res['subsets'][full]['r2']:.3f}"
              f"  偶然 {res['subsets'][full]['r2_chance']:.3f}")

    df = pd.DataFrame(rows)
    sh = pd.DataFrame(shared_rows)

    # 集計は既存解析と同じ分散重み付け（対象 PC の寄与率合計で正規化）
    w = float(df.drop_duplicates("pc").var_ratio.sum())
    value_cols = ["paper_sum", "paper_sum_capped", "oneway_group", "unique", "shapley"]
    for c in value_cols:
        df[f"w_{c}"] = df[c] * df.var_ratio / w
    if len(sh):
        sh["weighted"] = sh.value * sh.var_ratio / w

    tables = RESULTS / "tables" / out_ns if out_ns else RESULTS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    # 名前空間を持たないコホート（main と gse35846）は同じ tables に書くので、
    # ファイル名にコホート名を入れる。入れないと後に走った側が前を上書きする
    # （実際に main が gse35846 の出力を消した）。
    stem = "variance_decomposition_claim" if name.endswith("_claim") else "variance_decomposition"
    if not out_ns:
        stem = f"{stem}_{name}"
    df.to_csv(tables / f"{stem}.csv", index=False, encoding="utf-8")
    if len(sh):
        sh.to_csv(tables / f"{stem}_shared.csv", index=False, encoding="utf-8")

    agg = df.groupby("group")[[f"w_{c}" for c in value_cols]].sum()
    agg.columns = ["論文の定義", "同・丸めた後", "群まとめ", "固有分", "Shapley"]
    agg["二重計上分"] = agg["論文の定義"] - agg["Shapley"]
    print(f"\n  === 分散重み付けの集計（重みの合計 {w:.3f} で正規化）===")
    print("  " + agg.round(3).to_string().replace("\n", "\n  "))
    if len(sh):
        top = sh.groupby("shared_between").weighted.sum().sort_values(
            key=lambda s: s.abs(), ascending=False)
        print("\n  === 群の間で共有された分散 ===")
        for k, v in top.items():
            print(f"    {k:36s} {v:+.3f}")
    print(f"  -> {tables / (stem + '.csv')}")
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=list(COHORTS) + ["all"], default="all")
    ap.add_argument("--nperm", type=int, default=NPERM)
    args = ap.parse_args(argv)
    names = list(COHORTS) if args.cohort == "all" else [args.cohort]
    for n in names:
        try:
            run(n, args.nperm)
        except FileNotFoundError as e:
            print(f"\n{n}: 入力がない（{e}）。先に該当コホートの前処理を走らせる")
    return 0


if __name__ == "__main__":
    sys.exit(main())
