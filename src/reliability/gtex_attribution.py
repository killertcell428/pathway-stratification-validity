"""GTEx v8 全血で「主成分を何が説明するか」を分解する（結果 4 の RNA-seq 再現）。

これまでに分かっていること
  精製単球（マイクロアレイ）  PC1 の 0.899 が測定チップ
  全血（マイクロアレイ）      血球組成 1.053 / 技術 0.125 / 個人属性 0.087
  マクロファージ（RNA-seq）   祖先集団は PC1 の 1.9% しか説明しない

GTEx で新しく評価できるのは次の 2 点。
  1. RNA-seq で、技術・生物学・血球組成を同じ土俵で比べる（n が約 4 倍）
  2. 虚血時間（SMTSISCH）という、他のコホートにない軸。死後・採取後の時間経過は
     生物学とも技術ともつかない要因で、これが上位主成分を説明するなら
     「支配軸はその実験系で最大の攪乱要因」という主張がさらに強くなる

説明率は一元配置 R^2（連続変数は 5 分位に離散化）。群数で R^2 が変わるため、
ラベルを並べ替えた偶然水準を引いた「超過分」で比較する。

出力: results/tables/gtex_blood/pc_attribution.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..common import INTERIM, METADATA, TABLES, load_config, rng
from ..download.fetch_gene_sets import parse_gmt
from ..common import RAW
from ..scoring.methods import _standardize_rows


def r2_oneway(y: np.ndarray, labels: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return np.nan
    within = sum(float(((y[labels == k] - y[labels == k].mean()) ** 2).sum())
                 for k in np.unique(labels))
    return 1.0 - within / total


def discretize(v: pd.Series, q: int = 5) -> np.ndarray:
    """連続変数は分位で離散化する（カテゴリ変数と同じ土俵で R^2 を比べるため）。"""
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().mean() > 0.8 and num.nunique() > q:
        return pd.qcut(num.rank(method="first"), q, labels=False).to_numpy()
    return pd.factorize(v.astype(str))[0]


def main() -> int:
    cfg = load_config("analysis")
    att = cfg["attribution"]
    gen = rng(31)

    expr = pd.read_parquet(INTERIM / "expr_blood.parquet")
    cov = pd.read_csv(METADATA / "covariates.csv")
    cov = cov.set_index("SAMPID").reindex(expr.columns)
    print(f"検体 {expr.shape[1]} / 遺伝子 {expr.shape[0]:,}")

    x = expr.to_numpy(dtype=np.float64)
    z = np.nan_to_num(_standardize_rows(x))
    _, s, vt = np.linalg.svd(z - z.mean(axis=1, keepdims=True), full_matrices=False)
    var_ratio = (s ** 2) / float((s ** 2).sum())
    n_pc = int(att["n_pcs"])
    print("主成分の寄与率:", "  ".join(f"PC{i+1} {var_ratio[i]:.1%}" for i in range(n_pc)))

    # 血球組成の代理: 細胞種マーカーセットのスコア
    gmt = RAW / "gene_sets" / "celltype__PanglaoDB_Augmented_2021.gmt"
    sets = parse_gmt(gmt.read_text(encoding="utf-8")) if gmt.exists() else {}
    comp: dict[str, np.ndarray] = {}
    index = {g: i for i, g in enumerate(expr.index)}
    for full in att["composition_sets"]:
        name = full.split("|", 1)[1]
        genes = [g for g in sets.get(name, []) if g in index]
        if len(genes) >= 5:
            comp[name] = z[np.array([index[g] for g in genes])].mean(axis=0)
    print(f"組成の代理として使える細胞種マーカー: {len(comp)} 種")

    factors: dict[str, tuple[str, np.ndarray]] = {}
    for c in att["technical"]:
        if c in cov.columns and cov[c].notna().sum() > len(cov) * 0.5:
            factors[c] = ("技術", discretize(cov[c]))
    for c in att["biological"]:
        if c in cov.columns and cov[c].notna().sum() > len(cov) * 0.5:
            factors[c] = ("生物学", discretize(cov[c]))
    for k, v in comp.items():
        # ラベルは組織依存なので設定から取る。骨格筋で「血球組成」と出ると誤りになる。
        factors[f"組成:{k}"] = (att.get("composition_label", "血球組成"),
                              discretize(pd.Series(v)))

    rows = []
    nperm = int(att["n_permutations"])
    for k in range(n_pc):
        pc = vt[k]
        for name, (kind, lab) in factors.items():
            ok = ~pd.isna(lab)
            obs = r2_oneway(pc[ok], np.asarray(lab)[ok])
            perm = np.array([r2_oneway(pc[ok], gen.permutation(np.asarray(lab)[ok]))
                             for _ in range(nperm)])
            rows.append({
                "pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]),
                "factor": name, "kind": kind,
                "r2": obs, "r2_chance": float(np.nanmean(perm)),
                "excess": obs - float(np.nanmean(perm)),
                "p": float((np.sum(perm >= obs) + 1) / (nperm + 1)),
                "n_groups": int(len(np.unique(np.asarray(lab)[ok]))),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "pc_attribution.csv", index=False, encoding="utf-8")

    # 集計は分散重み付け（各主成分の寄与率で重み、対象 PC の寄与率合計で正規化）。
    # 単純合計だと PC5（寄与 3.3%）を PC1（29.3%）と同じ重みで足すことになり、
    # 別コホートとの比較ができなくなる。GSE35846 側（technical_axes_check.py）と
    # 同じ定義に揃えてある。
    w = df.drop_duplicates("pc").set_index("pc").var_ratio
    total_w = float(w.sum())
    df["weighted"] = df.excess * df.var_ratio / total_w
    df.to_csv(TABLES / "pc_attribution.csv", index=False, encoding="utf-8")

    print(f"\n=== 分類別の超過説明率（PC1-PC{n_pc} を分散で重み付け、"
          f"重みの合計 {total_w:.3f} で正規化）===")
    tot = df.groupby("kind").weighted.sum().sort_values(ascending=False)
    for k, v in tot.items():
        print(f"  {k:8s} {v:6.3f}   （参考: 単純合計 {df[df.kind==k].excess.sum():.3f}）")
    print("\n=== 要因別（分散重み付け）===")
    for f, v in df.groupby("factor").weighted.sum().sort_values(ascending=False).items():
        print(f"  {f:24s} {v:+.3f}")
    print("\n=== 上位の要因（超過分の大きい順）===")
    top = df.sort_values("excess", ascending=False).head(12)
    for _, r in top.iterrows():
        print(f"  {r.pc} ({r.var_ratio:5.1%})  {r.factor:24s} [{r.kind}] "
              f"R²={r.r2:.3f} 偶然={r.r2_chance:.3f} 超過={r.excess:+.3f} p={r.p:.3f}")
    print(f"\n-> {TABLES/'pc_attribution.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
