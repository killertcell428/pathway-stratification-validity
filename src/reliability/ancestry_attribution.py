"""GSE81046 で「祖先集団が主成分を説明するか」を測る。

このコホートはアフリカ系（AF）とヨーロッパ系（EU）の免疫応答差を調べるために
設計されている。設計変数として明示的に対比された生物学的個人差が、発現の主要な
個人差軸をどれだけ説明するかを見る。ここで小さければ、
「支配軸はその実験系で最大の攪乱要因であって、セットが名乗る生物学ではない」
という主張が、生物学が最初から仕込まれているコホートでも成り立つことになる。

祖先は個人 ID の接頭辞（AF / EU）から取る。GEO のファイル名規約に含まれている。

説明率は一元配置 R^2。2 群しかないので群数由来の膨らみは小さいが、他のコホートと
同じ手続きに揃えるため、ラベルを並べ替えた偶然水準を引いた超過分で比較する。

出力: results/tables/gse81046/ancestry_attribution.csv
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from ..common import METADATA, TABLES, expr_path, load_config, rng
from ..scoring.methods import _standardize_rows

PREFIX_RE = re.compile(r"^([A-Z]+)")

# GSE81046 の設定には attribution 節がないため、他コホート（GTEx・主コホート）と
# 同じ値を既定に置く。ここを揃えないと偶然水準の推定精度が変わって比較できない。
DEFAULT_ATTRIBUTION = {"n_pcs": 5, "n_permutations": 200}


def r2_oneway(y: np.ndarray, labels: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return np.nan
    within = sum(float(((y[labels == k] - y[labels == k].mean()) ** 2).sum())
                 for k in np.unique(labels))
    return 1.0 - within / total


def main() -> int:
    cfg = load_config("analysis")
    att = {**DEFAULT_ATTRIBUTION, **cfg.get("attribution", {})}
    gen = rng(37)

    resting = cfg["conditions"]["resting"]
    expr = pd.read_parquet(expr_path(resting))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    # 主コホートと同じく検証側で評価する（探索側はモジュール導出に使っている）
    cols = [c for c in expr.columns if c in set(split["validation"])]
    expr = expr[sorted(cols)]

    anc = np.array([PREFIX_RE.match(c).group(1) for c in expr.columns])
    uniq, cnt = np.unique(anc, return_counts=True)
    print(f"検体 {expr.shape[1]} / 遺伝子 {expr.shape[0]:,}")
    print("祖先集団:", dict(zip(uniq.tolist(), cnt.tolist())))
    if len(uniq) < 2:
        raise RuntimeError(f"祖先集団が 1 群しかない: {uniq}")

    z = np.nan_to_num(_standardize_rows(expr.to_numpy(dtype=np.float64)))
    _, s, vt = np.linalg.svd(z - z.mean(axis=1, keepdims=True), full_matrices=False)
    var_ratio = (s ** 2) / float((s ** 2).sum())
    n_pc = int(att["n_pcs"])
    nperm = int(att["n_permutations"])
    print("主成分の寄与率:", "  ".join(f"PC{i+1} {var_ratio[i]:.1%}" for i in range(n_pc)))

    lab = pd.factorize(anc)[0]
    rows = []
    for k in range(n_pc):
        pc = vt[k]
        obs = r2_oneway(pc, lab)
        perm = np.array([r2_oneway(pc, gen.permutation(lab)) for _ in range(nperm)])
        chance = float(np.nanmean(perm))
        rows.append({
            "pc": f"PC{k+1}", "var_ratio": float(var_ratio[k]),
            "kind": "個人属性", "factor": "祖先集団(AF/EU)",
            "r2": obs, "r2_chance": chance, "excess": obs - chance,
            "p": float((np.sum(perm >= obs) + 1) / (nperm + 1)),
            "n_groups": int(len(uniq)),
        })
    df = pd.DataFrame(rows)
    out = TABLES / "ancestry_attribution.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print("\n=== 祖先集団による説明率 ===")
    for _, r in df.iterrows():
        print(f"  {r.pc} ({r.var_ratio:5.1%})  R²={r.r2:.4f} 偶然={r.r2_chance:.4f} "
              f"超過={r.excess:+.4f} p={r.p:.3f}")
    p1 = df.iloc[0]
    print(f"\n第 1 主成分（全分散の {p1.var_ratio:.1%}）に対する説明率は "
          f"{p1.r2:.1%}、偶然水準との差は {p1.excess:+.4f}（p = {p1.p:.2f}）")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
