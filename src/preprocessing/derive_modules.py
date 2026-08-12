"""発現データ自体から共発現モジュールを作る（比較対象の 6 番目のファミリー）。

循環を避けるため、モジュールの導出は探索側の個人だけで行う。評価は検証側の個人で
行うので、「自分で作ったモジュールを自分で評価して内部整合性が高い」という自明な
結果にはならない。

出力: data/interim/data_derived_modules.gmt
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..common import METADATA, expr_path, load_config, modules_path

N_TOP_VARIABLE = 3000
MIN_SIZE = 5
MAX_SIZE = 200
DISTANCE_CUT = 0.7   # 1 - Spearman rho。0.7 は rho=0.3 相当で切る


def main() -> int:
    cfg = load_config("analysis")
    resting = cfg["conditions"]["resting"]

    expr = pd.read_parquet(expr_path(resting))
    split = json.loads((METADATA / "donor_split.json").read_text(encoding="utf-8"))
    donors = [d for d in split["discovery"] if d in expr.columns]
    print(f"導出に使う個人: {len(donors)} 名（探索側のみ）")

    sub = expr[donors]
    top = sub.var(axis=1).nlargest(N_TOP_VARIABLE).index
    x = sub.loc[top].to_numpy(dtype=np.float64)

    # 個人方向の順位相関（外れ値の影響を抑える）
    ranks = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)
    ranks = (ranks - ranks.mean(axis=1, keepdims=True)) / ranks.std(axis=1, keepdims=True)
    rho = (ranks @ ranks.T) / ranks.shape[1]
    np.fill_diagonal(rho, 1.0)

    dist = np.clip(1.0 - rho, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(z, t=DISTANCE_CUT, criterion="distance")

    modules: dict[str, list[str]] = {}
    for lab in np.unique(labels):
        genes = list(top[labels == lab])
        if MIN_SIZE <= len(genes) <= MAX_SIZE:
            modules[f"MODULE_{lab:04d}"] = sorted(genes)

    out = modules_path()
    with out.open("w", encoding="utf-8") as f:
        for name, genes in modules.items():
            f.write(f"{name}\tderived from {resting} discovery donors\t" + "\t".join(genes) + "\n")

    sizes = [len(g) for g in modules.values()]
    print(
        f"モジュール {len(modules)} 個（サイズ 中央値 {int(np.median(sizes)) if sizes else 0}, "
        f"最大 {max(sizes) if sizes else 0}） -> {out.name}"
    )
    if not modules:
        print("  [warn] モジュールが 0 個。DISTANCE_CUT を緩める必要がある")
    return 0


if __name__ == "__main__":
    sys.exit(main())
