"""目的変数の候補ごとに、コホート間でどれだけ順位が再現するかを出す。

なぜ要るか
  WP2 の事前登録は、当初の目的変数（増殖補正後の予後関連）を
  |meta-PCNA 相関| に差し替えている。その根拠が「補正後は Loi と NKI で
  順位が再現しない（rho = 0.19-0.22）が、採用した量は再現する（0.967-0.971）」
  という一枚の表で、事前登録は**その表を登録と一緒に公開すると約束している**。
  ところが表は原稿の中の数字としてしか存在せず、ファイルが無かった。
  ここで計算して CSV に出し、監査で照合できるようにする。

除外
  META-PCNA は既発表の予後シグネチャではなく、この研究が目的変数に使う
  増殖参照そのもの（prolif.metagene と 129/129 一致）。フレームから外す。
  残すと |相関| が構成上 1.000 になり、中央値も再現性も歪む。

出力: results/tables/wp2_target_reproducibility.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

EXCLUDE = "META-PCNA"

# (表示名, 列の作り方) 事前登録の表と同じ順序・同じ名前にする
QUANTITIES: list[tuple[str, str]] = [
    ("abs correlation with meta-PCNA (primary target)", "abs_cor"),
    ("correlation with meta-PCNA, signed", "pcna_cor"),
    ("unadjusted log(HR) (secondary target)", "log_hr_raw"),
    ("proliferation-adjusted log(HR)", "log_hr_adj"),
    ("ratio adjusted / unadjusted", "retention"),
]

PAIRS = [("Loi OS", "Loi RFS"), ("Loi OS", "NKI RFS"), ("Loi RFS", "NKI RFS")]


def main() -> int:
    from pathlib import Path

    t = Path(__file__).resolve().parents[1] / "results" / "tables"
    v = pd.read_csv(t / "wp2_venet_outcome.csv")
    n_before = v.signature.nunique()
    v = v[v.signature != EXCLUDE].copy()
    print(f"シグネチャ {n_before} → {v.signature.nunique()}（{EXCLUDE} を除外）")

    v["abs_cor"] = v.pcna_cor.abs()
    v["log_hr_raw"] = np.log(v.hr_raw)
    v["log_hr_adj"] = np.log(v.hr_adj)

    rows = []
    for label, col in QUANTITIES:
        rec = {"quantity": label}
        for a, b in PAIRS:
            piv = v.pivot_table(index="signature", columns="cohort", values=col)
            if a not in piv or b not in piv:
                rec[f"{a} vs {b}"], rec[f"n ({a} vs {b})"] = np.nan, 0
                continue
            d = piv[[a, b]].dropna()
            rec[f"{a} vs {b}"] = round(float(d[a].corr(d[b], method="spearman")), 3)
            rec[f"n ({a} vs {b})"] = len(d)
        rows.append(rec)

    out = pd.DataFrame(rows)
    dest = t / "wp2_target_reproducibility.csv"
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"\n書き出し: {dest.relative_to(dest.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
