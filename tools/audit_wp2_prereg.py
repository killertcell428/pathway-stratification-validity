"""WP2 事前登録に書いた数値が、成立性検査の出力と一致するかを機械照合する。

事前登録は OSF に出したら取り消せない。手で転記した数値がずれたまま登録すると、
「登録した計画と実際の数値が違う」という最も重い指摘を自分から作ることになる。
`tools/audit_numbers.py` と同じ方式で、CSV から真の値を計算し、
その値が事前登録の本文に文字列として現れるかを見る。

使い方:
  pixi run wp2-audit
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
PREREG = ROOT / "manuscript" / "06-wp2-preregistration-en.md"


def normalize(text: str) -> str:
    """全角・U+2212 マイナス・ノーブレークスペースを ASCII に寄せる。

    原稿側は表示のために U+2212（−）を使う箇所があり、素の比較では
    ASCII のハイフンと一致しない。過去に「−0.014 が見つからない」で
    偽陽性を出したので正規化を入れる。
    """
    t = unicodedata.normalize("NFKC", text)
    return t.replace("−", "-").replace(" ", " ")


def venet() -> pd.DataFrame:
    d = pd.read_csv(T / "wp2_venet_outcome.csv")
    return d[d.excluded.isna() | (d.excluded == "")].copy()


def checks() -> list[tuple[str, str]]:
    """(説明, 本文に現れるべき文字列) の一覧を CSV から作る。"""
    out: list[tuple[str, str]] = []

    # --- 被覆率と除外（TCGA 側）---
    cov = pd.read_csv(T / "wp2_tcga_coverage.csv")
    t = cov[cov.family == "乳がん予後"]
    n_main, n_sens = int(t["主解析"].sum()), int(t["感度分析"].sum())
    out.append(("TCGA 主解析の件数", str(n_main)))
    out.append(("感度分析（3-200 遺伝子）の件数", str(n_sens)))
    out.append(("フィルタ後の被覆率 中央値", f"{t.coverage.median():.1%}"))
    for _, r in t[~t["主解析"]].iterrows():
        out.append((f"除外 {r.signature} の検出数",
                    f"{r.n_present} ({r.coverage:.1%})"))
    big = sorted(t[t["主解析"] & ~t["感度分析"]].signature.tolist())
    out.append(("200 遺伝子超の 9 件", ", ".join(big)))

    # --- 発現フィルタと十分位（TCGA 側）---
    feas = pd.read_csv(T / "wp2_tcga_feasibility.csv")
    row = feas[feas["閾値 log2(FPKM-UQ+1)"] == 1.0].iloc[0]
    out.append(("採用閾値で残る遺伝子数", f"{int(row['残る遺伝子数']):,}"))
    out.append(("十分位のビン幅（最小-最大）",
                f"{int(row['発現量_最小ビン']):,}–{int(row['発現量_最大ビン']):,}"))
    out.append(("閾値 0 で残る遺伝子数",
                f"{int(feas[feas['閾値 log2(FPKM-UQ+1)'] == 0.0].iloc[0]['残る遺伝子数']):,}"))

    # --- 検体の絞り込み ---
    smp = pd.read_csv(T / "wp2_tcga_samples.csv")
    out.append(("原発腫瘍のみ・1 患者 1 検体の患者数", f"{int(smp.keep.sum()):,}"))

    # --- Venet コホートの要約 ---
    s = pd.read_csv(T / "wp2_venet_outcome_summary.csv")
    for _, r in s.iterrows():
        out.append((f"{r['コホート']} の meta-PCNA HR", f"{r['meta-PCNA HR']:.2f}"))
        out.append((f"{r['コホート']} の補正後に有意な件数", str(int(r["補正後に有意"]))))

    # --- 主要目的変数の分布 ---
    v = venet()
    v["abs_cor"] = v.pcna_cor.abs()
    for coh, label in (("Loi RFS", "Loi"), ("NKI RFS", "NKI")):
        g = v[v.cohort == coh]
        out.append((f"{label} の |増殖軸相関| 中央値", f"{g.abs_cor.median():.3f}"))
        out.append((f"{label} の |増殖軸相関| 最小", f"{g.abs_cor.min():.3f}"))

    # --- コホート間再現性（5.4 の表）---
    piv = v.pivot_table(index="signature", columns="cohort", values="abs_cor")
    both = piv[["Loi RFS", "NKI RFS"]].dropna()
    rho = both["Loi RFS"].corr(both["NKI RFS"], method="spearman")
    out.append(("|増殖軸相関| の Loi vs NKI 再現性", f"{rho:.3f}"))

    # --- 検出力（n は CSV から取り、検出力はここで計算する）---
    inter = set(t[t["主解析"]].signature) & set(v[v.cohort == "Loi RFS"].signature) \
        & set(v[v.cohort == "NKI RFS"].signature)
    n_eff = len(inter)
    out.append(("実効 n（3 コホートすべてで採点可）", str(n_eff)))
    for n, label in ((n_eff, "実効 n"), (n_sens, "感度分析 n")):
        for r in (0.30, 0.40, 0.50):
            p = float(norm.cdf(np.arctanh(r) * np.sqrt(n - 3) - norm.ppf(0.975)))
            out.append((f"{label}={n} での検出力 rho={r:.2f}", f"{p:.2f}"))

    # --- 20 通りの探索 ---
    # 事前登録では中央値分割の HR を引用している（6-3 の TCGA 全体の値と定義を
    # 揃えるため）。連続 1SD の列と混同すると別の数字になるので列名を明示する。
    grid = pd.read_csv(T / "wp2_tcga_endpoint_grid.csv")
    live = grid[grid["p(連続 1SD)"].notna()]
    hit = live[(live["p(連続 1SD)"] < 0.05) & (live["HR(連続 1SD)"] > 1)]
    out.append(("評価した組み合わせ数", str(len(live))))
    out.append(("有意だった組み合わせ数", str(len(hit))))
    er = live[(live["コホート"] == "ER 陽性") & (live["終点"] == "OS")].iloc[0]
    out.append(("ER 陽性 x OS の HR（中央値分割）", f"{er['HR(中央値分割)']:.3f}"))
    out.append(("ER 陽性 x OS の p（中央値分割）", f"{er['p(中央値分割)']:.2f}"))

    # --- TCGA で目的変数が作れないこと ---
    o = pd.read_csv(T / "wp2_tcga_outcome_summary.csv").set_index("項目")["値"]
    out.append(("TCGA の meta-PCNA HR", f"{float(o['meta-PCNA の HR']):.2f}"))
    out.append(("TCGA で補正前に有意な件数", str(o["補正前に有意な件数"])))
    out.append(("TCGA で補正後に有意な件数", str(o["補正後に有意な件数"])))
    return out


def main() -> int:
    if not PREREG.exists():
        print(f"事前登録が見つからない: {PREREG}")
        return 1
    text = normalize(PREREG.read_text(encoding="utf-8"))

    rows = checks()
    missing = [(label, value) for label, value in rows
               if normalize(value) not in text]

    # 短い値は部分文字列として偶然一致する。実際に "1.00" が "1.000" に当たって
    # 誤って通った例があったため、桁数の少ない照合は弱いものとして印を付ける。
    def weak(value: str) -> bool:
        return len(value.strip()) < 5

    print(f"=== WP2 事前登録の数値照合（{PREREG.name}）===")
    print(f"照合した項目 {len(rows)} 件 / 不一致 {len(missing)} 件\n")
    for label, value in rows:
        if (label, value) in missing:
            mark = "×"
        elif weak(value):
            mark = "△"      # 一致したが偶然の可能性がある
        else:
            mark = "○"
        print(f"  [{mark}] {label}: {value}")

    n_weak = sum(1 for l, v in rows if (l, v) not in missing and weak(v))
    if n_weak:
        print(f"\n△ {n_weak} 件は値が短く、別の箇所への偶然一致でも通る。"
              "本文で文脈を目視確認すること。")

    if missing:
        print("\n本文に見つからなかった値:")
        for label, value in missing:
            print(f"  - {label}: {value}")
        print("\n登録前に解消すること。")
        return 1
    print("\n不一致なし。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
