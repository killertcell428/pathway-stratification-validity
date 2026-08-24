"""WP2 のアウトカム定義を決めるため、終点 x コホートの全組み合わせを測る。

背景（docs/08-WP2再設計の論点.md #5 の失敗）:
TCGA-BRCA の全生存（OS）では meta-PCNA（増殖の代理、Venet の補正軸）が
予後因子として有意にならなかった（HR 1.29、p = 0.12）。実装は病期・年齢を
陽性対照にして検証済みで正しく動く。したがって OS という終点、または
全患者という母集団が、Venet の設定と噛み合っていないことになる。

Venet ら (2011) が使ったのは Loi（ER 陽性、タモキシフェン投与）と
NKI（リンパ節陰性、大半が未治療）である。TCGA-BRCA は全サブタイプ・
全治療を含み、追跡期間の中央値が 2.4 年しかない。

--------------------------------------------------------------------------
★ この探索の扱い（事前登録での開示義務）

終点とコホートを振って「meta-PCNA が有意になる組み合わせ」を探すのは、
放置すれば検定の水増しになる。次の 3 点で歯止めをかける。

1. **判定は陽性対照（meta-PCNA）だけで行う。** 説明変数（内部整合性の
   対照超過 z）は TCGA では一切計算しない。仮説には触れないまま、
   測定手段が使えるかだけを決める
2. **全組み合わせを出力し、全部を事前登録に載せる。** 勝った 1 セルだけを
   報告して残りを捨てるのが最も危険な形なので、表をそのまま添付する
3. **終点の選択には外部の根拠を優先する。** TCGA-CDR（Liu et al. Cell 2018）は
   BRCA では OS のイベントが不足するとして PFI を推奨している。
   これは本研究の結果を見る前に存在する推奨である

それでも「探索して選んだ」事実は消えないので、事前登録の Prior knowledge に
本表を明記する。
--------------------------------------------------------------------------

出力: results/tables/wp2_tcga_endpoint_grid.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.common import RAW, RESULTS
from tools.wp2_tcga_outcome import (cox, load_expr, load_target_signatures,
                                    logrank, signature_pc1)

TABLES = RESULTS / "tables"
CDR = RAW / "TCGA-CDR_Survival_S1_xena_sp.tsv"
CLINICAL_MATRIX = RAW / "TCGA.BRCA_clinicalMatrix.tsv"

# TCGA-CDR の 4 終点。Liu et al. Cell 2018 は BRCA では PFI を推奨し、
# OS はイベント不足、DFI は定義が不安定と注意している。
ENDPOINTS = ("OS", "DSS", "DFI", "PFI")


def patient_id(sample: str) -> str:
    return "-".join(sample.split("-")[:3])


def load_cdr() -> pd.DataFrame:
    d = pd.read_csv(CDR, sep="\t", low_memory=False)
    d = d[d["cancer type abbreviation"] == "BRCA"].copy()
    d["patient"] = d["sample"].map(patient_id)
    return d.drop_duplicates("patient").set_index("patient")


def load_strata() -> pd.DataFrame:
    """ER 状態・リンパ節・PAM50 を患者単位で返す。"""
    cm = pd.read_csv(CLINICAL_MATRIX, sep="\t", low_memory=False)
    cm["patient"] = cm["sampleID"].map(patient_id) if "sampleID" in cm.columns \
        else cm.iloc[:, 0].map(patient_id)
    cm = cm.drop_duplicates("patient").set_index("patient")
    out = pd.DataFrame(index=cm.index)
    out["er"] = cm.get("breast_carcinoma_estrogen_receptor_status")
    out["pam50"] = cm.get("PAM50Call_RNAseq")
    n = cm.get("pathologic_N")
    out["node_neg"] = n.astype(str).str.startswith("N0") if n is not None else np.nan
    return out


def cohorts(meta: pd.DataFrame) -> dict[str, pd.Index]:
    """母集団の候補。Venet の設定にどれだけ寄せるかで 5 通り。"""
    er_pos = meta.er.astype(str).str.lower() == "positive"
    lum = meta.pam50.isin(["LumA", "LumB"])
    return {
        "全患者": meta.index,
        "ER 陽性": meta.index[er_pos],
        "ER 陽性かつリンパ節陰性": meta.index[er_pos & (meta.node_neg == True)],  # noqa: E712
        "Luminal (PAM50 A/B)": meta.index[lum],
        "Luminal かつリンパ節陰性": meta.index[lum & (meta.node_neg == True)],  # noqa: E712
    }


def main() -> int:
    print("[1/4] データを読む")
    X = load_expr()
    samples = pd.Series({c: patient_id(c) for c in X.columns})
    cdr = load_cdr()
    strata = load_strata()

    # 患者単位に揃える（発現行列は既に 1 患者 1 検体）
    meta = pd.DataFrame({"sample": samples.index}, index=samples.values)
    meta = meta.join(strata, how="left")
    meta = meta[meta.index.isin(cdr.index)]
    print(f"  発現 x CDR で突合できた患者: {len(meta):,}")
    print(f"  ER 陽性: {int((meta.er.astype(str).str.lower() == 'positive').sum()):,} / "
          f"Luminal: {int(meta.pam50.isin(['LumA', 'LumB']).sum()):,} / "
          f"リンパ節陰性: {int((meta.node_neg == True).sum()):,}")  # noqa: E712

    print("[2/4] meta-PCNA を取る（陽性対照）")
    _, pcna = load_target_signatures(set(X.index))
    print(f"  meta-PCNA {len(pcna)} 遺伝子")

    print("[3/4] 終点 x コホートの全組み合わせで meta-PCNA の予後関連を測る")
    rows = []
    for cname, idx in cohorts(meta).items():
        idx = idx.intersection(meta.index)
        if len(idx) < 100:
            rows.append({"コホート": cname, "終点": "-", "n": len(idx),
                         "イベント": np.nan, "備考": "n < 100 で除外"})
            continue
        cols = meta.loc[idx, "sample"].tolist()
        pc1 = signature_pc1(X[cols], pcna)
        z = (pc1 - pc1.mean()) / pc1.std()
        hi = (pc1 > np.median(pc1)).astype(int)

        for ep in ENDPOINTS:
            t, e = cdr.loc[idx, f"{ep}.time"], cdr.loc[idx, ep]
            ok = t.notna() & (t > 0) & e.notna()
            if int(e[ok].sum()) < 10:
                rows.append({"コホート": cname, "終点": ep, "n": int(ok.sum()),
                             "イベント": int(e[ok].sum()),
                             "備考": "イベント < 10 で評価不能"})
                continue
            d = pd.DataFrame({
                "time": t[ok].to_numpy(dtype=float),
                "event": e[ok].to_numpy(dtype=float),
                "z": z[ok.to_numpy()], "hi": hi[ok.to_numpy()],
            })
            _, p_lr = logrank(d.time.to_numpy(), d.event.to_numpy(),
                              d.hi.to_numpy())
            split = cox(d, ["hi"])
            cont = cox(d, ["z"])
            rows.append({
                "コホート": cname, "終点": ep,
                "n": len(d), "イベント": int(d.event.sum()),
                "追跡中央値(日)": int(d.time.median()),
                "log-rank p": round(p_lr, 5),
                "HR(中央値分割)": round(split["hr"], 3),
                "p(中央値分割)": round(split["p"], 5),
                "HR(連続 1SD)": round(cont["hr"], 3),
                "p(連続 1SD)": round(cont["p"], 5),
                "備考": "",
            })
    grid = pd.DataFrame(rows)
    grid.to_csv(TABLES / "wp2_tcga_endpoint_grid.csv", index=False,
                encoding="utf-8")
    show = [c for c in ["コホート", "終点", "n", "イベント", "追跡中央値(日)",
                        "log-rank p", "HR(中央値分割)", "p(中央値分割)",
                        "HR(連続 1SD)", "p(連続 1SD)", "備考"] if c in grid.columns]
    print(grid[show].to_string(index=False))

    print("\n[4/4] 判定: meta-PCNA が予後因子として検出できる組み合わせ")
    live = grid[grid["p(連続 1SD)"].notna()] if "p(連続 1SD)" in grid.columns \
        else grid.iloc[0:0]
    hit = live[(live["p(連続 1SD)"] < 0.05) & (live["HR(連続 1SD)"] > 1)]
    if hit.empty:
        print("  × どの組み合わせでも meta-PCNA は予後因子にならない。")
        print("    → 「増殖で補正した後に残る関連」という目的変数を")
        print("       TCGA-BRCA では作れない。採点コホートを変える必要がある。")
    else:
        print(f"  ○ {len(hit)} 組で検出できた（HR > 1 かつ p < 0.05）:")
        for _, r in hit.sort_values("p(連続 1SD)").iterrows():
            print(f"    {r['コホート']} x {r['終点']}: "
                  f"n={r['n']} イベント={r['イベント']} "
                  f"HR={r['HR(連続 1SD)']:.2f} p={r['p(連続 1SD)']:.4g}")
        print(f"\n  評価した組み合わせ {len(live)} 通りのうち {len(hit)} 通り。"
              "全表を事前登録に添付する。")
    print(f"\n-> {TABLES / 'wp2_tcga_endpoint_grid.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
