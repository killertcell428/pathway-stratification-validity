"""WP2 の目的変数を Venet 自身のコホート（Loi / NKI）で作れるかを確かめる。

docs/08-WP2再設計の論点.md #5 の再挑戦。

■ TCGA-BRCA で目的変数が作れなかった理由（tools/wp2_tcga_endpoint_grid.py）
終点 4 種 x コホート 5 種 = 20 通りすべてで、meta-PCNA（増殖の代理）が
予後因子として検出できなかった（有意は 1 通り p = 0.048、α = 0.05 で 20 回
検定すれば偶然 1 回出る水準）。追跡期間の中央値が 2.4 年しかなく、
ER 陽性に絞ると逆に悪化した。増殖で補正するという目的変数の前提が立たない。

■ 識別子の問題が解けた
Venet の発現行列は **Entrez Gene ID** で索引されていた（1, 2, 9, 10, ...）。
以前「48 件の被覆率が中央値 0%」と判定したのは、記号と Entrez ID を
直接突き合わせていたためで、記号の世代差ではなかった。
NCBI gene_info（正式名 + 別名）で橋渡しすると Loi の被覆率は中央値 98.5%、
meta-PCNA は 98.4% になる。

■ したがって変数をコホートで分ける
| 変数 | コホート | 理由 |
|---|---|---|
| 説明変数: 内部整合性の対照超過 z | TCGA-BRCA | 対照の発現量マッチには絶対発現量が要る。log 比では作れない |
| 目的変数: 増殖補正後に残る予後関連 | Loi / NKI | 増殖が予後因子として効く（追跡 7 年）。HR 計算は log 比でも成立する |

log 比が塞いだのは対照マッチだけであって、主成分と Cox は log 比でも計算できる。
両者を同じコホートで揃える必要はない。

■ 事前登録との関係
ここで計算するのは目的変数だけ。説明変数（対照超過 z）は登録前に計算しない。

出力: results/tables/wp2_venet_outcome.csv（コホート x シグネチャの HR）
      results/tables/wp2_venet_outcome_summary.csv（判定）
"""

from __future__ import annotations

import gzip
import sys

import numpy as np
import pandas as pd
import rdata

from src.common import RAW, RESULTS
from tools.wp2_tcga_outcome import cox, logrank, signature_pc1

TABLES = RESULTS / "tables"
BUNDLE = RAW / "venet2011" / "ploscb-venet-dumont-detours" / "data"
GENE_INFO = RAW / "Homo_sapiens.gene_info.gz"

COHORTS = {
    "Loi OS": "expression-data-loi-os.Rda",
    "Loi RFS": "expression-data-loi-rfs.Rda",
    "NKI RFS": "expression-data-nki-rfs.Rda",
}
MIN_COVERAGE = 0.6
MIN_GENES = 3


def symbol_to_entrez() -> dict[str, int]:
    """記号 → Entrez ID。正式名を優先し、足りない分を別名で補う。

    Venet は 2007 年の HUGO 記号に揃えているため、正式名だけでは当たらない
    記号が残る。別名（Synonyms）を後から入れることで、正式名が別遺伝子と
    衝突する場合に正式名側が勝つようにしている。
    """
    with gzip.open(GENE_INFO, "rt", encoding="utf-8") as f:
        gi = pd.read_csv(f, sep="\t", low_memory=False,
                         usecols=["GeneID", "Symbol", "Synonyms"])
    m: dict[str, int] = {}
    for gid, sym, _ in gi.itertuples(index=False):
        m.setdefault(str(sym), int(gid))
    for gid, _, syn in gi.itertuples(index=False):
        if isinstance(syn, str) and syn != "-":
            for s in syn.split("|"):
                m.setdefault(s, int(gid))
    return m


def load_signatures(mapping: dict[str, int]) -> tuple[dict[str, list[int]], list[int]]:
    obj = rdata.conversion.convert(rdata.parser.parse_file(BUNDLE / "signatures.Rda"))

    def entrez(entry) -> tuple[int, list[int]]:
        syms = {str(s) for s in np.asarray(entry.get("symb")).ravel()
                if str(s) not in ("", "nan", "None")}
        return len(syms), sorted({mapping[s] for s in syms if s in mapping})

    sigs = {}
    for name, entry in obj["cancer.signatures"].items():
        n_sym, ids = entrez(entry)
        sigs[str(name)] = (n_sym, ids)
    pcna_entry = obj["prolif.metagene"][list(obj["prolif.metagene"].keys())[0]]
    return sigs, entrez(pcna_entry)[1]


def load_cohort(fname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = rdata.conversion.convert(rdata.parser.parse_file(BUNDLE / fname))["study"]
    ids = [int(float(x)) for x in np.asarray(st["genes"]).ravel()]
    X = pd.DataFrame(np.asarray(st["data"], dtype=np.float64), index=ids)
    sv = np.asarray(st["survival"], dtype=np.float64)
    d = pd.DataFrame({"time": sv[:, 0], "event": sv[:, 1]})
    keep = (d.time.notna() & (d.time > 0) & d.event.notna()).to_numpy()
    return X.loc[:, keep], d[keep].reset_index(drop=True)


def main() -> int:
    print("[1/4] 記号 → Entrez ID の対応を作る")
    mapping = symbol_to_entrez()
    sigs, pcna = load_signatures(mapping)
    print(f"  対応 {len(mapping):,} 件 / シグネチャ {len(sigs)} 件 / "
          f"meta-PCNA {len(pcna)} 遺伝子")

    rows, summaries = [], []
    for label, fname in COHORTS.items():
        print(f"\n[2/4] {label} を読む")
        X, d = load_cohort(fname)
        print(f"  {X.shape[0]:,} 遺伝子 x {X.shape[1]} 検体 / "
              f"イベント {int(d.event.sum())} 件 ({d.event.mean():.0%}) / "
              f"追跡中央値 {d.time.median():.1f} 年")

        pc_genes = [g for g in pcna if g in X.index]
        pcna_pc1 = signature_pc1(X, pc_genes)
        d = d.copy()
        d["pcna"] = (pcna_pc1 - pcna_pc1.mean()) / pcna_pc1.std()
        d["pcna_hi"] = (pcna_pc1 > np.median(pcna_pc1)).astype(int)
        _, p_lr = logrank(d.time.to_numpy(), d.event.to_numpy(),
                          d.pcna_hi.to_numpy())
        pc_cox = cox(d, ["pcna_hi"])
        print(f"  陽性対照 meta-PCNA（{len(pc_genes)} 遺伝子）: "
              f"HR {pc_cox['hr']:.2f} / log-rank p {p_lr:.3g}")

        print(f"[3/4] {label} で 48 件の HR を計算する")
        n_eval = 0
        for name, (n_sym, ids) in sorted(sigs.items()):
            hit = [g for g in ids if g in X.index]
            cov = len(hit) / n_sym if n_sym else 0.0
            rec = {"cohort": label, "signature": name, "n_symbols": n_sym,
                   "n_present": len(hit), "coverage": round(cov, 4)}
            if len(hit) < MIN_GENES or cov < MIN_COVERAGE:
                rec["excluded"] = ("被覆率 <60%" if cov < MIN_COVERAGE
                                   else f"検出 {len(hit)} 遺伝子")
                rows.append(rec)
                continue
            pc1 = signature_pc1(X, hit)
            dd = d.copy()
            dd["z"] = (pc1 - pc1.mean()) / pc1.std()
            dd["hi"] = (pc1 > np.median(pc1)).astype(int)
            _, p_sig = logrank(dd.time.to_numpy(), dd.event.to_numpy(),
                               dd.hi.to_numpy())
            raw, adj = cox(dd, ["hi"]), cox(dd, ["hi", "pcna"])
            rec.update({
                "logrank_p": round(p_sig, 6),
                "hr_raw": round(raw["hr"], 4), "p_raw": round(raw["p"], 6),
                "hr_adj": round(adj["hr"], 4), "p_adj": round(adj["p"], 6),
                "pcna_cor": round(float(np.corrcoef(pc1, pcna_pc1)[0, 1]), 4),
                "retention": round(abs(adj["coef"]) / abs(raw["coef"]), 4)
                if raw["coef"] != 0 else np.nan,
                "excluded": "",
            })
            rows.append(rec)
            n_eval += 1

        ev = pd.DataFrame([r for r in rows if r["cohort"] == label
                           and not r.get("excluded")])
        n_raw = int((ev.p_raw < 0.05).sum())
        n_adj = int((ev.p_adj < 0.05).sum())
        print(f"  評価 {n_eval} 件 → 補正前に有意 {n_raw} 件 ({n_raw / n_eval:.0%}) / "
              f"meta-PCNA 補正後も有意 {n_adj} 件 ({n_adj / n_eval:.0%})")
        print(f"  retention（|補正後 coef| / |補正前 coef|）中央値 "
              f"{ev.retention.median():.3f} / 四分位 "
              f"{ev.retention.quantile(0.25):.3f}〜{ev.retention.quantile(0.75):.3f}")
        summaries.append({
            "コホート": label, "n": X.shape[1], "イベント": int(d.event.sum()),
            "追跡中央値(年)": round(float(d.time.median()), 1),
            "meta-PCNA HR": round(pc_cox["hr"], 3),
            "meta-PCNA p": f"{pc_cox['p']:.3g}",
            "評価件数": n_eval, "補正前に有意": n_raw, "補正後に有意": n_adj,
            "retention 中央値": round(float(ev.retention.median()), 3),
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "wp2_venet_outcome.csv", index=False, encoding="utf-8")
    s = pd.DataFrame(summaries)
    s.to_csv(TABLES / "wp2_venet_outcome_summary.csv", index=False,
             encoding="utf-8")

    print("\n[4/4] 判定")
    print(s.to_string(index=False))
    ev = out[out.excluded == ""]
    checks = [
        ("全コホートで meta-PCNA が予後因子（p < 0.05）",
         all(float(x["meta-PCNA p"]) < 0.05 for x in summaries)),
        ("補正前に有意なシグネチャが 10 件以上ある（全コホート）",
         all(x["補正前に有意"] >= 10 for x in summaries)),
        ("補正で有意数が減る（Venet の所見の向き、全コホート）",
         all(x["補正後に有意"] < x["補正前に有意"] for x in summaries)),
        ("retention の中央値が 1 未満（補正で関連が弱まる）",
         all(x["retention 中央値"] < 1.0 for x in summaries)),
        ("retention に分散がある（目的変数として使える）",
         float(ev.retention.std()) > 0.05),
    ]
    for label, passed in checks:
        print(f"  [{'○' if passed else '×'}] {label}")

    print("\n  === コホートをまたいだ目的変数の一致（同じシグネチャで再現するか）===")
    piv = ev.pivot_table(index="signature", columns="cohort", values="retention")
    for a in COHORTS:
        for b in COHORTS:
            if a < b and a in piv.columns and b in piv.columns:
                both = piv[[a, b]].dropna()
                r = both[a].corr(both[b], method="spearman")
                print(f"    {a} vs {b}: n={len(both)} Spearman rho={r:.3f}")

    print(f"\n-> {TABLES / 'wp2_venet_outcome.csv'}")
    print(f"-> {TABLES / 'wp2_venet_outcome_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
