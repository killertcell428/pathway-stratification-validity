"""WP2 の目的変数（アウトカム関連の強さ）を TCGA-BRCA で作れるかを確かめる。

docs/08-WP2再設計の論点.md 6 節の #4 と #5。

#4 生存データの欠測率とイベント数: イベント（死亡）が少なすぎると Cox が動かない。
    TCGA-BRCA は追跡が長く打ち切りが多いことで知られるので、実測が必要。

#5 Venet の HR 定義を TCGA で再現できるか: Venet ら (2011) は各シグネチャの
    第 1 主成分で検体を中央値分割し、log-rank で予後関連を見た。さらに meta-PCNA
    （増殖の代理）で補正すると、関連が残るシグネチャはごく少数になることを示した。
    この「補正前は多くが有意 → 補正後はほとんど消える」という形が TCGA で再現できれば、
    WP2 の目的変数が作れる。再現できなければ目的変数の定義を変えるしかない。

--------------------------------------------------------------------------
★ 事前登録との関係（重要）

ここで計算するのは **目的変数だけ** である。
WP2 の仮説は「内部整合性の対照超過 z（説明変数）が、meta-PCNA 補正後に残る
アウトカム関連（目的変数）を予測するか」。

**説明変数（対照超過 z）は TCGA では一切計算しない。** 目的変数を先に確定させるのは
測定手段の成立性確認であり、仮説の検定ではない。両方を見てから登録すれば、
事前登録が「結果を知ってから書いた計画」に堕ちる。この分離を守るために、
本スクリプトは遺伝子セットの内部整合性を計算する関数を一切呼ばない。

Venet の所見の再現は、彼らの既発表結果に対する陽性対照であって、
本研究の仮説を覗く行為ではない。
--------------------------------------------------------------------------

出力: results/tables/wp2_tcga_outcome.csv（シグネチャごとの HR と p）
      results/tables/wp2_tcga_outcome_summary.csv（#4/#5 の要約）
"""

from __future__ import annotations

import gzip
import sys

import numpy as np
import pandas as pd
import rdata
from scipy.stats import chi2, norm
from statsmodels.duration.hazard_regression import PHReg

from src.common import DATA, RAW, RESULTS

TABLES = RESULTS / "tables"
EXPR = DATA / "interim" / "tcga_brca" / "expr_tumor.parquet"
SURVIVAL = RAW / "TCGA-BRCA.survival.tsv.gz"
SIGNATURES = (RAW / "venet2011" / "ploscb-venet-dumont-detours"
              / "data" / "signatures.Rda")

MIN_COVERAGE = 0.6
MIN_GENES = 3


def load_expr() -> pd.DataFrame:
    if not EXPR.exists():
        raise SystemExit(f"先に tools.wp2_tcga_feasibility を実行して {EXPR} を作る")
    return pd.read_parquet(EXPR)


def load_survival(samples: list[str]) -> pd.DataFrame:
    """全生存（OS）を読み、発現行列の検体に合わせる（#4）。"""
    with gzip.open(SURVIVAL, "rt", encoding="utf-8") as f:
        s = pd.read_csv(f, sep="\t")
    s = s.set_index("sample")
    common = [c for c in samples if c in s.index]
    out = s.loc[common, ["OS", "OS.time"]].copy()
    out = out.rename(columns={"OS": "event", "OS.time": "time"})
    return out


def signature_pc1(X: pd.DataFrame, genes: list[str]) -> np.ndarray:
    """シグネチャ遺伝子の第 1 主成分（Venet の定義）。

    遺伝子ごとに標準化してから第 1 主成分を取る。符号は任意なので、
    シグネチャ遺伝子の平均発現と正に相関する向きに揃える（向きが検体ごとに
    反転すると HR の方向が意味を持たなくなる）。
    """
    M = X.loc[genes].to_numpy(dtype=np.float64)
    M = M - M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    M = M / sd
    # 検体数 < 遺伝子数 でも動くよう、共分散は検体側で取る
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    pc1 = vt[0] * s[0]
    if np.corrcoef(pc1, M.mean(axis=0))[0, 1] < 0:
        pc1 = -pc1
    return pc1


def logrank(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    """2 群の log-rank 検定。統計量と p 値を返す。

    群は 0/1。O-E 法で計算する（統計量 = (O1-E1)^2 / V）。
    """
    order = np.argsort(time, kind="stable")
    t, e, g = time[order], event[order], group[order]
    o1 = e1 = v = 0.0
    n1 = float((g == 1).sum())
    n_all = float(len(t))
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] == t[i]:
            j += 1
        d = float(e[i:j].sum())
        if d > 0 and n_all > 1:
            d1 = float(e[i:j][g[i:j] == 1].sum())
            exp1 = d * n1 / n_all
            o1 += d1
            e1 += exp1
            v += (d * (n1 / n_all) * (1 - n1 / n_all)
                  * (n_all - d) / (n_all - 1))
        # このタイム点にいた検体をリスク集合から外す
        n1 -= float((g[i:j] == 1).sum())
        n_all -= float(j - i)
        i = j
    if v <= 0:
        return np.nan, np.nan
    stat = (o1 - e1) ** 2 / v
    return float(stat), float(chi2.sf(stat, 1))


def cox(df: pd.DataFrame, covariates: list[str]) -> dict:
    """Cox 比例ハザードを当てる（statsmodels PHReg、Efron 法）。"""
    X = df[covariates].to_numpy(dtype=np.float64)
    m = PHReg(df.time.to_numpy(dtype=np.float64), X,
              status=df.event.to_numpy(dtype=np.float64), ties="efron")
    r = m.fit(disp=False)
    return {"coef": float(r.params[0]), "se": float(r.bse[0]),
            "hr": float(np.exp(r.params[0])),
            "p": float(2 * norm.sf(abs(r.params[0] / r.bse[0])))}


def load_target_signatures(present: set[str]) -> tuple[dict, list[str]]:
    """乳がん予後シグネチャ（主解析対象）と meta-PCNA を返す。"""
    obj = rdata.conversion.convert(rdata.parser.parse_file(SIGNATURES))

    def symbols(entry) -> list[str]:
        return sorted({str(s) for s in np.asarray(entry.get("symb")).ravel()
                       if str(s) not in ("", "nan", "None")})

    sigs = {}
    for name, entry in obj["cancer.signatures"].items():
        g = symbols(entry)
        hit = [x for x in g if x in present]
        if len(hit) >= MIN_GENES and len(hit) / len(g) >= MIN_COVERAGE:
            sigs[str(name)] = hit

    pcna_entry = obj["prolif.metagene"][list(obj["prolif.metagene"].keys())[0]]
    pcna = [x for x in symbols(pcna_entry) if x in present]
    return sigs, pcna


def main() -> int:
    print("[1/5] 発現行列と生存データを読む")
    X = load_expr()
    surv = load_survival(list(X.columns))
    print(f"  発現行列 {X.shape[0]:,} 遺伝子 x {X.shape[1]:,} 患者")
    print(f"  生存データが付く患者: {len(surv):,}")

    print("[2/5] #4 生存データの欠測とイベント数")
    n_expr = X.shape[1]
    missing = n_expr - len(surv)
    bad_time = int((surv.time.isna() | (surv.time <= 0)).sum())
    usable = surv[surv.time.notna() & (surv.time > 0) & surv.event.notna()].copy()
    events = int(usable.event.sum())
    print(f"  生存データなし: {missing} 名 / 追跡期間が欠測または 0 以下: {bad_time} 名")
    print(f"  解析可能: {len(usable):,} 名、うちイベント（死亡）{events} 件 "
          f"({events / len(usable):.1%})")
    print(f"  追跡期間（日）中央値 {usable.time.median():.0f} / "
          f"最大 {usable.time.max():.0f}")
    print(f"  打ち切り例の追跡期間 中央値 "
          f"{usable[usable.event == 0].time.median():.0f} 日")
    # 目安: Cox の共変量 1 つあたりイベント 10 件（EPV 10）。
    # 補正モデルは共変量 2 つ（シグネチャ + meta-PCNA）なので 20 件あれば動く。
    print(f"  共変量 2 つの Cox に必要なイベント数の目安（EPV 10）: 20 件 → "
          f"{'足りる' if events >= 20 else '足りない'}")

    print("[3/5] 対象シグネチャを取る")
    Xs = X[usable.index]
    sigs, pcna = load_target_signatures(set(X.index))
    print(f"  主解析対象 {len(sigs)} 件 / meta-PCNA {len(pcna)} 遺伝子")

    print("[4/5] #5 Venet の HR 定義を再現する（PC1 → 中央値分割 → log-rank）")
    pcna_pc1 = signature_pc1(Xs, pcna)
    base = usable.copy()
    base["pcna"] = (pcna_pc1 - pcna_pc1.mean()) / pcna_pc1.std()
    base["pcna_hi"] = (pcna_pc1 > np.median(pcna_pc1)).astype(int)

    st, p = logrank(base.time.to_numpy(), base.event.to_numpy(),
                    base.pcna_hi.to_numpy())
    c = cox(base, ["pcna_hi"])
    print(f"  meta-PCNA（陽性対照）: log-rank p = {p:.4g} / "
          f"HR = {c['hr']:.2f} (p = {c['p']:.4g})")

    rows = []
    for name, genes in sorted(sigs.items()):
        pc1 = signature_pc1(Xs, genes)
        d = base.copy()
        d["sig"] = (pc1 - pc1.mean()) / pc1.std()
        d["sig_hi"] = (pc1 > np.median(pc1)).astype(int)
        _, p_lr = logrank(d.time.to_numpy(), d.event.to_numpy(),
                          d.sig_hi.to_numpy())
        try:
            raw = cox(d, ["sig_hi"])
            adj = cox(d, ["sig_hi", "pcna"])
        except Exception as exc:                       # 収束しない場合を残す
            rows.append({"signature": name, "n_genes": len(genes),
                         "logrank_p": p_lr, "error": str(exc)[:60]})
            continue
        rows.append({
            "signature": name, "n_genes": len(genes),
            "logrank_p": round(p_lr, 6),
            "hr_raw": round(raw["hr"], 4), "p_raw": round(raw["p"], 6),
            "hr_adj": round(adj["hr"], 4), "p_adj": round(adj["p"], 6),
            "pcna_cor": round(float(np.corrcoef(pc1, pcna_pc1)[0, 1]), 4),
            # 目的変数の候補: 補正で関連がどれだけ残ったか
            "retention": round(abs(adj["coef"]) / abs(raw["coef"]), 4)
            if raw["coef"] != 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "wp2_tcga_outcome.csv", index=False, encoding="utf-8")

    print("[5/5] Venet の所見が再現するかを判定")
    ok = out[out.p_raw.notna()]
    n_raw = int((ok.p_raw < 0.05).sum())
    n_adj = int((ok.p_adj < 0.05).sum())
    print(f"  補正前に有意（p < 0.05）: {n_raw} / {len(ok)} 件 "
          f"({n_raw / len(ok):.0%})")
    print(f"  meta-PCNA 補正後も有意 : {n_adj} / {len(ok)} 件 "
          f"({n_adj / len(ok):.0%})")
    print(f"  シグネチャ PC1 と meta-PCNA PC1 の相関 中央値 "
          f"{ok.pcna_cor.median():.3f}（|r| 中央値 {ok.pcna_cor.abs().median():.3f}）")
    print(f"  目的変数候補 retention（|補正後 coef| / |補正前 coef|）: "
          f"中央値 {ok.retention.median():.3f} / "
          f"範囲 {ok.retention.min():.3f}〜{ok.retention.max():.3f}")

    verdict = []
    verdict.append(("イベント数が Cox に足りる", events >= 20))
    verdict.append(("meta-PCNA 自体が予後因子として有意", c["p"] < 0.05))
    verdict.append(("補正前に有意なシグネチャが複数ある", n_raw >= 5))
    verdict.append(("補正で有意数が減る（Venet の所見の向き）", n_adj < n_raw))
    verdict.append(("目的変数 retention に分散がある",
                    float(ok.retention.std()) > 0.05))
    print("\n  === 判定 ===")
    for label, passed in verdict:
        print(f"  [{'○' if passed else '×'}] {label}")

    summary = pd.DataFrame({
        "項目": [v[0] for v in verdict] + [
            "解析可能な患者数", "イベント数", "補正前に有意な件数",
            "補正後に有意な件数", "meta-PCNA の HR", "meta-PCNA の p"],
        "値": [("○" if v[1] else "×") for v in verdict] + [
            len(usable), events, n_raw, n_adj,
            round(c["hr"], 3), f"{c['p']:.4g}"],
    })
    summary.to_csv(TABLES / "wp2_tcga_outcome_summary.csv",
                   index=False, encoding="utf-8")
    print(f"\n-> {TABLES / 'wp2_tcga_outcome.csv'}")
    print(f"-> {TABLES / 'wp2_tcga_outcome_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
