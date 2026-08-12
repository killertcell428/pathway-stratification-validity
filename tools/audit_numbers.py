"""原稿に書かれた数値が、解析出力の CSV と一致するかを機械照合する。

原稿は改稿のたびに数値を手で転記している。転記ミスと、解析をやり直したのに
原稿を直し忘れた箇所は、査読で最も致命的な指摘になる。投稿前に毎回これを通す。

やり方は「CSV から真の値を計算し、その値が原稿本文に文字列として出てくるか」を
見る方式にした。原稿側の数値をスクリプトに書き写すと、その転記自体が新しい
ミス源になるため。

使い方:
  pixi run audit                      manuscript/02-投稿原稿_日本語_v1.md を照合
  pixi run python -m tools.audit_numbers path/to/manuscript.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
# 和文と英文の両方を既定で照合する。英訳の過程で数値がずれるのが最大の危険なので、
# 片方だけ通しても意味がない。
DEFAULT_MDS = [
    ROOT / "manuscript" / "02-投稿原稿_日本語_v1.md",
    ROOT / "manuscript" / "04-preprint-v1-en.md",
]

# 廃止した値。原稿に残っていたら失敗にする。
#
# 「正しい値が本文のどこかにある」ことを確認するだけでは、改稿で直し忘れた別の節に
# 古い値が残っている状態を検出できない。実際に GSE81046 の数値を作り直したとき、
# 3.1 節は直ったのに要旨と結論に旧値が残った。それを機械で捕まえるための一覧。
RETIRED = [
    ("2,424", "GSE81046 の評価セット数。分位正規化された古い行列での値（正: 2,514）"),
    ("45.2%", "GSE81046 の内部整合性合格率。同上（正: 37.7%）"),
    ("48.0%", "GSE81046 の「条件効果のみ」。同上（正: 55.7%）"),
    ("88.1%", "GSE81046 の条件効果。同上（正: 88.3%）"),
    ("4.199", "GTEx 血球組成の単純合計。加重に統一済み（正: 1.510）"),
    ("1.840", "GTEx 技術の単純合計。同上（正: 0.594）"),
    ("−0.209", "TMM log-CPM の検出率との相関。実際は相関は消えない（正: 0.585）"),
    ("44.3%", "log2(TPM+1) の第 1 主成分寄与率。遺伝子集合を固定して再測定（正: 45.3%）"),
    ("−0.916", "log2(TPM+1) の検出率との相関。同上（正: 0.600）"),
    ("0.829", "アレイと RNA-seq のファミリー順位一致。実際は保存されない（正: 0.00）"),
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name)


def weighted_total(df: pd.DataFrame, kind: str, excess_col: str) -> float:
    """主成分の寄与率で重み付けした超過説明率の合計。

    単純和にすると寄与率 3% の PC5 が 29% の PC1 と同じ重みになり、
    主成分構造の異なるコホート間で比較できなくなる。2 コホートで
    定義を揃えるため、集計はここに一本化する。
    """
    w = df.drop_duplicates("pc").set_index("pc").var_ratio.sum()
    s = df[df.kind == kind]
    return float((s[excess_col] * s.var_ratio).sum() / w)


def checks() -> list[tuple[str, float, int, bool]]:
    """(ラベル, 計算値, 小数点以下桁数, 原稿に必須か) の一覧を返す。

    必須でないものは、解析としては出しているが原稿では触れていない値。
    数値そのものを載せる予定がないので、見つからなくても失敗にしない。
    ただし計算はしておく。将来この値を原稿に書くとき、書き間違えれば
    「参考値」欄との突き合わせで気づける。
    """
    out: list[tuple[str, float, int, bool]] = []

    def add(label: str, value: float, nd: int = 3, required: bool = True) -> None:
        out.append((label, float(value), nd, required))

    # --- 主コホート: 条件効果と内部整合性の乖離 ---
    m = load("gene_set_metrics.csv")
    add("条件効果が有意なセットの割合(%)", 100 * m.delta_q.lt(0.05).mean(), 1)
    both = m.null_q.lt(0.05) & m.var_null_q.lt(0.05)
    add("2種の対照を両方通る割合(%)", 100 * both.mean(), 1)
    add("発現量そろえ対照のみを通る割合(%)", 100 * m.null_q.lt(0.05).mean(), 1)
    add("評価したセット数", len(m), 0)
    add("条件効果の Cohen の d の中央値", m.cohens_d.abs().median(), 3, required=False)

    # --- ファミリー別 ---
    # ファミリー別の合格率は「2 種の対照の両方を通る」厳しい側で数える。
    # family_summary.csv の ic_above_varnull_frac は分散そろえ対照だけの
    # 緩い側なので、原稿の値と定義が違う。ここで両方から数え直す。
    fam = load("family_summary.csv")
    both_by_family = m.assign(both=both).groupby("family").both.mean()
    for _, r in fam.iterrows():
        add(f"[{r.family}] 内部整合性の中央値", r.ic_median, 3)
        add(f"[{r.family}] 対照の中央値", r.ic_null_median, 3,
            required=r.family != "anchor")
        add(f"[{r.family}] 2対照を通る割合(%)", 100 * both_by_family[r.family], 1)

    # --- チップ帰属（主コホート）---
    b = load("batch_check.csv")
    r = b[(b.pc == "PC1") & (b.factor == "chip")].iloc[0]
    add("PC1 のチップ説明率", r.r2, 3)
    add("PC1 のチップ偶然水準", r.r2_chance, 3)
    add("PC1 のチップ超過分", r.r2_excess, 3)
    add("主コホート PC1 の寄与率(%)", 100 * r.var_ratio, 1)

    # --- GSE35846（マイクロアレイ全血）の軸分解 ---
    ta = load("technical_axes.csv")
    for kind, jp in [("composition", "細胞組成"), ("technical", "技術"), ("biological", "人口統計")]:
        add(f"GSE35846 {jp}（加重）", weighted_total(ta, kind, "r2_excess"), 3)

    # --- GTEx v8（RNA-seq 全血）の軸分解 ---
    g = load("gtex_blood/pc_attribution.csv")
    for kind in ["血球組成", "技術", "生物学"]:
        add(f"GTEx {kind}（加重）", weighted_total(g, kind, "excess"), 3)
    w = g.drop_duplicates("pc").set_index("pc").var_ratio.sum()
    per = (g.excess * g.var_ratio / w).groupby(g.factor).sum()
    for f in per.index:
        add(f"GTEx 要因 {f}（加重）", per[f], 3)
    pc1 = g[g.pc == "PC1"]
    for f in ["組成:Neutrophils", "組成:Platelets", "SMTSISCH", "DTHHRDY"]:
        row = pc1[pc1.factor == f]
        if len(row):
            add(f"GTEx PC1 {f} の R²", row.iloc[0].r2, 3)
    add("GTEx PC1 の寄与率(%)", 100 * float(pc1.var_ratio.iloc[0]), 1)

    # --- GSE81046（RNA-seq マクロファージ）の祖先集団 ---
    a = pd.read_csv(T / "gse81046" / "ancestry_attribution.csv")
    p1 = a[a.pc == "PC1"].iloc[0]
    add("GSE81046 PC1 の祖先説明率", p1.r2, 3)
    add("GSE81046 PC1 の祖先偶然水準", p1.r2_chance, 3)
    add("GSE81046 PC1 の祖先超過分", p1.excess, 3)
    add("GSE81046 PC1 の寄与率(%)", 100 * p1.var_ratio, 1)

    # --- 反復測定信頼性 ---
    rt = load("retest_metrics.csv")
    add("ICC が対照を上回るセットの割合(%)", 100 * rt.icc_q.lt(0.05).mean(), 1)
    add("内部整合性が対照を上回る割合(再測定コホート)(%)", 100 * rt.ic_q.lt(0.05).mean(), 1,
        required=False)

    # --- 表現型 ---
    ph = load("phenotype_metrics.csv")
    add("表現型相関が対照を上回る割合(%)", 100 * ph.abs_rho_q.lt(0.05).mean(), 1)

    # --- GSE81046（RNA-seq 再現）の交差表 ---
    g8 = pd.read_csv(T / "gse81046" / "gene_set_metrics.csv")
    cond8 = g8.delta_q < 0.05
    both8 = (g8.null_q < 0.05) & (g8.var_null_q < 0.05)
    add("GSE81046 評価セット数", len(g8), 0)
    add("GSE81046 条件効果あり(%)", 100 * cond8.mean(), 1)
    add("GSE81046 内部整合性あり(%)", 100 * both8.mean(), 1)
    add("GSE81046 条件効果のみ(%)", 100 * (cond8 & ~both8).mean(), 1)

    # 細胞種マーカーの上位。存在しない細胞種が上位に来ることが主張の核なので数値も照合する
    for _, r in g8[g8.family == "celltype"].nlargest(4, "internal_consistency").iterrows():
        add(f"GSE81046 {r.set.split('|')[-1]} の整合性", r.internal_consistency, 3)

    # 雑音軸除去による整合性の変化率。原稿が実際に引用しているファミリーだけを
    # 必須にする（主コホートは 3 ファミリーのみ本文に出る）。
    cited = {"主コホート": {"pathway", "celltype", "data_derived"},
             "GSE81046": {"pathway", "signature", "regulon", "celltype", "data_derived"}}
    for ns, tag in [("", "主コホート"), ("gse81046/", "GSE81046")]:
        af = pd.read_csv(T / f"{ns}attribution_by_family.csv").set_index("family")
        for fam in ["pathway", "signature", "regulon", "celltype", "data_derived"]:
            if fam not in af.index:
                continue
            r = af.loc[fam]
            add(f"{tag} {fam} 雑音軸除去の変化(%)",
                100 * (r.ic_all - r.ic_none) / r.ic_none, 0,
                required=fam in cited[tag])

    # ファミリー順位が測定方式をまたいで保存されるかは、原稿が明示的に否定した
    # 主張なので数値を固定して監視する（以前は保存されると書いていた）
    fa = load("family_summary.csv").set_index("family").ic_median
    fb = pd.read_csv(T / "gse81046" / "family_summary.csv").set_index("family").ic_median
    idx = list(fa.index.intersection(fb.index))
    add("アレイと RNA-seq のファミリー順位一致", spearmanr(fa[idx], fb[idx]).statistic, 2)
    no_anchor = [f for f in idx if f != "anchor"]
    add("同（参照アンカーを除く）", spearmanr(fa[no_anchor], fb[no_anchor]).statistic, 2)

    # --- 定量・正規化の比較（表 4）---
    nc = pd.read_csv(T / "gse81046" / "normalization_comparison.csv", index_col=0)
    for name, r in nc.iterrows():
        tag = name.split("（")[0]
        add(f"[{tag}] 第1主成分の寄与率(%)", r["第1主成分の寄与率"], 1)
        add(f"[{tag}] 平均発現量との相関", r["平均発現量との相関"], 3)
        add(f"[{tag}] 検出率との相関", abs(r["検出率との相関"]), 3)
        add(f"[{tag}] 対照の整合性", r["ランダム対照の整合性(中央値)"], 3)

    return out


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    targets = [Path(a) for a in args] if args else DEFAULT_MDS
    rc = 0
    checks_cache = checks()
    for md in targets:
        if not md.exists():
            print(f"★ 見つからない: {md}")
            rc = 1
            continue
        rc |= audit_one(md, checks_cache)
        print()
    return rc


def audit_one(md: Path, rows: list[tuple[str, float, int, bool]]) -> int:
    text = md.read_text(encoding="utf-8-sig")
    # 桁区切り、全角、和文で使う負号（U+2212）の揺れを吸収してから探す
    flat = (text.replace(",", "").replace("，", "")
                .replace("−", "-").replace("－", "-"))

    ok, ng, ref = [], [], []
    for label, value, nd, required in rows:
        s = f"{value:.{nd}f}" if nd else f"{value:.0f}"
        hit = s in flat
        if not required:
            ref.append((label, s, hit))
        elif hit:
            ok.append((label, s))
        else:
            ng.append((label, s))

    stale = [(v, why) for v, why in RETIRED if v.replace("−", "-") in flat]

    print(f"=== 数値照合: {md.name} ===")
    print(f"必須 {len(ok)} / {len(ok) + len(ng)} 一致 ／ 廃止値の残存 {len(stale)} 件")
    if stale:
        print("\n【廃止値が残っている】改稿で直し忘れた節がある")
        for v, why in stale:
            print(f"  {v:10s} {why}")
    if ng:
        print("\n【一致しない】原稿の値が解析出力と食い違っている")
        for label, s in ng:
            print(f"  {label:44s} 解析出力 {s}")
    if ref:
        print("\n【参考値】原稿では数値に触れていない項目")
        for label, s, hit in ref:
            print(f"  {label:44s} {s}{'  （原稿にも出現）' if hit else ''}")
    bad = bool(ng or stale)
    print("\n" + ("要修正" if bad else "食い違いなし"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
