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

import json
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
    ("88.9%", "表 5b の ICC 超過。判定基準が混在していた（正: 72.2%）"),
    ("16 / 18", "同上（正: 13 / 18）"),
    ("16 of 18", "同上（正: 13 of 18）"),
    ("3 / 18（16.7%）", "表 5b の共変動超過・除去後（正: 1 / 18（5.6%））"),
    ("3 / 18 (16.7%)", "同上（正: 1 / 18 (5.6%)）"),
    ("0.577（0.389）", "表 5b の ICC 対照。対照 10,000 個前の引き（正: 0.577（0.385））"),
    ("0.577 (0.389)", "同上（正: 0.577 (0.385)）"),
    ("2,424", "GSE81046 の評価セット数。分位正規化された古い行列での値（正: 2,514）"),
    ("45.2%", "GSE81046 の内部整合性合格率。分位正規化の古い行列（正: 27.4%）"),
    ("88.1%", "GSE81046 の条件効果。同上（正: 88.3%）"),
    # --- 対照 20 個 + 正規近似で出していた値（対照 10,000 個 + 経験 p に移行）---
    # 平均ペア Spearman の帰無分布は右に歪むので、正規近似は右裾で甘い側に外れていた。
    # 合格率はどれも下がり、乖離はどれも上がる。方向が一貫しているので、
    # 1 箇所でも旧値が残ると論旨が矛盾する。
    ("11.1%", "主コホートの適格率。対照 20 個 + 正規近似の値（正: 4.6%）"),
    ("80.9%", "主コホートの乖離。同上（正: 86.7%）"),
    ("14.3%", "主コホートの単一対照合格率。同上（正: 4.8%）"),
    ("243 ", "両対照を通ったセット数。同上（正: 100）"),
    ("37.7%", "GSE81046 の適格率。同上（正: 27.4%）"),
    ("55.7%", "GSE81046 の乖離。同上（正: 64.4%）"),
    ("43.1%", "GSE81046 の単一対照合格率。同上（正: 28.8%）"),
    ("13.1%", "チップ除去前の合格率。同上（正: 5.0%）"),
    ("30.5%", "チップ除去後の合格率。同上（正: 21.4%）"),
    ("0.416", "反復測定 ICC の対照平均。同上（正: 0.422）"),
    ("67.6%", "細胞種マーカーの適格率（表 2）。同上（正: 32.4%）"),
    ("98.4%", "データ由来モジュールの適格率（表 2）。同上（正: 88.5%）"),
    ("22.9%", "発現シグネチャの適格率（表 2）。同上（正: 17.1%）"),
    ("10.3%", "反応経路の適格率（表 2）。同上（正: 2.0%）"),
    ("4.199", "GTEx 血球組成の単純合計。加重に統一済み（正: 1.510）"),
    ("1.840", "GTEx 技術の単純合計。同上（正: 0.594）"),
    ("−0.209", "TMM log-CPM の検出率との相関。実際は相関は消えない（正: 0.585）"),
    ("44.3%", "log2(TPM+1) の第 1 主成分寄与率。遺伝子集合を固定して再測定（正: 45.3%）"),
    ("−0.916", "log2(TPM+1) の検出率との相関。同上（正: 0.600）"),
    ("0.829", "アレイと RNA-seq のファミリー順位一致。実際は保存されない（正: 0.00）"),
    ("4,618", "系統的検索の取得件数。上限 1,000 で 19.2% が未取得だった（正: 5,715）"),

    ("read in full", "英文で full-text 精読を意味してしまう。実際は抄録精読"),
    ("agrees with Tables 3 and 3b in all three cohorts",
     "表 3c は 9 項目中 4 項目しか一致しない（対照の引き直しによる差）"),
    ("raises the control by the same amount",
     "チップ除去で対照は −63%、注釈側は −35%。同じ量ではない"),
    ("対照側も同じだけ上げる", "同上（和文）"),
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

    # --- 判定基準をそろえた場合（条件効果も同じ対照に対する超過として測る）---
    # 条件効果は「条件間の差がゼロか」、内部整合性は「ランダムセットより高いか」を問う。
    # 帰無仮説が違うものを交差させると、割合の差の一部は判定基準の非対称性から出る。
    # 条件効果を同じ対照 10,000 組に対する超過として測り直した値を登録する。
    _cond_pass = m.cond_null_q.lt(0.05)
    add("条件効果が対照を上回る割合(%)", 100 * _cond_pass.mean(), 1)
    add("条件効果が対照を上回るセット数", int(_cond_pass.sum()), 0)
    add("対照基準の交差表: 条件効果のみの割合(%)", 100 * (_cond_pass & ~both).mean(), 1)
    add("対照基準の交差表: 条件効果のみのセット数", int((_cond_pass & ~both).sum()), 0)
    add("対照基準の交差表: どちらも通らない割合(%)", 100 * (~_cond_pass & ~both).mean(), 1)
    add("対照の条件効果 |d| の中央値", m.cond_null_mean.abs().median(), 3)
    add("条件効果の対照超過 |d| の中央値",
        (m.cohens_d.abs() - m.cond_null_mean.abs()).median(), 3)

    # --- 同方向性も同じ対照に対する超過として測る ---
    # 摂動が転写全体を一方向に押すなら、ランダムに集めた遺伝子でも同方向性は高く出る。
    # 二項検定の帰無（同方向の確率 0.5）は、対照の実測（0.624）より緩い側にずれている。
    _dir_pass = m.dir_null_q.lt(0.05)
    add("同方向性が対照を上回る割合(%)", 100 * _dir_pass.mean(), 1)
    add("同方向性の観測 中央値", m.frac_same_direction.median(), 3)
    add("同方向性の対照 中央値", m.dir_null_mean.median(), 3)
    add("同方向性の対照超過 中央値", (m.frac_same_direction - m.dir_null_mean).median(), 3)
    for fam, lab in (("complex", "CORUM 複合体"), ("regulon", "転写因子レギュロン")):
        _f = m[m.family == fam]
        add(f"[{lab}] 同方向性の対照超過 中央値",
            (_f.frac_same_direction - _f.dir_null_mean).median(), 3)

    # --- ファミリー別の内部整合性の対照超過 ---
    # 注釈の供給源によって、スコアを反映型の尺度として読むことの妥当性が違う。
    # 反映型の前提が最も成り立つ発現シグネチャでも超過が小さいことを示すために登録する。
    # --- 帰無モデルの感度分析（対照を注釈遺伝子プールから引く）---
    # 対照が「注釈されている遺伝子である」性質を保存していなければ、超過は注釈の質ではなく
    # プールの違いを測っていることになる。判定がどれだけ動くかを登録する。
    # --- 遺伝子セットの重複を考慮した有効単位数 ---
    # 2,195 セットは独立ではない。Jaccard 類似度でまとめた単位で数え直した値を登録する。
    _rd = load("redundancy_analysis.csv").set_index("jaccard_threshold")
    for th in (0.5, 0.25):
        r = _rd.loc[th]
        add(f"[Jaccard {th}] クラスタ数", int(r.n_clusters), 0)
        add(f"[Jaccard {th}] クラスタ単位の合格率(%) any", float(r.pass_any_pct), 1)
        add(f"[Jaccard {th}] クラスタ単位の合格率(%) rep", float(r.pass_rep_pct), 1)
        add(f"[Jaccard {th}] 代表集合で BH を引き直した合格率(%)", float(r.pass_redo_pct), 1)

    _nm = load("null_model_sensitivity.csv")
    add("注釈遺伝子プール対照での合格率(%)", 100 * _nm.pass_annot.mean(), 1)
    add("全遺伝子プール対照での合格率(%)", 100 * _nm.pass_all.mean(), 1)
    add("2 つの対照で判定が変わったセット数", int((_nm.pass_all != _nm.pass_annot).sum()), 0)
    add("注釈遺伝子プール対照の水準 中央値", _nm.null_annot_mean.median(), 4)
    add("全遺伝子プール対照の水準 中央値", _nm.null_all_mean.median(), 4)
    # PC1 寄与をそろえた対照（発現量十分位 x PC1 五分位の二重マッチ）
    add("PC1 そろえ対照での合格率(%)", 100 * _nm.pass_pc1.mean(), 1)
    add("PC1 そろえ対照の水準 中央値", _nm.null_pc1_mean.median(), 4)
    add("PC1 そろえで判定が変わったセット数", int((_nm.pass_all != _nm.pass_pc1).sum()), 0)
    add("注釈遺伝子プール対照での合格セット数", int(_nm.pass_annot.sum()), 0)
    add("全遺伝子プール対照での合格セット数", int(_nm.pass_all.sum()), 0)
    add("PC1 そろえ対照での合格セット数", int(_nm.pass_pc1.sum()), 0)

    # --- コホートの規模（Methods の表 M1）---
    # 検証側だけでなく総数も表に出すので、分割ファイルから数えて照合する。
    _split = json.loads((Path(__file__).resolve().parents[1] / "data" / "metadata" / "donor_split.json").read_text(encoding="utf-8"))
    add("主コホートの総ドナー数", sum(len(v) for v in _split.values() if isinstance(v, list)), 0)

    # --- 引用追跡（語の検索が取りこぼした研究がないかの第 2 経路）---
    # 不在の主張は経路を 1 本しか持たないと弱い。件数を原稿と機械照合する。
    _cc = load("citation_chase_summary.csv").set_index("stage")["n"]
    for k in ("引用関係として取得したレコード（延べ）", "一意のレコード",
              "うち補遺 S1 の検索で既出", "検索で出ていない新規レコード",
              "うち抄録が取得できず判定不能", "第 1 段の規則を通過（個別に読む対象）",
              "選定基準をすべて満たした研究", "基準を満たさないが近い研究（near-miss）"):
        add(f"[引用追跡] {k}", int(_cc[k]), 0)

    _ic_ex = (m.internal_consistency - m.null_mean).groupby(m.family).median()
    for key, lab in (("signature", "発現シグネチャ"), ("celltype", "細胞種マーカー"),
                     ("pathway", "反応経路"), ("complex", "CORUM 複合体"),
                     ("regulon", "転写因子レギュロン")):
        if key in _ic_ex.index:
            add(f"[{lab}] 内部整合性の対照超過 中央値", float(_ic_ex[key]), 4)

    # --- 図 7: 2 採血の重なり（キャプションが引く排他カウント）---
    # 図中のラベルは「その採血でのみ通った数」なので、通った総数とは違う。
    # 生成コード fig7_phenotype_rerandomized と同じ定義で数え直す。
    _ph = load("phenotype_metrics.csv")
    _z0 = int((_ph.abs_rho_z > 2).sum())
    _z7 = int((_ph["abs_rho_z_day-7"] > 2).sum())
    _both = int(((_ph.abs_rho_z > 2) & (_ph["abs_rho_z_day-7"] > 2)).sum())
    add("[図7] day 0 のみで通った数", _z0 - _both, 0, required=False)
    add("[図7] day -7 のみで通った数", _z7 - _both, 0, required=False)
    # 2 時点の一致度。day 0 で選んで day -7 を見る比較は平均への回帰を含むが、
    # 全セットの順位相関は選択を経ないのでその影響を受けない。
    add("[表現型] 2 時点の対照超過 z の順位相関",
        float(_ph[["abs_rho_z", "abs_rho_z_day-7"]].corr(method="spearman").iloc[0, 1]), 3)
    add("[表現型] 年齢・性別を調整した day 0 の |rho| 中央値",
        float(_ph.rho_day0_adj.abs().median()), 3)
    add("[図7] 両方で通った数", _both, 0, required=False)

    # --- 図 5: コホート横断の PC1 帰属（キャプションが引く値）---
    # 図の生成コード fig5_cross_cohort_attribution と同じ表・同じ行から取る。
    ta = load("technical_axes.csv")
    ta = ta[ta.pc == "PC1"]
    add("[図5 全血] 民族の超過説明率",
        float(ta[ta.factor == "ethnicity"].r2_excess.iloc[0]), 3, required=False)
    gtx = pd.read_csv(T / "gtex_blood" / "pc_attribution.csv")
    gtx = gtx[gtx.pc == "PC1"]
    add("[図5 GTEx] 虚血時間の超過説明率",
        float(gtx[gtx.factor == "SMTSISCH"].excess.iloc[0]), 3, required=False)
    add("[図5 GTEx] 年齢の超過説明率",
        float(gtx[gtx.factor == "AGE"].excess.iloc[0]), 3, required=False)

    # --- 図 6: 対照を上回らなかったセット数（キャプションの内訳）---
    rt = load("retest_metrics.csv")
    add("反復測定で対照を上回らなかったセット数",
        len(rt) - int(rt.icc_q.lt(0.05).sum()), 0, required=False)

    # --- 図 3: 共変動の水準別に見た手法間一致度 ---
    # 図 3 のキャプションが区間ごとのセット数と最不一致対の中央値を引く。
    # 図の生成コード（fig3_method_agreement）と同じ区間で数え直して照合対象にする。
    for lo, hi in ((-1, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 1.0)):
        k = (m.internal_consistency >= lo) & (m.internal_consistency < hi)
        tag = f"<{hi}" if lo < 0 else f"{lo}-{hi}"
        # キャプションが引く値は qc_tables が逆向きに照合するので、
        # ここでは計算だけしておき、本文に出るかどうかは必須にしない。
        add(f"[図3 {tag}] セット数", int(k.sum()), 0, required=False)
        add(f"[図3 {tag}] 全手法対の平均", m.loc[k, "method_agreement_mean"].median(), 3,
            required=False)
        add(f"[図3 {tag}] 最不一致対", m.loc[k, "method_agreement_min"].median(), 3,
            required=False)

    # --- ファミリーのサイズ帯の偏り ---
    # 図 2 のキャプションが「ファミリーの上下をそのまま比べてはならない」根拠として
    # この 2 件を引く。サイズ帯が重なっていないことを数で示す部分なので照合対象にする。
    add("CORUM 複合体のうち 25 遺伝子以下の件数",
        (m[(m.family == "complex") & (m.n_genes_tested <= 25)]).shape[0], 0)
    add("細胞種マーカーのうち 61-200 遺伝子の件数",
        (m[(m.family == "celltype") & m.n_genes_tested.between(61, 200)]).shape[0], 0)

    # --- 参照アンカー NOX2 複合体 ---
    # 図 1B のキャプションが「条件効果は明快だが安静時の共変動は対照を下回る」典型例として
    # この 2 値を名指しで引くので、機械照合の対象に入れる（qc_tables が図キャプションを
    # 逆向きに検査するため、ここに無いと裏付けなしとして落ちる）。
    nox2 = m[m.set == "anchor|NOX2_complex"]
    if len(nox2) == 1:
        r = nox2.iloc[0]
        add("NOX2 アンカーの条件効果 |d|", abs(r.cohens_d), 2)
        add("NOX2 アンカーの内部整合性", r.internal_consistency, 3)

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

    # --- 帰属推定値の 95% 区間（中心化部分抽出）---
    # 原稿は「両コホートの細胞組成は区間が重ならない」と主張しているので、
    # 上限・下限の両方を照合対象にする。片方だけだと主張を裏づけられない。
    for path, tag in [("gtex_blood/pc_attribution_ci.csv", "GTEx"),
                      ("pc_attribution_ci.csv", "GSE35846")]:
        try:
            ci = load(path)
        except FileNotFoundError:
            continue
        lo_col = "部分標本 2.5%" if "部分標本 2.5%" in ci.columns else "95%CI 下限"
        hi_col = "部分標本 97.5%" if "部分標本 97.5%" in ci.columns else "95%CI 上限"
        for _, r in ci.iterrows():
            add(f"{tag} {r['分類']} 範囲下限", r[lo_col], 3)
            add(f"{tag} {r['分類']} 範囲上限", r[hi_col], 3)

    # --- GSE81046 を共通の対照基準で見たときの区画 ---
    # 条件効果をゼロ帰無ではなく同じ 10,000 対照への超過で測ると、主コホートと同じ向きになる。
    _g8 = pd.read_csv(T / "gse81046" / "gene_set_metrics.csv")
    _g8_ic = _g8.null_q.lt(0.05) & _g8.var_null_q.lt(0.05)
    _g8_cond = _g8.cond_null_q.lt(0.05)
    add("GSE81046 共通対照での条件効果(%)", 100 * _g8_cond.mean(), 1)
    add("GSE81046 共通対照で条件効果のみ(%)", 100 * (_g8_cond & ~_g8_ic).mean(), 1)
    add("GSE81046 共通対照でどちらも通らない(%)", 100 * (~_g8_cond & ~_g8_ic).mean(), 1)
    add("GSE81046 条件効果の対照 |d| 中央値", float(_g8.cond_null_mean.median()), 3)

    # --- GSE81046（RNA-seq マクロファージ）の祖先集団 ---
    a = pd.read_csv(T / "gse81046" / "ancestry_attribution.csv")
    p1 = a[a.pc == "PC1"].iloc[0]
    add("GSE81046 PC1 の祖先説明率", p1.r2, 3)
    add("GSE81046 PC1 の祖先偶然水準", p1.r2_chance, 3)
    add("GSE81046 PC1 の祖先超過分", p1.excess, 3)
    add("GSE81046 PC1 の寄与率(%)", 100 * p1.var_ratio, 1)

    # --- 通過 17 件が組成マーカーとどれだけ相関するか（対照との比較つき）---
    # 転写全体が組成の軸に載っているなら、何を集めても組成マーカーと相関する。
    # 絶対値ではなく対照への超過で見ないと「組成を測っている」とは言えない。
    _pc = pd.read_csv(T / "passing_set_composition.csv")
    add("[通過17件] 組成相関が対照を上回る件数", int((_pc.q < 0.05).sum()), 0)
    add("[通過17件] 組成相関の超過 中央値", float(_pc["超過"].median()), 3)
    add("[通過17件] 組成相関の対照中央値 の中央値", float(_pc["対照の最大絶対相関 中央値"].median()), 3)

    # --- 反復測定信頼性 ---
    rt = load("retest_metrics.csv")
    add("ICC が対照を上回るセットの割合(%)", 100 * rt.icc_q.lt(0.05).mean(), 1)
    add("内部整合性が対照を上回る割合(再測定コホート)(%)", 100 * rt.ic_q.lt(0.05).mean(), 1,
        required=False)

    # --- 表現型 ---
    ph = load("phenotype_metrics.csv")
    add("表現型相関が対照を上回る割合(%)", 100 * ph.abs_rho_q.lt(0.05).mean(), 1)

    # --- 3.10 節・2.2 節 通過セットの内訳と非独立性 ---
    # ここは以前コードとして残っておらず原稿にだけ数値があった。対照数を変えると
    # 通過セットの集合ごと動くので、機械照合に載せる。
    po = load("passing_set_overlap.csv").iloc[0]
    add("反復測定を通過したセット数", po["通過セット数"], 0)
    add("通過セットのうち細胞種マーカー", po["細胞種マーカー数"], 0)
    add("通過セットのうち反応経路", po["反応経路数"], 0)
    add("通過セットの Jaccard 中央値", po["Jaccard中央値"], 3)
    add("通過セットの Jaccard 最大", po["Jaccard最大"], 3)
    add("通過セットの合計遺伝子数", po["合計遺伝子数"], 0)
    add("通過セットの重複除去後の遺伝子数", po["重複除去後"], 0)
    add("通過セットの実質遺伝子割合(%)", po["実質割合(%)"], 1)
    add("通過セットの接種直前の絶対相関中央値", po["接種直前_絶対相関_中央値"], 3)
    add("残りセットの接種直前の絶対相関中央値", po["残り_絶対相関_中央値"], 3)
    add("通過セットの 7 日前の絶対相関中央値", po["7日前_絶対相関_中央値"], 3)
    add("通過セットの接種直前の超過 z 中央値", po["接種直前_z_中央値"], 2)
    add("通過セットの 7 日前の超過 z 中央値", po["7日前_z_中央値"], 2)
    # 細胞種マーカーとラベルされていない通過セットも血球組成を測っているか。
    # 3.9 節の「通過セットは組成に対応する」を、ラベルに頼らず内容で裏づける。
    add("非マーカー通過セット数", po["非マーカー通過セット数"], 0, required=False)
    add("非マーカーの組成相関の最小絶対値", po["非マーカーの組成相関の絶対値の最小"], 3, required=False)
    try:
        pc = load("passing_set_composition.csv")
        for _, r in pc.iterrows():
            add(f"[組成相関] {r['set'].split('|')[1][:28]}", r["最大相関"], 3, required=False)
    except FileNotFoundError:
        pass

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

    # --- 一次元性（3.5 節。反映型 / 形成型の測定モデル）---
    dim = pd.read_csv(T / "dimensionality.csv")
    add("一次元性の中央値(%)", 100 * dim.pc1_frac.median(), 1)
    add("同 第1四分位(%)", 100 * dim.pc1_frac.quantile(0.25), 1)
    add("同 第3四分位(%)", 100 * dim.pc1_frac.quantile(0.75), 1)
    add("PC1>=50% のセット数", (dim.pc1_frac >= 0.5).sum(), 0)
    add("同 割合(%)", 100 * (dim.pc1_frac >= 0.5).mean(), 1)
    lo = dim[dim.pc1_frac < 0.3]
    hi = dim[dim.pc1_frac >= 0.3]
    add("PC1<30% のセット数", len(lo), 0)
    add("PC1>=30% のセット数", len(hi), 0)
    add("PC1<30% の最不一致手法対の相関", lo.method_agreement_min.median(), 3)
    add("PC1>=30% の最不一致手法対の相関", hi.method_agreement_min.median(), 3)
    # 参照アンカーは 2 セットしかなく中央値を報告する意味がないので本文に出していない。
    # 監査対象に入れると必ず不一致になるため除く。
    for fam, g in dim[dim.family != "anchor"].groupby("family"):
        add(f"[{fam}] 一次元性の中央値(%)", 100 * g.pc1_frac.median(), 1)

    # --- 前処理の感度（3.11 節・表 7b/7c）---
    ps = load("preprocessing_sensitivity.csv")
    for _, r in ps.iterrows():
        tag = r["条件"][:22]
        add(f"[{tag}] 評価セット", r["評価セット"], 0)
        add(f"[{tag}] 条件効果あり(%)", r["条件効果あり(%)"], 1)
        add(f"[{tag}] 共変動あり(%)", r["個人間の共変動あり(%)"], 1)
        add(f"[{tag}] 条件効果のみ(%)", r["条件効果のみ(%)"], 1)
    # log2(TPM+1) で陽性対照が壊れることが、この腕を棄却する 2 つ目の根拠になる。
    for tag, f, jp in (("採用", "gse81046/gene_set_metrics.csv", "TMM logCPM"),
                       ("分位", "gse81046/gene_set_metrics_quantile.csv", "TMM+分位"),
                       ("TPM", "gse81046/gene_set_metrics_tpm.csv", "log2(TPM+1)")):
        try:
            d = load(f)
        except FileNotFoundError:
            continue
        dd = d[d.family == "data_derived"]
        both = dd.null_q.lt(0.05) & dd.var_null_q.lt(0.05)
        add(f"[{jp}] データ由来モジュールの合格率(%)", 100 * both.mean(), 1)
        add(f"[{jp}] 対照の中央値", d.null_mean.median(), 3)

    # --- ssGSEA を加えた手法間一致度（3.5 節・表 2c）---
    sa = load("ssgsea_agreement.csv")
    add("ssGSEA 解析のセット数", len(sa), 0)
    for b in ["<0.02", "0.02-0.10", "0.10-0.30", ">=0.30"]:
        sub = sa[sa["bin"] == b]
        if not len(sub):
            continue
        add(f"[ssGSEA {b}] セット数", len(sub), 0)
        add(f"[ssGSEA {b}] 4 手法平均", sub.agreement_mean_4.median(), 3)
        # 表 2c は GSVA を入れた 6 手法版にしたので、5 手法の値は原稿に出てこない。
        # 参考値として計算だけ残す（5 手法の値は改稿前との突き合わせに使える）。
        add(f"[ssGSEA {b}] 5 手法平均", sub.agreement_mean_5.median(), 3, required=False)
        add(f"[ssGSEA {b}] 5 手法最不一致", sub.agreement_min_5.median(), 3,
            required=False)
        add(f"[ssGSEA {b}] ssGSEA 対の平均", sub.agreement_mean_ssgsea.median(), 3,
            required=b == "<0.02")
        add(f"[ssGSEA {b}] ssGSEA 対の最不一致", sub.agreement_min_ssgsea.median(), 3,
            required=False)
        # GSVA を実物で入れたので 6 手法の列も照合する（表 2c は 6 手法版）。
        if "agreement_mean_6" in sub.columns:
            add(f"[ssGSEA {b}] 6 手法平均", sub.agreement_mean_6.median(), 3)
            add(f"[ssGSEA {b}] 6 手法最不一致", sub.agreement_min_6.median(), 3)
            add(f"[ssGSEA {b}] GSVA 対の平均", sub.agreement_mean_gsva.median(), 3)
            add(f"[ssGSEA {b}] GSVA 対の最不一致", sub.agreement_min_gsva.median(), 3,
                required=b == "<0.02")
    ok = sa.internal_consistency.notna()
    add("ρ(整合性, 5 手法平均)",
        spearmanr(sa.loc[ok, "internal_consistency"], sa.loc[ok, "agreement_mean_5"],
                  nan_policy="omit").statistic, 3, required=False)
    if "agreement_mean_6" in sa.columns:
        add("ρ(整合性, 6 手法平均)",
            spearmanr(sa.loc[ok, "internal_consistency"], sa.loc[ok, "agreement_mean_6"],
                      nan_policy="omit").statistic, 3)

    # --- 内部整合性の定義の感度（3.11 節）---
    cd_ = load("consistency_definition.csv")
    for tag, jp in (("spearman", "Spearman"), ("pearson", "Pearson")):
        add(f"[定義 {jp}] 合格率(%)", 100 * cd_[f"pass_{tag}"].mean(), 1)
        add(f"[定義 {jp}] 整合性の中央値", cd_[f"ic_{tag}"].median(), 4)
    flip = (cd_.pass_spearman != cd_.pass_pearson)
    add("定義で合否が入れ替わるセット数", int(flip.sum()), 0)
    add("定義で合否が入れ替わる割合(%)", 100 * flip.mean(), 1)
    add("Pearson のみ合格するセット数", int((~cd_.pass_spearman & cd_.pass_pearson).sum()), 0)

    # --- 経験 p の下限と、その下限が BH を縛らないことの検査（2.6 節）---
    # 対照 B 個から作る経験 p は 1/(B+1) を下回れない。旧版は B=20 で下限 0.048、
    # BH-FDR 0.05 を通せず正規近似に頼っていた。B=10,000 では下限が下がるが、
    # 「下限より BH の閾値が上にある」ことを確認しないと下限が判定を縛りうる。
    # そこを機械照合に載せる。
    gm = load("gene_set_metrics.csv")
    n_ctrl = int(gm.n_control.median()) if "n_control" in gm.columns else 10_000
    add("対照セット数", n_ctrl, 0)
    at_floor = int((gm.null_p_empirical <= 1 / (n_ctrl + 1) + 1e-12).sum())
    add("経験 p が下限にあるセット数", at_floor, 0)
    # そのランクにおける BH の閾値。下限より大きければ下限は判定を縛らない。
    add("下限にいるランクでの BH 閾値", 0.05 * at_floor / len(gm), 4)
    assert 1 / (n_ctrl + 1) < 0.05 * at_floor / len(gm), \
        "経験 p の下限が BH の閾値を縛っている。対照数を増やす必要がある"
    passed_both = gm.null_q.lt(0.05) & gm.var_null_q.lt(0.05)
    # 正規近似は使わなくなったので z は参考値に落とす（原稿は引かない）。
    add("両対照通過セットの z 中央値", gm.loc[passed_both, "null_z"].median(), 2,
        required=False)
    add("両対照通過セットの z 最大", gm.loc[passed_both, "null_z"].max(), 1,
        required=False)
    add("帰無分布の歪度の中央値", gm.null_skew.median(), 2)

    # --- GTEx 全血と骨格筋のドナー重複（3.7 節・表 9）---
    # 「固形組織で再現した」は独立再現ではない。重複率を機械照合する。
    def _donors(ns):
        c = pd.read_csv(ROOT / "data" / "metadata" / ns / "covariates.csv")
        return set(c.SAMPID.str.split("-").str[:2].str.join("-"))
    try:
        d1, d2 = _donors("gtex_blood"), _donors("gtex_muscle")
        add("GTEx 全血と骨格筋のドナー重複(%)", 100 * len(d1 & d2) / len(d1), 1)
        add("GTEx 全血と骨格筋の共通ドナー数", len(d1 & d2), 0)
    except FileNotFoundError:
        pass

    # --- チップ交絡と 3 コホートの合格率（3.1 節・3.7 節）---
    # 主コホートの 11.1% はチップと個人が完全交絡した唯一のコホートの値。
    # 単一対照どうしで揃えた 3 コホートの比較を機械照合する。
    bf = load("batch_by_family.csv")
    w = bf.n.sum()
    add("チップ除去前の合格率(%)", 100 * (bf.n * bf["合格_raw"]).sum() / w, 1)
    add("チップ除去後の合格率(%)", 100 * (bf.n * bf["合格_chip"]).sum() / w, 1)
    add("チップ除去前の対照平均", bf.null_raw_mean.median(), 4)
    add("チップ除去後の対照平均", bf.null_chip_mean.median(), 4)
    add("PBMC の整合性合格率(%)", 100 * load("retest_metrics.csv").ic_q.lt(0.05).mean(), 1)
    add("マクロファージの単一対照合格率(%)",
        100 * pd.read_csv(T / "gse81046" / "gene_set_metrics.csv").null_q.lt(0.05).mean(), 1)

    # --- サイズ層別（3.2 節・表 2b）---
    # 合格率は効果量ではなく検出力なので、サイズ帯別の値を機械照合する。
    sz = load("size_stratified.csv")
    for _, r in sz.iterrows():
        add(f"[サイズ {r['band']}] セット数", r["n"], 0)
        add(f"[サイズ {r['band']}] 合格率(%)", r["pass_rate"], 1)
        add(f"[サイズ {r['band']}] 超過分の中央値", r["excess_median"], 4)
        add(f"[サイズ {r['band']}] 対照 SD の中央値", r["null_sd_median"], 4)
        add(f"[サイズ {r['band']}] 超過分÷対照SD", r["excess_over_null_sd"], 2)
    fb = load("size_stratified_family.csv")
    # 本文が引いている complex と pathway の同一帯比較
    for band in ["3-5", "6-10", "11-25", "26-60"]:
        for fam in ["complex", "pathway"]:
            row = fb[(fb.band == band) & (fb.family == fam)]
            if len(row):
                add(f"[{band} {fam}] 超過分", row.iloc[0].excess_median, 4)
    cp = load("cross_platform_rank.csv")
    for _, r in cp.iterrows():
        # 合格率ベースの順位相関は本文で引用しない。合格率そのものがサイズに規定された
        # 検出力統計量なので、それで測った順位一致を根拠に使うと同じ交絡を持ち込む。
        # 効果量（超過分）と中央値の版だけを必須にする。
        req = r["量"] != "pass_rate" and r["層"] in ("全体", "26-60")
        # 中央値の全体版は、原稿が参照アンカーを含む 7 項目で 0.00 と報告しており
        # ここの 6 ファミリー版（0.257）とは対象が違う。必須にしない。
        if r["量"] == "ic_median" and r["層"] == "全体":
            req = False
        add(f"[順位一致 {r['量']} {r['層']}]", r["順位相関"], 3, required=req)

    # --- 系統的検索（1 節・補遺 S1）---
    # 不在の主張の強さは検索の記録に依存するので、段階別件数を機械照合する。
    pr = pd.read_csv(T / "systematic_search_prisma.csv")
    for _, r in pr.iterrows():
        # 0 件は「該当なし」という主張そのもの。1 桁の値は偶然一致しやすいので
        # 必須にしない（audit の弱一致判定に任せる）。
        add(f"[検索] {r['段階'][:34]}", r["件数"], 0, required=int(r["件数"]) >= 100)
    sq = pd.read_csv(T / "systematic_search_queries.csv")
    add("[検索] 検索式の本数", sq["query"].nunique(), 0)

    # --- 分散分解（3.7 節・表 3c）---
    # 一元配置の足し算（paper_sum）は表 3・表 3b の値と一致しなければならない。
    # ここが合わないと、分解が別の符号化で走っていることになり比較が成立しない。
    for tag, path in [("GSE35846", "variance_decomposition_gse35846.csv"),
                      ("GTEx血", "gtex_blood/variance_decomposition.csv"),
                      ("GTEx筋", "gtex_muscle/variance_decomposition.csv")]:
        vd = load(path)
        w = float(vd.drop_duplicates("pc").var_ratio.sum())
        for g, sub in vd.groupby("group"):
            def wsum(col):
                return float((sub[col] * sub.var_ratio / w).sum())
            add(f"[{tag} {g}] 一元配置の足し算", wsum("paper_sum"), 3)
            add(f"[{tag} {g}] Shapley", wsum("shapley"), 3)
            add(f"[{tag} {g}] 固有分", wsum("unique"), 3)
            add(f"[{tag} {g}] 二重計上分", wsum("paper_sum") - wsum("shapley"), 3)

    # 主コホートのチップは PC1 の値を本文が引いている（加重集計ではない）
    vm = load("variance_decomposition_main.csv")
    pc1 = vm[(vm.pc == "PC1") & (vm.group == "chip")].iloc[0]
    add("主コホート PC1 チップ 一元配置", pc1.paper_sum, 3)
    # pc1.unique は Series.unique メソッドに当たるので添字で取る
    add("主コホート PC1 チップ 固有分", pc1["unique"], 3)
    add("主コホート PC1 チップ Shapley", pc1.shapley, 3)
    add("主コホート PC1 チップ 共有分", pc1.paper_sum - pc1.shapley, 3)

    # 採取時の状況と組成の共有分（両組織で最大の共有成分）
    for tag, path in [("GTEx血", "gtex_blood/variance_decomposition_claim_shared.csv"),
                      ("GTEx筋", "gtex_muscle/variance_decomposition_claim_shared.csv")]:
        s = load(path)
        w = float(s.drop_duplicates("pc").var_ratio.sum())
        s = s.assign(wt=s.value * s.var_ratio / w).groupby("shared_between").wt.sum()
        top = s.reindex(s.abs().sort_values(ascending=False).index)
        add(f"[{tag}] 最大の共有成分", top.iloc[0], 3)

    # 主張別 4 群の比（本文が「1/9.8」「17.6 分の 1」等で引いている）
    for tag, path in [("GTEx血", "gtex_blood/variance_decomposition_claim.csv"),
                      ("GTEx筋", "gtex_muscle/variance_decomposition_claim.csv")]:
        c = load(path)
        w = float(c.drop_duplicates("pc").var_ratio.sum())
        agg = {g: {col: float((sub[col] * sub.var_ratio / w).sum())
                   for col in ("paper_sum", "shapley")}
               for g, sub in c.groupby("group")}
        st = agg["安定した個人属性"]
        sit = agg["採取時の状況"]
        comp = agg[[k for k in agg if "組成" in k][0]]
        for col, jp in (("paper_sum", "一元配置"), ("shapley", "Shapley")):
            add(f"[{tag}] 採取状況÷安定属性（{jp}）", sit[col] / st[col], 1)
            add(f"[{tag}] 組成÷安定属性（{jp}）", comp[col] / st[col], 1)

    # --- 対照セット数の感度分析（3.11 節・表 9）---
    # 対照数 x p 値の作り方。表 8 は「B を増やしても正規近似は動かない」ことを
    # 示すために両方を並べているので、両方を照合対象にする。
    cc = pd.read_csv(T / "control_count_sensitivity.csv")
    for _, r in cc.iterrows():
        n = int(r["対照数"])
        how = "正規" if r["p 値の作り方"] == "正規近似" else "経験"
        add(f"[対照{n}個/{how}] 合格率(%)", 100 * r["合格率"], 2)
        if how == "経験":
            add(f"[対照{n}個] 経験 p の下限", r["経験 p の下限"], 5)
            add(f"[対照{n}個] 帰無分布の歪度", r["帰無分布の歪度の中央値"], 3)
    # --- 4.3 節 減衰補正（古典テスト理論の第 3 の柱）---
    # 補正は注釈側と対照側にほぼ同じだけかかるので合格率は動かない、という主張。
    # 「動かない」を主張する以上、動かない幅を機械照合に載せる。
    try:
        at = load("attenuation_correction.csv")
        add("減衰補正前の共変動の中央値", at.ic_raw.median(), 4)
        add("減衰補正後の共変動の中央値", at.ic_corrected.median(), 4)
        add("減衰補正前の対照の中央値", at.null_raw.median(), 4)
        add("減衰補正後の対照の中央値", at.null_corrected.median(), 4)
        add("減衰補正前の合格率(%)", 100 * at.pass_raw.mean(), 1)
        add("減衰補正後の合格率(%)", 100 * at.pass_corrected.mean(), 1)
        add("減衰補正で判定が変わったセット数", int((at.pass_raw != at.pass_corrected).sum()), 0)
        add("減衰補正で評価したセット数", len(at), 0)
    except FileNotFoundError:
        pass

    # --- qc_tables が「裏付けなし」と報告した表の値を登録する ---
    # 逆向きの検査（表の数値 -> 解析出力にあるか）で、誰も照合していない数値が
    # 表 2 / 3b / 4 / 5 / 5b / 6 / 7 / 9 に残っていることが分かった。
    # 数値が正しいかどうかは、ここに登録して初めて機械で守られる。

    # 表 2: 出自ごとの件数（中央値と合格率は既に登録してある）
    for fam, n in m.groupby("family").size().items():
        add(f"[{fam}] 件数", int(n), 0)

    # 表 3b: GTEx 骨格筋の分解と、両組織の「年齢と性別」の合計
    try:
        gm = load("gtex_muscle/pc_attribution.csv")
        for kind in gm.kind.unique():
            add(f"GTEx 骨格筋 {kind}（加重）", weighted_total(gm, kind, "excess"), 3)
        wm = gm.drop_duplicates("pc").set_index("pc").var_ratio.sum()
        perm = (gm.excess * gm.var_ratio / wm).groupby(gm.factor).sum()
        ages = [f for f in perm.index if f in ("AGE", "SEX")]
        if ages:
            add("GTEx 骨格筋 年齢+性別（加重）", float(perm[ages].sum()), 3)
        pc1m = gm[gm.pc == "PC1"]
        top = pc1m.loc[pc1m.excess.idxmax()] if len(pc1m) else None
        if top is not None:
            add("GTEx 骨格筋 PC1 の首位要因の超過", float(top.excess), 3)
    except FileNotFoundError:
        pass
    # g / per はこの関数の前半で別の用途に使い回されているので、読み直す。
    gb = load("gtex_blood/pc_attribution.csv")
    wb = gb.drop_duplicates("pc").set_index("pc").var_ratio.sum()
    perb = (gb.excess * gb.var_ratio / wb).groupby(gb.factor).sum()
    ages_b = [x for x in perb.index if x in ("AGE", "SEX")]
    if ages_b:
        add("GTEx 全血 年齢+性別（加重）", float(perb[ages_b].sum()), 3)
    pc1b = gb[gb.pc == "PC1"]
    if len(pc1b):
        add("GTEx 全血 PC1 の首位要因の超過", float(pc1b.excess.max()), 3)

    # 表 4: 正規化ごとの対照の 5〜95 パーセンタイル
    try:
        nc = load("gse81046/normalization_comparison.csv")
        for _, r in nc.iterrows():
            tag = str(r.iloc[0])[:18]
            for col in ("第1主成分の寄与率", "平均発現量との相関", "検出率との相関",
                        "ランダム対照の整合性(中央値)"):
                if col in nc.columns:
                    nd = 1 if "寄与率" in col else 3  # 表 4 は小数第 3 位で書いている
                    add(f"[正規化 {tag}] {col}", float(r[col]), nd)
            # 5〜95 パーセンタイルは "0.083〜0.261" の文字列で入っている
            band = str(r.get("ランダム対照の整合性(5-95%)", ""))
            for sep in ("〜", "~", "-"):
                if sep in band[1:]:
                    lo, hi = band[:1] + band[1:].split(sep, 1)[0], band[1:].split(sep, 1)[1]
                    try:
                        add(f"[正規化 {tag}] 対照の 5 分位", float(lo), 3)
                        add(f"[正規化 {tag}] 対照の 95 分位", float(hi), 3)
                    except ValueError:
                        pass
                    break
    except FileNotFoundError:
        pass

    # 表 5: 反復測定の ICC 中央値と件数
    add("反復測定 注釈セットの ICC 中央値", rt.icc.median(), 3)
    add("反復測定 対照の ICC 中央値", rt.icc_null_mean.median(), 3)
    add("反復測定 ICC が 0.5 を超える割合(%)", 100 * rt.icc.gt(0.5).mean(), 1)
    add("反復測定 評価セット数", len(rt), 0)
    add("反復測定 対照を上回ったセット数", int(rt.icc_q.lt(0.05).sum()), 0)

    # 表 5b: 選抜群の組成軸除去（変種ごと）
    try:
        st = load("stability_selected.csv")
        for variant, sub in st.groupby("variant"):
            v = str(variant)[:14]
            add(f"[選抜 {v}] 共変動の中央値", sub.ic.median(), 3)
            add(f"[選抜 {v}] 対照の中央値", sub.ic_null_mean.median(), 3)
            add(f"[選抜 {v}] ICC の中央値", sub.icc.median(), 3)
            add(f"[選抜 {v}] ICC 対照の中央値", sub.icc_null_mean.median(), 3)
            add(f"[選抜 {v}] 共変動で上回った群", int(sub.ic_q.lt(0.05).sum()), 0)
            add(f"[選抜 {v}] ICC で上回った群", int(sub.icc_q.lt(0.05).sum()), 0)
            add(f"[選抜 {v}] 群数", len(sub), 0)
            add(f"[選抜 {v}] 共変動で上回った割合(%)",
                100 * sub.ic_q.lt(0.05).mean(), 1)
            add(f"[選抜 {v}] ICC で上回った割合(%)",
                100 * sub.icc_q.lt(0.05).mean(), 1)
    except FileNotFoundError:
        pass

    # 表 6: 表現型の全行（table_rows で行ごと照合しているが、値単位でも登録する）
    add("表 6 2SD 超え（接種直前）", int((ph.abs_rho_z > 2).sum()), 0)
    add("表 6 2SD 超え（7 日前）", int((ph["abs_rho_z-7"] > 2).sum()
                                 if "abs_rho_z-7" in ph.columns
                                 else (ph["abs_rho_z_day-7"] > 2).sum()), 0)
    add("表 6 全セットの絶対相関中央値（接種直前）", ph.rho_day0.abs().median(), 3)
    add("表 6 対照の絶対相関中央値（接種直前）", ph.abs_rho_null_mean.median(), 3)
    add("表 6 評価セット数", len(ph), 0)

    # 表 7: GSE81046 の発現フィルタ閾値ごと
    try:
        ef = load("gse81046/expression_filter_sensitivity.csv")
        for _, r in ef.iterrows():
            tag = str(r.iloc[0])[:14]
            add(f"[閾値 {tag}] 評価セット", int(r["評価セット"]), 0)
            add(f"[閾値 {tag}] 条件効果あり(%)", float(r["条件効果あり(%)"]), 1)
            add(f"[閾値 {tag}] 共変動あり(%)", float(r["内部整合性あり(%)"]), 1)
            add(f"[閾値 {tag}] 条件効果のみ(%)", float(r["条件効果のみ(%)"]), 1)
            add(f"[閾値 {tag}] 対照の中央値", float(r["対照の中央値"]), 3)
    except FileNotFoundError:
        pass

    # 図 1 の 4 区画（キャプションが割合を書いているので値単位でも照合する）
    from src.visualization.fig1_overview import _regions as _reg
    for tag, fname in (("主", "gene_set_metrics.csv"),
                       ("GSE81046", "gse81046/gene_set_metrics.csv")):
        try:
            rr = _reg(load(fname))
        except FileNotFoundError:
            continue
        for k in ("cond_only", "both", "coh_only", "neither"):
            add(f"[図 1 {tag}] {k}(%)", rr[k], 1)

    # 補遺 S3: ラベルと内容が一致しない細胞種マーカー集合の遺伝子数。
    # 補遺の表は「定義上のサイズ」を書いているので、コホートに存在する数
    # （n_genes_present）ではなく注釈の定義から数える。
    # 同名の集合が pathway| 側にもあるので celltype| に限る。
    try:
        from src.common import METADATA, load_config as _lc
        from src.reliability.run_evaluation import load_all_sets as _las
        _all = _las(_lc("gene_sets"))
        for name in ("Hemangioblasts", "Trophoblast Progenitor Cells", "Reticulocytes",
                     "Pluripotent Stem Cells", "Spermatozoa",
                     "Undefined Placental Cells"):
            key = f"celltype|{name}"
            if key in _all:
                add(f"[マーカー集合] {name} の遺伝子数", len(_all[key][1]), 0,
                    required=False)
    except Exception:
        pass

    # 3.5 節と表 9 の順位相関は、層化抽出した 427 セットではなく
    # コホートの全セットで計算した値である（原稿もその旨を書いている）。
    # 別の量なので別に登録する。ここが未登録だったため、
    # 表の側から見る検査（qc_tables）で「裏付けなし」と出ていた。
    for tag, fname in (("主コホート", "gene_set_metrics.csv"),
                       ("GSE81046", "gse81046/gene_set_metrics.csv")):
        try:
            mm = load(fname)
        except FileNotFoundError:
            continue
        okm = mm.internal_consistency.notna() & mm.method_agreement_mean.notna()
        if okm.sum() > 2:
            add(f"[{tag}] ρ(整合性, 手法一致度平均・全セット)",
                spearmanr(mm.loc[okm, "internal_consistency"],
                          mm.loc[okm, "method_agreement_mean"]).statistic, 3)
        add(f"[{tag}] 共変動 0.30 以上のセット数",
            int(mm.internal_consistency.ge(0.30).sum()), 0)

    # 第 2 コホートでも ssGSEA を含む手法一致度を実測した（3.5 節の裏づけ）。
    try:
        sa8 = load("gse81046/ssgsea_agreement.csv")
        ok8 = sa8.internal_consistency.notna()
        for col in ("agreement_mean_4", "agreement_mean_5"):
            if col in sa8.columns:
                add(f"[GSE81046 層化 {col}] ρ(整合性)",
                    spearmanr(sa8.loc[ok8, "internal_consistency"],
                              sa8.loc[ok8, col], nan_policy="omit").statistic, 3,
                    required=False)
        add("GSE81046 層化抽出したセット数", len(sa8), 0, required=False)
    except FileNotFoundError:
        pass

    # --- 3.11 節 BH の独立性仮定を外した検査（Benjamini-Yekutieli）---
    # 「中心の主張は残る」「陽性所見は残らない」の両方を主張するので、
    # 両方の数値を照合に載せる。片方だけ載せると都合のよい側だけが守られる。
    try:
        dp = load("dependency_fdr.csv")
        dr = load("dependency_fdr_regions.csv")
        by_pen = dp[dp["対象"] == "主コホート 条件効果"]["BY の閾値の厳しさ（倍）"]
        if len(by_pen):
            add("BY の閾値の厳しさ（主コホート、倍）", float(by_pen.iloc[0]), 2)
        for label, key in (("主コホート", "主コホート"), ("GSE81046", "GSE81046")):
            sub = dr[dr["コホート"] == key]
            for q in ("条件効果あり(%)", "共変動あり(%)", "条件効果のみ(%)"):
                row = sub[sub["量"] == q]
                if not len(row):
                    continue
                add(f"[BY {label}] {q}", float(row.iloc[0]["BY"]), 1)
        for name in ("反復測定 ICC", "表現型相関（接種直前）"):
            row = dp[dp["対象"] == name]
            if len(row):
                add(f"[BY] {name} の BH 通過", int(row.iloc[0]["BH 通過"]), 0)
    except FileNotFoundError:
        pass

    # 提言の根拠になる下限。検定数 m のうち K 件を通すには 1/(B+1) < alpha K / m。
    m_main = len(load("gene_set_metrics.csv"))
    k_main = int((load("gene_set_metrics.csv").null_q.lt(0.05)
                  & load("gene_set_metrics.csv").var_null_q.lt(0.05)).sum())
    add("提言の対照数の下限", 1 / (0.05 * k_main / m_main) - 1, 0)

    return out



def oneway_vs_tables_3(paper: dict[tuple[str, str], float]) -> tuple[int, int, float]:
    """表 3c の「一元配置の足し算」列が表 3・表 3b と何項目一致するかを数える。

    ここは一度「3 コホートすべてで一致する」と書いてしまった箇所である。
    実際には対照の引き直しで小数第 3 位が動き、9 項目のうち一致するのは 4 項目で、
    残りの差は最大 0.008 だった。数え直せる形にしておかないと同じ誤りに戻る。
    """
    ta = load("technical_axes.csv")
    calc: dict[tuple[str, str], float] = {}
    for kind, jp in (("composition", "細胞組成"), ("technical", "技術"),
                     ("biological", "生物学")):
        calc[("GSE35846 全血", jp)] = weighted_total(ta, kind, "r2_excess")
    for tag, sub in (("GTEx 全血", "gtex_blood"), ("GTEx 骨格筋", "gtex_muscle")):
        try:
            g = load(f"{sub}/pc_attribution.csv")
        except FileNotFoundError:
            continue
        for kind in g.kind.unique():
            calc[(tag, kind)] = weighted_total(g, kind, "excess")

    n_agree = 0
    worst = 0.0
    for k, v in paper.items():
        c = calc.get(k)
        if c is None:
            continue
        d = abs(round(c, 3) - v)
        if d < 5e-4:
            n_agree += 1
        else:
            worst = max(worst, d)
    return n_agree, len(paper), round(worst, 3)

def table_rows() -> list[tuple[str, str, str]]:
    """表の行を「ラベル + 数値の並び」ごと照合する。(項目, 和文の行, 英文の行)。

    なぜ値単位の照合では足りないか
      checks() は「この数値が原稿のどこかに文字列として出てくるか」を見る方式で、
      0・15・117 のような小さい整数はどこにでも現れるため一致が保証にならない。
      実際に表 6 の行ラベルが本文と食い違ったまま、必須照合を全件通してしまった。
      表 6 の行はどれも小さい整数なので、値単位では原理的に守れなかった。

    そこで表の行そのものを Markdown の行として組み立て、原稿に同じ並びで
    存在するかを見る。数値が変われば行が一致しなくなり、行ラベルを書き換えれば
    ここも直さざるを得ない。ラベルと数値の対応が固定されるのが要点である。
    """
    ph = load("phenotype_metrics.csv")
    z0 = int((ph.abs_rho_z > 2).sum())
    z7 = int((ph["abs_rho_z_day-7"] > 2).sum())
    q0 = int(ph.abs_rho_q.lt(0.05).sum())
    q7 = int(ph["abs_rho_q_day-7"].lt(0.05).sum())
    both2 = (ph.abs_rho_z > 2) & (ph["abs_rho_z_day-7"] > 2)
    b2 = int(both2.sum())
    b2q = int((both2 & (ph.abs_rho_q.lt(0.05) | ph["abs_rho_q_day-7"].lt(0.05))).sum())
    r0 = ph.rho_day0.abs().median()
    r7 = ph["rho_day-7"].abs().median()
    n0 = ph.abs_rho_null_mean.median()
    n7 = ph["abs_rho_null_mean_day-7"].median()
    n = len(ph)

    # 図 1 のキャプションが持つ 8 つの割合。区画の定義は fig1_overview._regions に
    # 一本化してあるので、図とキャプションが同じ計算から出ることを保証できる。
    from src.visualization.fig1_overview import _regions
    rb = _regions(load("gene_set_metrics.csv"))
    rc = _regions(load("gse81046/gene_set_metrics.csv"))

    def _reg_ja(r: dict) -> str:
        return (f"どちらも通らない {r['neither']:.1f}%、条件効果のみ {r['cond_only']:.1f}%、"
                f"共変動のみ {r['coh_only']:.1f}%、両方 {r['both']:.1f}%")

    def _reg_en(r: dict) -> str:
        return (f"neither {r['neither']:.1f}%, condition effect only {r['cond_only']:.1f}%, "
                f"coherence only {r['coh_only']:.1f}%, both {r['both']:.1f}%")

    # 表 3c の一元配置列が表 3・表 3b と何項目一致するか。原稿の主張を数え直す。
    PAPER_3C = {
        ("GSE35846 全血", "細胞組成"): 1.050, ("GSE35846 全血", "技術"): 0.133,
        ("GSE35846 全血", "生物学"): 0.089,
        ("GTEx 全血", "血球組成"): 1.510, ("GTEx 全血", "技術"): 0.593,
        ("GTEx 全血", "生物学"): 0.388,
        ("GTEx 骨格筋", "細胞組成"): 0.992, ("GTEx 骨格筋", "技術"): 0.315,
        ("GTEx 骨格筋", "生物学"): 0.340,
    }
    n_ag, n_tot, worst = oneway_vs_tables_3(PAPER_3C)

    return [
        # 原稿は「9 項目のうち 4 項目一致、残る 5 項目は対照の引き直しによる差」と書き、
        # 最大差を「0.125 対 0.133」という実値の対で示している。差の絶対値ではなく
        # この対を照合する（差だけを照合していた頃は、重複していた旧ブロックの
        # 「最大 0.008」という言い方に引っ張られていた）。
        ("表 3c と表 3・表 3b の一致数",
         f"表 3・表 3b と一致するのは {n_tot} 項目のうち {n_ag} 項目で、"
         f"残る {n_tot - n_ag} 項目は対照の引き直しによる差が残る",
         f"It agrees with Tables 3 and 3b for four of the nine entries; "
         f"the other five differ because the controls are redrawn"),
        ("図 1 パネル B の 4 区画", _reg_ja(rb), _reg_en(rb)),
        ("図 1 パネル C の 4 区画（GSE81046）", _reg_ja(rc), _reg_en(rc)),
        ("表 6 2 標準偏差超え",
         f"| 自分のランダム対照を 2 標準偏差以上上回ったセット | {z0} 件 | {z7} 件 |",
         f"| Sets exceeding their own random controls by 2 SD or more | {z0} | {z7} |"),
        ("表 6 BH-FDR 通過",
         f"| 対照に対する BH-FDR 0.05 を通ったセット | {q0} 件 | {q7} 件 |",
         f"| Sets clearing BH-FDR 0.05 against their own controls | {q0} | {q7} |"),
        ("表 6 両時点で 2 標準偏差超え",
         f"| 両方の採血で 2 標準偏差以上上回ったセット | {b2} 件 | — |",
         f"| Sets exceeding 2 SD at both draws | {b2} | — |"),
        ("表 6 うち BH-FDR 通過",
         f"| そのうち、いずれかの採血で BH-FDR 0.05 を通ったセット | {b2q} 件 | — |",
         f"| Of those, sets clearing BH-FDR 0.05 at either draw | {b2q} | — |"),
        ("表 6 全セットの絶対相関中央値",
         f"| 全 {n:,} セットの相関の絶対値の中央値 | {r0:.3f} | {r7:.3f} |",
         f"| Median absolute correlation across all {n:,} sets | {r0:.3f} | {r7:.3f} |"),
        ("表 6 対照の絶対相関中央値",
         f"| 同サイズ・同発現量のランダム対照 | {n0:.3f} | {n7:.3f} |",
         f"| Size- and expression-matched random controls | {n0:.3f} | {n7:.3f} |"),
        # 本文が表 6 の 14 件を受けて書いている箇所。表と本文を同じ数値に縛る。
        ("本文 接種直前の BH-FDR 通過件数",
         f"接種直前の採血では {q0} セットが対照に対する BH-FDR 0.05 を通り",
         f"At the draw immediately before vaccination, {q0} sets exceed their controls "
         f"at BH-FDR 0.05"),
    ]

def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    targets = [Path(a) for a in args] if args else DEFAULT_MDS
    rc = 0
    checks_cache = checks()
    rows_cache = table_rows()
    for md in targets:
        if not md.exists():
            print(f"★ 見つからない: {md}")
            rc = 1
            continue
        rc |= audit_one(md, checks_cache, rows_cache)
        print()
    return rc


def audit_one(md: Path, rows: list[tuple[str, float, int, bool]],
              trows: list[tuple[str, str, str]] | None = None) -> int:
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

    # flat はカンマを外した本文。廃止値の側もカンマを外さないと、
    # 「2,424」「1,776」のようなカンマ入りの値が一度も検査されない。
    # 実際にこの 4 件は長らく検査されていなかった。
    def _flatten(v: str) -> str:
        return v.replace(",", "").replace("，", "").replace("−", "-")

    stale = [(v, why) for v, why in RETIRED if _flatten(v) in flat]

    # 表の行照合。原稿が和文か英文かで照合する文字列を切り替える。
    is_en = md.name.endswith("-en.md")
    row_ng = [(label, en if is_en else ja)
              for label, ja, en in (trows or [])
              if (en if is_en else ja) not in text]

    print(f"=== 数値照合: {md.name} ===")
    print(f"必須 {len(ok)} / {len(ok) + len(ng)} 一致 ／ 廃止値の残存 {len(stale)} 件"
          f" ／ 表の行 {len(trows or []) - len(row_ng)} / {len(trows or [])} 一致")
    if row_ng:
        print()
        print("【表の行が一致しない】ラベルと数値の対応が原稿と違う")
        for label, s in row_ng:
            print(f"  {label}")
            print(f"    期待: {s}")
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
    bad = bool(ng or stale or row_ng)
    print("\n" + ("要修正" if bad else "食い違いなし"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
