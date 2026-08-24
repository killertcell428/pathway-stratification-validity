"""系統的検索の結果を 2 段階でスクリーニングし、PRISMA の段階別件数を出す。

段階の分け方
  第 1 段（規則ベース・自動）
    選定基準を満たすには、抄録に少なくとも
      (i) 測定性質の語（反復測定 / 内部整合性）と遺伝子セットの語、または
      (ii) 対照設計の語（ランダム遺伝子集合等）と遺伝子セットの語
    が現れる必要がある。どちらも無いレコードは基準を満たしえないので除く。
    PRISMA 2020 項目 8 に従い、自動化を使ったこととその規則をここに記録する。
    **規則は緩い側に振ってある**（片方の語だけで通る）。落とすリスクを避けるため。

  第 2 段（抄録の読み取り・人手）
    第 1 段を通ったものについて、選定基準と除外理由 E1-E5 を当てる。
    近いが基準を満たさないもの（near-miss）は別に記録し、本文で論じる。

感度の担保
  既知の最近接研究 4 件（引用 [1][2][3][12]）が第 1 段を通ることを assert する。
  通らなければ規則が厳しすぎる合図なので止める。

出力
  results/tables/systematic_search_screened.csv  第 1 段を通ったレコード
  results/tables/systematic_search_prisma.csv    段階別件数
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from ._search_blocks import B

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"

# 第 1 段の規則に使う語群。_search_blocks の定義を再利用する（二重管理を避ける）。
OUTCOME = B["RETEST"] + B["CONSISTENCY"] + ["reliability", "reproducib", "ICC"]
CTRL = B["CONTROL"]
SETW = B["SET"] + B["SET_NARROW"]

# 既知の最近接研究。第 1 段を通ることを確かめる（規則の感度チェック）。
SEEDS = {
    "[1] Toro-Domínguez 2025": "Benchmarking single-sample gene set scoring",
    "[2] Starmans 2011": "simple but highly effective approach to evaluate the prognostic",
    "[3] Venet 2011": "Most random gene expression signatures",
    "[12] Scheid 2018": "longitudinal stability and interindividual variability",
}


# ---- 第 2 段: 抄録の読み取り ----
# 第 1 段を通った 775 件を、さらに規則で 4 クラスに分ける。
# C1（測定性質 + 対照設計の両方に言及）だけを全読みし、
# 残りは該当する除外理由で落とす。読んだのは C1 のみであることを明記する。
REPEAT = ["test-retest", "test retest", "intraclass", "icc ", "same individuals",
          "same subjects", "same donors", "same patients", "two timepoints",
          "two time points", "repeated measure", "repeated sampling", "repeat sampling",
          "paired sample", "within-subject", "within subject", "longitudinal stability",
          "serial sampl", "repeat blood", "re-measure", "remeasure"]
COVAR = ["across individuals", "between individuals", "inter-individual", "interindividual",
         "inter individual", "internal consistency", "cronbach", "inter-gene correlation",
         "intergene correlation", "split-half", "co-expression across", "correlated across"]
CTRL_STRICT = ["random gene set", "random gene sets", "random signature", "random signatures",
               "random gene expression signature", "size-matched", "matched random",
               "background gene set", "null gene set", "permuted gene set",
               "randomly selected gene", "random gene list", "matched control gene",
               "random set of gene"]

# C1 を全読みした結果。抄録を読んで下した判定をここに残す（査読者が追えるように）。
# 「該当なし」= 選定基準を満たす、はこの検索では 1 件も出なかった。
READ_VERDICTS = [
    ("residual-ratio framework for auditing",
     "near-miss N1",
     "サイズをそろえたランダム 30 遺伝子対照と比べて署名を監査する。ただし測る量は"
     "発現主成分部分空間への吸収率であり、スコアの個人間共変動でも再測定安定性でもない"),
    ("Mandatory Validation Gates",
     "near-miss N2",
     "層別化の報告前にランダム遺伝子集合ゲートを課す。判定対象はクラスタリング結果であり、"
     "個々の遺伝子セットの測定性質ではない"),
    ("Most random gene expression signatures",
     "near-miss N3", "ランダム署名が予後と関連することを示す。転帰との関連であり個人間共変動ではない"),
    ("simple but highly effective approach to evaluate the prognostic",
     "near-miss N3", "同上（予後性能の評価法）"),
    ("Why breast cancer signatures are no better than random signatures",
     "near-miss N3", "同上（機構の説明）"),
    ("Association between expression of random gene sets and survival",
     "near-miss N3", "同上（多癌種への拡張）"),
    ("Random gene sets in predicting survival",
     "near-miss N3", "同上（肝細胞癌）"),
    ("Removing the association of random gene sets and survival time",
     "near-miss N3", "同上（補正法）"),
    ("Brawer", "near-miss N3", "同上（補正法）"),
    ("longitudinal stability and interindividual variability",
     "near-miss N4", "個人内で縦断安定・個人間で変動する遺伝子を選抜。対照との比較を含まない（引用 [12]）"),
    ("Benchmarking single-sample gene set scoring",
     "near-miss N5", "変数が手法でありセットの出自ではない（引用 [1]、除外理由 E2）"),
    ("estimating coherence of molecular mechanisms",
     "除外 E5", "コヒーレンスの対象がタンパク相互作用網の次数分布で、発現の個人間共変動ではない"),
    ("Prognostic Gene Pairs From Cancer Patient-Specific",
     "除外 E1", "患者別相関網から予後遺伝子対を導く。転帰予測であり測定性質の評価ではない"),
    ("A significant enrichment that is not",
     "除外 E5", "空間的自己相関を保つ帰無モデルの話で、対象は脳領域の空間富化"),
    ("Regulator-Centric Eligible Universe",
     "除外 E5", "matched random は遺伝子制御網のランダム辺。遺伝子セットスコアでも個人でもない"),
    ("Baseline gene expression in subcutaneous adipose tissue predicts",
     "除外 E1", "ランダム遺伝子と比べるが、比較対象は群分類（体重減少の高低）の予測性能であり測定性質ではない"),
    # 取得上限を 1000 → 3000 に上げて S7 の全件を取得したことで C1 に入った 3 件。
    # いずれも 2001-2004 年の腫瘍 vs 正常の cDNA マイクロアレイで、
    # "randomly selected genes" は RT-PCR 検証用に無作為抽出した遺伝子を指す偶然一致。
    ("Gene expression profiling of breast cancers with emphasis of beta-catenin",
     "除外 E5", "ランダム抽出した遺伝子は RT-PCR での確認対象。対照設計ではない"),
    ("Gene Expression Profiling of Non-Small Cell Lung Cancer",
     "除外 E5", "同上"),
    ("Expression profiling suggested a regulatory role of liver-enriched transcription factors",
     "除外 E5", "同上"),
]

# C2（測定性質はあるが対照設計の言及なし）に、対照なしで反復測定信頼性を測った研究が
# 埋もれていないかを確かめる。セットの「スコア」語と反復測定語の両方を含むものを全読みした。
# 結果は 10 件すべて偽陽性だった（"ICC" が intrahepatic cholangiocarcinoma、
# "pathway activity" が皮質脊髄路・HEART pathway 等）。この確認自体を記録する。
C2_EXTRA_SCORE = ["gene set score", "pathway activity score", "signature score",
                  "enrichment score", "pathway activity", "ssgsea", "gsva", "singscore"]


def stage2(screened: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    s = screened.copy()
    s["t"] = (s.title + " " + s.abstract).str.lower()
    # 列名に cov / rep を使ってはいけない。s.cov は DataFrame.cov() メソッドに
    # 解決され、s.rep | s.cov が「配列 | メソッド」になって AssertionError になる。
    s["has_repeat"] = s.t.map(lambda x: any(k in x for k in REPEAT))
    s["has_covar"] = s.t.map(lambda x: any(k in x for k in COVAR))
    s["has_ctrl"] = s.t.map(lambda x: any(k in x for k in CTRL_STRICT))
    prop = s["has_repeat"] | s["has_covar"]

    def cls(prop_i, ctl_i):
        if prop_i and ctl_i:
            return "C1 全読み対象（測定性質と対照設計の両方に言及）"
        if prop_i:
            return "C2 除外（測定性質はあるが対照設計の言及なし）"
        if ctl_i:
            return "C3 除外（対照設計はあるが測定性質が再測定/個人間共変動でない）"
        return "C4 除外（どちらも該当しない）"

    s["stage2_class"] = [cls(p, c) for p, c in zip(prop, s["has_ctrl"])]
    s["verdict"] = ""
    s["verdict_reason"] = ""
    for pat, verdict, reason in READ_VERDICTS:
        m = s.title.str.contains(pat, case=False, regex=False)
        s.loc[m, "verdict"] = verdict
        s.loc[m, "verdict_reason"] = reason
    c2 = s.stage2_class.str.startswith("C2")
    extra = c2 & s.t.map(lambda x: any(k in x for k in C2_EXTRA_SCORE)) & s["has_repeat"]
    s.loc[extra & (s.verdict == ""), "verdict"] = "除外 E5（C2 追加読み）"
    s.loc[extra & (s.verdict == "除外 E5（C2 追加読み）"), "verdict_reason"] = (
        "検索語の偶然一致。ICC が intrahepatic cholangiocarcinoma、"
        "pathway activity が皮質脊髄路・HEART pathway 等を指していた")
    counts = [(k, int(v)) for k, v in s.stage2_class.value_counts().sort_index().items()]
    counts.append(("うち C2 から追加で全読みした件数（スコア語+反復測定語）", int(extra.sum())))
    return s, counts


def has(text: str, terms: list[str]) -> bool:
    t = text.lower()
    return any(x.lower() in t for x in terms)


def main() -> int:
    d = pd.read_csv(TABLES / "systematic_search_records.csv").fillna("")
    d["text"] = (d.title + " " + d.abstract)
    n_records = len(d)

    # 抄録が取れなかったものは判定できない。件数を分けて記録する。
    no_abstract = d.abstract.str.len() < 40
    d_abs = d[~no_abstract].copy()

    d_abs["r_outcome"] = d_abs.text.map(lambda s: has(s, OUTCOME))
    d_abs["r_ctrl"] = d_abs.text.map(lambda s: has(s, CTRL))
    d_abs["r_set"] = d_abs.text.map(lambda s: has(s, SETW))
    keep = d_abs.r_set & (d_abs.r_outcome | d_abs.r_ctrl)
    screened = d_abs[keep].copy()

    # 感度チェック: 既知 4 件が残っているか
    print("=== 第 1 段の規則の感度（既知の最近接研究）===")
    missing = []
    for name, pat in SEEDS.items():
        hit = screened.title.str.contains(pat, case=False, regex=False).any()
        print(f"  {'通過' if hit else '落選'}  {name}")
        if not hit:
            missing.append(name)
    assert not missing, f"規則が厳しすぎる。落ちた既知文献: {missing}"

    screened["screen_rule"] = [
        ("測定性質+セット" if o and not c else "対照+セット" if c and not o else "両方")
        for o, c in zip(screened.r_outcome, screened.r_ctrl)]
    cols = ["id", "year", "journal", "title", "found_in", "sources", "screen_rule", "abstract"]
    screened[cols].to_csv(TABLES / "systematic_search_screened.csv",
                          index=False, encoding="utf-8")

    s2, s2_counts = stage2(screened)
    s2.drop(columns=["t"]).to_csv(TABLES / "systematic_search_stage2.csv",
                                  index=False, encoding="utf-8")
    n_c1 = int(s2.stage2_class.str.startswith("C1").sum())
    n_read = int((s2.verdict != "").sum())
    n_included = int(s2.verdict.str.startswith("該当").sum())

    q = pd.read_csv(TABLES / "systematic_search_queries.csv")
    prisma = [
        ("識別: ヒット総数（検索式 x DB、重複含む）", int(q.hits.sum())),
        ("識別: 取得したレコード", int(q.retrieved.sum())),
        ("重複除去後のレコード", n_records),
        ("除外: 抄録が取得できず判定不能", int(no_abstract.sum())),
        ("第 1 段の判定対象", len(d_abs)),
        ("第 1 段で除外（測定性質も対照設計も言及なし）", int((~keep).sum())),
        ("第 2 段（抄録読み取り）に進んだレコード", len(screened)),
    ] + s2_counts + [
        ("抄録を全読みして判定を記録したレコード（C1 全件 + 近接タイトル + C2 追加）", n_read),
        ("選定基準をすべて満たした研究（included）", n_included),
        ("基準を満たさないが近い研究（near-miss、本文で論じる）",
         int(s2.verdict.str.startswith("near-miss").sum())),
    ]
    pd.DataFrame(prisma, columns=["段階", "件数"]).to_csv(
        TABLES / "systematic_search_prisma.csv", index=False, encoding="utf-8")

    print(f"\n=== PRISMA 段階別 ===")
    for k, v in prisma:
        print(f"  {k:48s} {v:6,d}")
    print(f"\n=== 第 1 段通過分の規則別 ===")
    for k, v in screened.screen_rule.value_counts().items():
        print(f"  {k:20s} {v:5d}")
    print(f"\n-> {TABLES/'systematic_search_screened.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
