"""検索式を概念ブロック 1 か所から組み立てる。

なぜ分けたか
  PubMed と Europe PMC に別々の文字列を手書きすると、同じ概念を検索しているという
  保証がなくなる。実際に最初は Europe PMC 側を全文検索のまま書いてしまい、
  S3 が 25,318 件・S5 が 17,108 件ヒットして取得上限 400 件で切れた。
  それでは「何件をスクリーニングしたか」を書けない（PRISMA が成立しない）。

  そこで概念ブロックを 1 か所に置き、各 DB の構文へ機械的に展開する。
  PubMed は [tiab]、Europe PMC は TITLE_ABS: で、どちらもタイトルと抄録に限定する。
"""

from __future__ import annotations

# 概念ブロック。値はフレーズまたは単語の一覧。
B = {
    # 既知の最近接研究 4 件で感度を確かめたところ、当初の SET は
    # Scheid 2018（"Gene expression signatures ..."）と Venet 2011
    # （"Most random gene expression signatures ..."）を拾えなかった。
    # "gene expression signature" 系の語形を足す。
    "SET": ["gene set", "gene signature", "gene signatures", "gene-set", "pathway",
            "gene expression signature", "gene expression signatures",
            "expression signature", "expression signatures",
            "transcriptional signature", "transcriptomic signature"],
    "SET_NARROW": ["gene set score", "gene set scoring", "pathway score",
                   "signature score", "ssGSEA", "GSVA", "singscore",
                   "single-sample GSEA", "single sample gene set enrichment"],
    "SCORE": ["score", "scoring", "enrichment"],
    "RETEST": ["test-retest", "test retest", "intraclass correlation",
               "repeat measurement", "repeated measurement", "repeated sampling",
               "longitudinal stability", "within-person stability"],
    # Venet 2011 と Starmans 2011 はいずれも「ランダムなシグネチャ」を対照に使うが、
    # "random gene set" とは書かない。語形を足さないと最近接研究を落とす。
    "CONTROL": ["random gene set", "random gene sets", "size-matched",
                "matched random", "background gene set", "null gene set",
                "permuted gene set", "random signature", "random signatures",
                "random gene expression signature", "random gene expression signatures",
                "randomly selected genes", "random gene lists"],
    "CONSISTENCY": ["internal consistency", "Cronbach", "inter-gene correlation",
                    "intergene correlation", "co-expression across individuals",
                    "coexpression across individuals", "split-half"],
    "PERSON": ["single-sample", "single sample", "per-patient", "patient-level",
               "individual-level", "per-sample"],
    "VALIDITY": ["stratification", "stratify", "measurement validity",
                 "construct validity", "measurement property",
                 "measurement properties", "reliability"],
    "STABLE": ["reliability", "reproducibility", "stability", "robustness",
               "validity", "variability"],
}


def _pubmed(block: str) -> str:
    return "(" + " OR ".join(f'"{t}"[tiab]' for t in B[block]) + ")"


def _epmc(block: str) -> str:
    # Europe PMC はフレーズを二重引用符で囲めば TITLE_ABS に効く。
    return "(" + " OR ".join(f'TITLE_ABS:"{t}"' for t in B[block]) + ")"


def build(blocks: list[str], db: str) -> str:
    f = _pubmed if db == "pubmed" else _epmc
    return " AND ".join(f(b) for b in blocks)


# 検索式の定義。S1-S4 が主検索、S5 は取りこぼしを見る網。
QUERY_DEF = {
    "S1": (["SET", "SCORE", "RETEST"], "遺伝子セット/経路スコアの反復測定信頼性"),
    "S2": (["SET", "SCORE", "CONTROL"], "遺伝子セットスコアをサイズをそろえたランダム対照と比べる"),
    "S3": (["SET", "CONSISTENCY"], "遺伝子セットの内部整合性・個人間の遺伝子間相関"),
    "S4": (["PERSON", "SET", "SCORE", "VALIDITY"], "単一サンプルスコアを個人の層別化に使う妥当性"),
    "S5": (["SET_NARROW", "STABLE"], "取りこぼしを見る網（単一サンプルスコアの再現性一般）"),
    # S1 は SCORE を要求するため、スコア化に触れず遺伝子単位で縦断安定性を測った
    # 研究（Scheid 2018 など）を落とす。SCORE を外した版を独立の検索式として置く。
    "S6": (["SET", "RETEST"], "スコア化に言及せず遺伝子セットの縦断安定性を測る研究"),
    # Venet / Starmans 型（ランダムなシグネチャを対照に使う）を正面から取る。
    "S7": (["CONTROL"], "ランダム遺伝子集合を対照に使う研究（対照設計そのもの）"),
}


def queries() -> dict[str, dict]:
    return {k: {"狙い": aim,
                "blocks": blocks,
                "pubmed": build(blocks, "pubmed"),
                "epmc": build(blocks, "epmc")}
            for k, (blocks, aim) in QUERY_DEF.items()}
