"""最近接 4 研究の引用を前後にたどり、検索が取りこぼした研究がないかを確かめる。

なぜ要るか
  補遺 S1 の検索は語による検索である。語が合わなければ、どれだけ網を広げても
  該当研究は出てこない。実際、初回の S1-S5 は最近接 4 件のうち 3 件を落としており、
  概念ブロックに語形を足して初めて 4/4 になった。**語の網には、語形の想定という
  同じ弱点が残る。**引用をたどる経路はこの弱点を持たない。ある研究が最近接の研究を
  引いている、あるいは引かれているという関係は、語形の一致を要求しないからである。

  PRISMA 2020 の項目 7 は、データベース検索以外の情報源（引用の追跡を含む）を
  使ったならその方法を報告することを求める。補遺 S1 は当初この経路を実施して
  いないことを限界として明記していた。ここで実施し、限界を 1 つ埋める。

やり方
  最近接 4 研究（引用 [1][2][3][12]）について Europe PMC の references（その研究が
  引いているもの＝後ろ向き）と citations（その研究を引いているもの＝前向き）を取り、
  検索で既出のものを除いたうえで、**補遺 S1 の第 1 段と同じ規則**を当てる。
  規則を作り直さないのは、経路が違っても判定の基準は同じでなければ比較にならないため。

出力
  results/tables/citation_chase.csv          追跡で得た新規レコードと判定
  results/tables/citation_chase_summary.csv  段階別の件数

使い方:
  pixi run citation-chase
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .screen_search import CTRL, OUTCOME, SETW, has

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UA = {"User-Agent": "T26-citation-chase (mailto:noreply@example.org)"}
PAGE = 1000
BATCH = 50          # 抄録をまとめて引く 1 リクエストあたりの件数

# 最近接 4 研究。補遺 S1 の感度チェックで使っているものと同じ集合。
SEEDS = {
    "[1] Toro-Dominguez 2025": "41405962",
    "[2] Starmans 2011": "22163293",
    "[3] Venet 2011": "22028643",
    "[12] Scheid 2018": "29352003",
}


# 第 1 段を通過したものを個別に読んだ結果。査読者が追えるように判定をここに残す。
# 除外理由は補遺 S1 の E1-E5 と同じものを使う。**選定基準を満たしたものは 1 件もない。**
VERDICTS = {
    "37951307": ("除外 E1", "内分泌療法抵抗性の予後シグネチャの導出。群間比較の妥当性のみ"),
    "38205542": ("除外 E5", "腎障害の代謝の話で、遺伝子セットスコアが主題ではない"),
    "37085598": ("除外 E1", "低酸素分類器の臨床実装。アッセイの再現性は測るが同一個人の再測定ではなく、対照もない"),
    "37096121": ("除外 E1", "乳癌の生存に関連する遺伝子の同定。転帰予測であり測定性質ではない"),
    "37444614": ("near-miss C1",
                 "resampling で作った対照シグネチャを基準に既発表シグネチャを評価し、"
                 "どれも対照を有意に上回らないと結論する。ただし比較する量は臨床転帰の予測性能であり、"
                 "スコアの個人間共変動でも再測定信頼性でもない（引用 [3] の系譜）"),
    "34827124": ("near-miss C2",
                 "同一検体の反復シーケンスで ssGSEA と singscore のスコアの対相関を測り、"
                 "再現するには反復 2 回以上が要るとする。測っているのは技術的反復であって"
                 "同一個人の再測定ではなく、サイズをそろえた対照との比較もない"),
    "30081096": ("除外 E1", "シグネチャ推論の頑健化の指針。転帰予測の再現性が対象"),
    "31062858": ("除外 E2", "差次発現の検定統計量の提案。変数は手法でありセットの出自ではない"),
    "29560831": ("除外 E2", "部分集団に感度のある網解析手法の提案"),
    "28916538": ("除外 E1", "パーキンソン病の血液シグネチャ。患者対対照の群間比較"),
    "29258445": ("除外 E2", "表現型予測のための正則化ニューラルネットの提案"),
    "29322932": ("除外 E1", "DNA メチル化網からの予後シグネチャ。転帰予測であり発現でもない"),
    "30090857": ("除外 E1", "精神病超高リスクの分類。正規化手法の比較で、判定対象は群分類の性能"),
    "26763444": ("除外 E1", "転移性黒色腫の miRNA 予後シグネチャの交差検証"),
    "26247463": ("除外 E1", "膵管癌の 15 遺伝子予後シグネチャ"),
    "26737284": ("除外 E3", "特徴選択の安定性。訓練データの変動に対する頑健性であり、同一個人の再測定ではない"),
    "24558298": ("除外 E2", "GSA 手法 7 種の順位性能の比較。変数は手法"),
    "25654473": ("除外 E1", "クラウドソーシングで遺伝子を選び生存予測に使う"),
    "24369726": ("除外 E1", "肺腺癌の予後シグネチャ。再現性は独立データ間の予測性能を指す"),
    "17401012": ("除外 E1", "ER 陽性乳癌の分子サブタイプの定義"),
}


def fetch(url: str) -> dict:
    """Europe PMC を叩く。落ちることがあるので 3 回まで待って試す。"""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    return {}


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def links(pmid: str, kind: str) -> list[dict]:
    """references / citations を全ページ取る。"""
    out: list[dict] = []
    page = 1
    outer = "referenceList" if kind == "references" else "citationList"
    inner = "reference" if kind == "references" else "citation"
    while True:
        d = fetch(f"{EPMC}/MED/{pmid}/{kind}?format=json&pageSize={PAGE}&page={page}")
        items = d.get(outer, {}).get(inner, [])
        out += items
        if len(items) < PAGE:
            return out
        page += 1


def abstracts(keys: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """(source, id) の組に対して、タイトルと抄録をまとめて引く。"""
    got: dict[tuple[str, str], dict] = {}
    for i in range(0, len(keys), BATCH):
        chunk = keys[i : i + BATCH]
        q = " OR ".join(f"(SRC:{s} AND EXT_ID:{k})" for s, k in chunk)
        url = (f"{EPMC}/search?query={urllib.parse.quote(q)}"
               f"&format=json&pageSize={BATCH}&resultType=core")
        for r in fetch(url).get("resultList", {}).get("result", []):
            got[(r.get("source", ""), str(r.get("id", "")))] = {
                "title": r.get("title", "") or "",
                "abstract": r.get("abstractText", "") or "",
                "year": r.get("pubYear", "") or "",
                "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title", ""),
            }
        time.sleep(0.4)
    return got


def main() -> int:
    print("[1/5] 最近接 4 研究の引用を前後にたどる")
    rows = []
    for name, pmid in SEEDS.items():
        for kind, label in (("references", "後ろ向き"), ("citations", "前向き")):
            items = links(pmid, kind)
            print(f"  {name:26} {label} {len(items):4}")
            for it in items:
                rows.append({
                    "seed": name,
                    "direction": label,
                    "source": it.get("source", "") or "",
                    "id": str(it.get("id", "") or ""),
                    "title_raw": it.get("title", "") or "",
                })
            time.sleep(0.4)
    df = pd.DataFrame(rows)
    n_raw = len(df)

    print("[2/5] 重複を除く")
    df = df[df["id"] != ""].copy()
    # 同じ研究が複数の seed から出てくる。どの seed 由来かは連結して残す。
    agg = (df.groupby(["source", "id"])
             .agg(seeds=("seed", lambda s: "; ".join(sorted(set(s)))),
                  directions=("direction", lambda s: "; ".join(sorted(set(s)))),
                  title_raw=("title_raw", "first"))
             .reset_index())
    print(f"  延べ {n_raw} → 一意 {len(agg)}")

    print("[3/5] 検索で既出のものを分ける")
    rec = pd.read_csv(TABLES / "systematic_search_records.csv")
    known_ids = set(rec["id"].astype(str))
    known_titles = {norm_title(t) for t in rec["title"].fillna("")}
    agg["already_found"] = (agg["id"].isin(known_ids)
                            | agg["title_raw"].map(lambda t: norm_title(t) in known_titles))
    n_known = int(agg.already_found.sum())
    print(f"  検索で既出 {n_known} / 新規 {len(agg) - n_known}")

    print("[4/5] 新規レコードの抄録を引く")
    new = agg[~agg.already_found].copy()
    keys = list(zip(new["source"], new["id"]))
    meta = abstracts(keys)
    for col in ("title", "abstract", "year", "journal"):
        new[col] = [meta.get(k, {}).get(col, "") for k in keys]
    new.loc[new["title"] == "", "title"] = new["title_raw"]
    n_noabs = int((new["abstract"] == "").sum())
    print(f"  抄録あり {len(new) - n_noabs} / 抄録なし（判定不能） {n_noabs}")

    print("[5/5] 補遺 S1 の第 1 段と同じ規則を当てる")
    txt = (new["title"].fillna("") + " " + new["abstract"].fillna("")).str.lower()
    has_out = txt.map(lambda x: has(x, OUTCOME))
    has_ctl = txt.map(lambda x: has(x, CTRL))
    has_set = txt.map(lambda x: has(x, SETW))
    new["passes_stage1"] = ((has_out & has_set) | (has_ctl & has_set))
    new["screen_rule"] = [
        " / ".join(filter(None, ["測定性質+セット" if o and s else "",
                                 "対照+セット" if c and s else ""]))
        for o, c, s in zip(has_out, has_ctl, has_set)
    ]
    new.loc[new["abstract"] == "", "passes_stage1"] = False
    new.loc[new["abstract"] == "", "screen_rule"] = "抄録が取得できず判定不能"

    # 第 1 段を通過したものに、読んで下した判定を当てる。
    # 通過したのに判定が無いレコードがあれば、読み落としなので止める。
    new["verdict"] = [VERDICTS.get(str(i), ("", ""))[0] for i in new["id"]]
    new["verdict_reason"] = [VERDICTS.get(str(i), ("", ""))[1] for i in new["id"]]
    unread = new[new.passes_stage1 & (new.verdict == "")]
    if len(unread):
        print("\n第 1 段を通過したのに判定が記録されていない:")
        for _, r in unread.iterrows():
            print(f"  {r['source']}:{r['id']}  {str(r['title'])[:80]}")
        print("VERDICTS に追記してから通し直す。")
        return 1

    cols = ["seeds", "directions", "source", "id", "year", "journal", "title",
            "passes_stage1", "screen_rule", "verdict", "verdict_reason", "abstract"]
    out = new[cols].sort_values(["passes_stage1", "year"], ascending=[False, False])
    out.to_csv(TABLES / "citation_chase.csv", index=False)

    n_near = int(new.verdict.str.startswith("near-miss").sum())
    counts = [
        ("引用関係として取得したレコード（延べ）", n_raw),
        ("一意のレコード", len(agg)),
        ("うち補遺 S1 の検索で既出", n_known),
        ("検索で出ていない新規レコード", len(agg) - n_known),
        ("うち抄録が取得できず判定不能", n_noabs),
        ("第 1 段の規則を通過（個別に読む対象）", int(new.passes_stage1.sum())),
        ("選定基準をすべて満たした研究", int((new.verdict == "採用").sum())),
        ("基準を満たさないが近い研究（near-miss）", n_near),
    ]
    pd.DataFrame(counts, columns=["stage", "n"]).to_csv(
        TABLES / "citation_chase_summary.csv", index=False)

    print()
    for k, v in counts:
        print(f"  {k:44} {v:5}")
    print()
    hits = new[new.passes_stage1].sort_values("verdict")
    if len(hits):
        print("=== 第 1 段を通過したものと、読んで下した判定 ===")
        for _, r in hits.iterrows():
            print(f"  {r['verdict']:12} [{r['year']}] {str(r['title'])[:78]}")
            print(f"               {r['verdict_reason']}")
    print(f"\n書き出し: {TABLES / 'citation_chase.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
