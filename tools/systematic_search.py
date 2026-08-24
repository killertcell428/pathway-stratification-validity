"""不在の主張を裏づける系統的検索を PRISMA の手続きで走らせる。

なぜ要るか
  1 節は「遺伝子セットスコアの個人間の共変動と反復測定信頼性を、対照を基準として
  測った研究は見当たらない」と述べている。これは不在の主張なので、検索の網羅性が
  そのまま主張の強さになる。従来は PubMed と Europe PMC を場当たりに引いただけで、
  検索式も段階別件数も記録しておらず、プレプリントも見ていなかった。
  そのため 1 節は「この検索の範囲に限定される」と自分で限定をかけていた。

  ここでは検索式・データベース・日付・選定基準・段階別件数をすべて記録する。
  査読者が同じ検索を再実行できる状態にするのが目的で、
  「探したが無かった」を「この検索式でこの件数を見て無かった」に変える。

選定基準
  含める  検体ごとに算出した注釈遺伝子セットのスコアについて、
          (a) 構成遺伝子の個人間の共変動、または
          (b) 同一個人の反復測定に対するスコアの信頼性
          のいずれかを定量し、**サイズをそろえたランダム遺伝子セット等の対照と比較**
          している研究
  除く    E1 群間比較の妥当性のみを扱う（個人の順位づけを扱わない）
          E2 手法ベンチマークで、変数が手法でありセットの出自ではない
          E3 「安定性」が入力条件への頑健性（標本サイズ・欠損遺伝子・DB サイズ）で、
             同一個人の再測定ではない
          E4 単一細胞でスコアの細胞間安定性を見ており、個人間ではない
          E5 遺伝子セットスコアを扱っていない（検索語の偶然一致）

出力
  results/tables/systematic_search_queries.csv   検索式ごとの件数（再現用）
  results/tables/systematic_search_records.csv   重複除去後の全レコード（スクリーニング用）
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from ._search_blocks import queries

PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UA = {"User-Agent": "T26-systematic-search (mailto:noreply@example.org)"}
MAX_PER_QUERY = 3000       # 1 検索式あたりの取得上限。
# 上限 1000 のときは S7 が PubMed 1,473→999、Europe PMC 1,624→1,000 で切れ、
# ヒット 5,717 件のうち 1,099 件（19.2%）が未取得だった。しかも取得された側は
# API の既定ソート順の上位なので無作為標本ではない。**不在の主張を掲げる論文で
# identification から retrieval への 19% 脱落を理由なしに残すのは通らない。**
# S7 の最大ヒット 1,624 件を完全に覆う値に上げた。

# 検索式は tools/_search_blocks.py の概念ブロックから生成する。
# 両 DB を同じ概念で引いていることを保証するため、文字列は手書きしない。
QUERIES = queries()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def pubmed_search(term: str) -> tuple[int, list[str]]:
    q = urllib.parse.quote(term)
    raw = fetch(f"{PUBMED}/esearch.fcgi?db=pubmed&retmax={MAX_PER_QUERY}&term={q}").decode()
    count = int(re.search(r"<Count>(\d+)</Count>", raw).group(1))
    return count, re.findall(r"<Id>(\d+)</Id>", raw)


def pubmed_records(pmids: list[str]) -> list[dict]:
    """タイトルと抄録を取る。スクリーニングは抄録まで読んで行う。"""
    out = []
    for i in range(0, len(pmids), 150):
        chunk = pmids[i:i + 150]
        raw = fetch(f"{PUBMED}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(chunk)}").decode(
            "utf-8", errors="replace")
        for art in raw.split("<PubmedArticle>")[1:]:
            pmid = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            title = re.search(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S)
            abst = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
            # PubMed XML は PMID の直後に DateCompleted / DateRevised が来るので、
            # 最初の <Year> を取ると刊行年ではなく改訂年になる（Scheid 2018 が 2019、
            # Venet 2011 が 2012 と出た）。PubDate 配下の Year を明示的に取る。
            year = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S)
            jour = re.search(r"<Title>(.*?)</Title>", art, re.S)
            strip = lambda s: re.sub(r"<[^>]+>", "", s).strip() if s else ""
            out.append({
                "source": "PubMed", "id": pmid.group(1) if pmid else "",
                "title": strip(title.group(1) if title else ""),
                "abstract": strip(abst), "year": year.group(1) if year else "",
                "journal": strip(jour.group(1) if jour else ""),
            })
        time.sleep(0.4)
    return out


def epmc_search(term: str, preprints_only: bool = False) -> tuple[int, list[dict]]:
    q = term + (" AND SRC:PPR" if preprints_only else "")
    recs, cursor, total = [], "*", None
    while True:
        url = (f"{EPMC}?query={urllib.parse.quote(q)}&format=json&pageSize=100"
               f"&cursorMark={urllib.parse.quote(cursor)}&resultType=core")
        d = json.loads(fetch(url))
        if total is None:
            total = int(d.get("hitCount", 0))
        for r in d.get("resultList", {}).get("result", []):
            recs.append({
                "source": "Europe PMC (preprint)" if preprints_only else "Europe PMC",
                "id": r.get("doi") or r.get("id", ""),
                "title": (r.get("title") or "").strip().rstrip("."),
                "abstract": (r.get("abstractText") or "").strip(),
                "year": str(r.get("pubYear", "")), "journal": r.get("journalTitle", "") or "preprint",
            })
        nxt = d.get("nextCursorMark")
        if not nxt or nxt == cursor or len(recs) >= min(total, MAX_PER_QUERY):
            break
        cursor = nxt
        time.sleep(0.3)
    return total, recs


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())[:90]


def main() -> int:
    from pathlib import Path
    tables = Path(__file__).resolve().parents[1] / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    from datetime import date
    search_date = date.today().isoformat()
    print(f"最終検索日: {search_date}")
    qrows, all_recs = [], []
    for key, spec in QUERIES.items():
        print(f"\n[{key}] {spec['狙い']}")
        n_pm, pmids = pubmed_search(spec["pubmed"])
        print(f"  PubMed              {n_pm:5d} 件（取得 {len(pmids)}）")
        recs = pubmed_records(pmids)
        n_ep, r_ep = epmc_search(spec["epmc"])
        print(f"  Europe PMC          {n_ep:5d} 件（取得 {len(r_ep)}）")
        n_pp, r_pp = epmc_search(spec["epmc"], preprints_only=True)
        print(f"  Europe PMC プレプリント {n_pp:5d} 件（取得 {len(r_pp)}）")
        for src, n, got in (("PubMed", n_pm, len(recs)), ("Europe PMC", n_ep, len(r_ep)),
                            ("Europe PMC (preprint)", n_pp, len(r_pp))):
            qrows.append({"query": key, "狙い": spec["狙い"], "database": src,
                          "hits": n, "retrieved": got, "search_date": search_date,
                          "blocks": " AND ".join(spec["blocks"]),
                          "search_string": spec["pubmed" if src == "PubMed" else "epmc"]})
        for r in recs + r_ep + r_pp:
            r["query"] = key
            all_recs.append(r)
        time.sleep(0.5)

    q = pd.DataFrame(qrows)
    q.to_csv(tables / "systematic_search_queries.csv", index=False, encoding="utf-8")

    df = pd.DataFrame(all_recs)
    df["key"] = df.title.map(norm_title)
    before = len(df)
    # 同一文献が複数の検索式・複数 DB で出る。どの検索式で当たったかは残す。
    hits_by = df.groupby("key")["query"].apply(lambda s: "+".join(sorted(set(s))))
    src_by = df.groupby("key")["source"].apply(lambda s: "+".join(sorted(set(s))))
    df = df.drop_duplicates("key").set_index("key")
    df["found_in"] = hits_by
    df["sources"] = src_by
    df = df.reset_index(drop=True).drop(columns=["query", "source"])
    df.to_csv(tables / "systematic_search_records.csv", index=False, encoding="utf-8")

    print(f"\n=== PRISMA 識別段階 ===")
    print(f"  検索式 x DB の組み合わせ: {len(q)}")
    print(f"  ヒット総数（重複含む）: {int(q.hits.sum()):,}")
    print(f"  取得したレコード: {before:,}")
    print(f"  重複除去後: {len(df):,}")
    print(f"\n  DB 別の内訳（重複除去後）:")
    for s, n in df.sources.value_counts().items():
        print(f"    {s:34s} {n:4d}")
    print(f"\n-> {tables/'systematic_search_queries.csv'}")
    print(f"-> {tables/'systematic_search_records.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
