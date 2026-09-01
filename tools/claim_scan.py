"""主張が結果より強くなりやすい語句を、原稿から全部拾って一覧にする。

なぜ要るか
  2026-08 の改稿は外部レビューを 4 往復した。そのうち最も繰り返したのが
  「非有意から同等を言う」型の誤りで、**1 回目に 9 箇所直したのに、短縮で書いた
  新しい文章に再発した**。同じことが proxy を実測のように書く型でも 2 回、
  absence を断定する型でも 2 回起きた。

  語句は grep で確実に拾える。**拾えないのは判定**である。`did not exceed` が
  観測値の記述なら許容され、主張なら書き換えが要る。だからここでは拾うだけにして、
  判定は skill `claim-discipline` に渡す。ここで自動的に書き換えないのは、
  文脈を見ずに置換すると意味が変わるからである。

拾う 6 型（いずれも今回実際に指摘された）
  equivalence  優越性検定の非有意から同等を言う
  proxy        マーカースコアの代理を実測値のように書く
  absence      検出されなかったことを「存在しない」と断定する
  causal       関連を因果や排他のように書く
  only         例外があるのに「〜のときだけ」と書く
  asymmetry    帰無仮説が違う 2 つの量を同じ土俵で比べる

走査範囲
  既定は**本文だけ**（補遺の前まで）。補遺の感度分析は同じ割合を何度も正当に出すので、
  そこまで拾うと一覧が読めなくなる。補遺も見たいときは --all を付ける。

使い方
  pixi run claim-scan              両方の原稿の本文
  pixi run claim-scan --en         英文の本文だけ
  pixi run claim-scan --all        補遺も含める
  pixi run claim-scan --type equivalence
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MANUSCRIPTS = Path(__file__).resolve().parents[1] / "manuscript"
FILES = [("en", MANUSCRIPTS / "04-preprint-v1-en.md"),
         ("ja", MANUSCRIPTS / "02-投稿原稿_日本語_v1.md")]

# 型ごとの語句と、なぜ危ないかの一行。skill がこの理由を判定の起点に使う。
PATTERNS: dict[str, tuple[str, list[str]]] = {
    "equivalence": (
        "優越性検定が非有意だったことから同等性は言えない。言えるのは「超過を検出できなかった」まで",
        [r"\bdoes not exceed\b", r"\bdid not exceed\b", r"\bno better than\b",
         r"\bcarries no information\b", r"\bindistinguishable from\b",
         r"\bsame reliability\b", r"\bequally well\b", r"\bas well as (?:its|their|the) controls?\b",
         r"対照を上回らな", r"同[じ程]度[にの]", r"情報量を持たない", r"区別できな", r"変わらない"],
    ),
    "proxy": (
        "マーカースコアは実測の細胞分画ではない。値が出る場所では毎回そう書く",
        [r"\bcell (?:fraction|proportion)s?\b", r"\bneutrophil (?:fraction|proportion)\b",
         r"\bcomposition dominates\b", r"\bmeasured composition\b",
         r"細胞(?:比率|分画|割合)", r"組成が支配", r"好中球の割合", r"好中球比率"],
    ),
    "absence": (
        "検出されなかったことと存在しないことは違う。タイトルと見出しで特に出やすい",
        [r"\blacks?\b", r"\bis absent\b", r"\bwas absent\b", r"\bis rare\b", r"\bare rare\b",
         r"\bno evidence of\b", r"\bfound none\b", r"\bnone of the\b",
         r"存在しない", r"欠[くけい]", r"稀である", r"見られない"],
    ),
    "causal": (
        "同じ発現行列から作った量の間の関連は、因果でも排他でもない",
        [r"\bdominated by\b", r"\bdriven by\b", r"\bbecause of the\b", r"\bexplained by\b",
         r"\brather than (?:with )?the annotated\b", r"\bnot the annotated\b",
         r"に起因", r"が原因", r"支配される", r"ではなく注釈"],
    ),
    "only": (
        "「〜のときだけ」は例外がないことを主張する。最も高い区間の値を確かめる",
        [r"\bonly when\b", r"\bonly if\b", r"\balways\b", r"\bnever\b", r"\bin every case\b",
         r"ときだけ", r"に限られる", r"いずれの場合も", r"常に", r"決して"],
    ),
    "asymmetry": (
        "条件効果とコヒーレンスは帰無が違う。同じ土俵で比べたと読める書き方をしない",
        [r"\bcondition effect only\b", r"86\.7%", r"64\.4%",
         r"条件効果のみ"],
    ),
}


SUPP_HEADS = ("## Supplementary Section S1 ", "## 補遺 S1 ")


def body_only(text: str) -> str:
    """補遺の前までを返す。行番号を保つため、以降は空行で埋める。"""
    for key in SUPP_HEADS:
        if key in text:
            i = text.index(key)
            return text[:i]
    return text


def scan(text: str, types: list[str]) -> list[tuple[str, int, str, str]]:
    """(型, 行番号, 語句, 前後の文) を返す。"""
    hits = []
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    def line_of(idx: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    for tname in types:
        _why, pats = PATTERNS[tname]
        for pat in pats:
            for m in re.finditer(pat, text, re.I):
                s = max(0, m.start() - 110)
                e = min(len(text), m.end() + 110)
                ctx = re.sub(r"\s+", " ", text[s:e]).strip()
                hits.append((tname, line_of(m.start()), m.group(0), ctx))
    return hits


def main() -> int:
    args = sys.argv[1:]
    only_lang = next((a[2:] for a in args if a in ("--en", "--ja")), None)
    types = list(PATTERNS)
    if "--type" in args:
        t = args[args.index("--type") + 1]
        if t not in PATTERNS:
            print(f"型は {', '.join(PATTERNS)} のいずれか")
            return 1
        types = [t]

    total = 0
    for lang, path in FILES:
        if only_lang and lang != only_lang:
            continue
        if not path.exists():
            print(f"原稿が見つからない: {path}")
            continue
        text = path.read_text(encoding="utf-8-sig")
        if "--all" not in args:
            text = body_only(text)
        hits = scan(text, types)
        by_type: dict[str, list] = {}
        for h in hits:
            by_type.setdefault(h[0], []).append(h)
        scope = "本文と補遺" if "--all" in args else "本文のみ"
        print(f"=== 主張の強さの走査: {path.name}（{scope}）===")
        if not hits:
            print("  該当なし")
        for tname in types:
            group = by_type.get(tname, [])
            if not group:
                continue
            why, _ = PATTERNS[tname]
            print(f"\n  [{tname}] {len(group)} 件 — {why}")
            for _t, line, word, ctx in sorted(group, key=lambda x: x[1]):
                print(f"    L{line:<5} {word!r}")
                print(f"           ...{ctx[:150]}...")
        total += len(hits)
        print()
    print(f"合計 {total} 件。**これは違反の一覧ではなく、文脈を見る対象の一覧である。**")
    print("判定は skill `claim-discipline` に渡す。観測値の記述なら残し、主張なら書き換える。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
