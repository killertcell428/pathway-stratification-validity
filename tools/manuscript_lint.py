"""原稿の構造が壊れていないかを機械で確かめる。

なぜ要るか
  `audit` は原稿の数値が解析出力と一致するかを見て、`qc_tables` は表と図の数値に
  裏づけがあるかを逆向きに見る。どちらも 500 件超を照合して通っていたのに、
  2026-08 の改稿では**数値以外の構造**が繰り返し壊れた。実際に起きたもの:

    図番号が 1,2,3,5 に飛んだ            2 回（図を補遺に移したとき）
    表番号が 1,2,2c,3c,5 になった        1 回（短縮で表を移したとき）
    存在しない表への参照が残った          14 箇所
    補遺の見出しが重複した               2 組（S2/S17、S10/S18）
    補遺の順序が乱れた                   1 回（和文が S12,S15,S16,S17,S14,S13）
    本文が丸ごと複製された               1 回（105,585 文字。t.index の範囲逆転）
    Abstract が語数上限を超えた           3 回（表現を直すたびに）
    引用番号が衝突した                   1 回（6 件追加時）

  いずれも読めば分かるが、改稿のたびに人が全部を目で追うのは続かない。
  機械が見れば確実に落ちるので、ここで落とす。

  **数値は見ない。**それは audit と qc_tables の仕事である。ここは番号・参照・
  順序・重複・語数という構造だけを見る。

使い方
  pixi run lint            両方の原稿を検査する
  pixi run lint --en       英文だけ
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANUSCRIPTS = Path(__file__).resolve().parents[1] / "manuscript"

# 投稿規定ではなく docs/05 の自主目標。超えたら削る合図。
ABSTRACT_LIMIT = 300


@dataclass
class Spec:
    """原稿ごとの見出しと書式の違いを 1 箇所に集める。"""

    path: Path
    label: str
    abstract_head: str
    next_after_abstract: str
    supp_head: str          # 補遺の見出しの前置き（"## Supplementary Section " など）
    refs_head: str
    table_word: str         # "Table" / "表"
    figure_word: str        # "Figure" / "図"
    supp_figure_word: str   # "Supplementary Figure" / "補遺図"
    section_ref: str        # 節参照の書き方（"section" / "節"）


SPECS = [
    Spec(MANUSCRIPTS / "04-preprint-v1-en.md", "en",
         "## Abstract", "## 1.", "## Supplementary Section ", "## References",
         "Table", "Figure", "Supplementary Figure", "section"),
    Spec(MANUSCRIPTS / "02-投稿原稿_日本語_v1.md", "ja",
         "## 要旨", "## 1.", "## 補遺 ", "## 引用文献",
         "表", "図", "補遺図", "節"),
]


@dataclass
class Result:
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def _split(text: str, spec: Spec) -> tuple[str, str]:
    """本文と補遺に切る。補遺が無ければ参考文献の前までを本文とする。"""
    for key in (spec.supp_head + "S1 ", spec.refs_head):
        if key in text:
            i = text.index(key)
            return text[:i], text[i:]
    return text, ""


def check_duplicate_headings(text: str, r: Result) -> None:
    """同じ見出しが 2 回出たら、たいてい文書が複製されている。"""
    heads = re.findall(r"^(#{1,3} .+)$", text, re.M)
    seen: dict[str, int] = {}
    for h in heads:
        seen[h] = seen.get(h, 0) + 1
    for h, n in seen.items():
        if n > 1:
            r.err(f"見出しが {n} 回ある（文書の複製を疑う）: {h[:70]}")


def check_numbering(body: str, word: str, kind: str, r: Result,
                    exclude: set[str] | None = None, in_image: bool = False) -> list[str]:
    """本文の図表番号が 1 から連番かを見る。M1 のような接頭辞つきは別枠。

    表は行頭の `**Table 1** ...`、図は `![**Figure 1** ...](path)` と書式が違う。
    """
    pat = (rf"!\[\*\*{re.escape(word)} ([\w.]+)\*\*" if in_image
           else rf"^\*\*{re.escape(word)} ([\w.]+)\*\*")
    found = re.findall(pat, body, 0 if in_image else re.M)
    found = [x for x in found if not (exclude and x in exclude)]
    plain = [x for x in found if x.isdigit()]
    other = [x for x in found if not x.isdigit()]
    if other:
        r.note(f"{kind}に数字以外の番号がある（意図的なら可）: {', '.join(other)}")
    nums = [int(x) for x in plain]
    if nums != list(range(1, len(nums) + 1)):
        r.err(f"{kind}の番号が 1 からの連番でない: {nums}")
    return found


def check_references_exist(body: str, supp: str, spec: Spec, r: Result,
                           defined_tables: list[str], defined_figs: list[str]) -> None:
    """本文が参照している図表・節・補遺が実在するかを見る。"""
    w, f = re.escape(spec.table_word), re.escape(spec.figure_word)
    # 本文が指す表番号（S つきは補遺の表）
    for m in re.finditer(rf"{w} (S?\d+[a-z]?)(?![\w.])", body):
        num = m.group(1)
        if num.startswith("S"):
            if f"**{spec.table_word} {num}**" not in supp:
                r.err(f"本文が補遺の {spec.table_word} {num} を指しているが、補遺に定義がない")
        elif num not in defined_tables:
            r.err(f"本文が {spec.table_word} {num} を指しているが、本文に定義がない")
    for m in re.finditer(rf"(?<!{re.escape(spec.supp_figure_word)} ){f} (S?\d+)(?![\w.])", body):
        num = m.group(1)
        if not num.startswith("S") and num not in defined_figs:
            r.err(f"本文が {spec.figure_word} {num} を指しているが、本文に定義がない")
    # 節参照（3.x）が実在するか
    defined_sections = set(re.findall(r"^### (\d+\.\d+)", body, re.M))
    pat = rf"{re.escape(spec.section_ref)} (\d+\.\d+)" if spec.label == "en" else r"(\d+\.\d+) 節"
    for m in re.finditer(pat, body + supp):
        s = m.group(1)
        if s not in defined_sections and not s.startswith(("1.", "2.", "4.", "5.")):
            r.err(f"{s} 節への参照があるが、その節が本文にない")
    # 補遺参照
    defined_supp = set(re.findall(re.escape(spec.supp_head) + r"(S\d+)", supp))
    for m in re.finditer(r"(?:Supplementary Section|補遺) (S\d+)", body + supp):
        s = m.group(1)
        if s not in defined_supp:
            r.err(f"補遺 {s} への参照があるが、その節が存在しない")


def check_supplement_order(supp: str, spec: Spec, r: Result) -> None:
    """補遺が S1 から連番で、順序どおりに並び、見出しが重複していないか。"""
    ids = re.findall(re.escape(spec.supp_head) + r"S(\d+)", supp)
    nums = [int(x) for x in ids]
    if not nums:
        r.err("補遺が 1 つも見つからない")
        return
    if nums != sorted(nums):
        r.err(f"補遺の並び順が番号順でない: {['S' + str(n) for n in nums]}")
    if nums != list(range(1, len(nums) + 1)):
        r.err(f"補遺の番号が S1 からの連番でない: {['S' + str(n) for n in nums]}")
    titles = re.findall(re.escape(spec.supp_head) + r"S\d+ (.+)$", supp, re.M)
    norm = [t.strip().rstrip("。.").lower() for t in titles]
    for i, t in enumerate(norm):
        for j in range(i + 1, len(norm)):
            # 一方が他方の先頭に含まれる＝実質同じ見出し
            if t and (t == norm[j] or t.startswith(norm[j]) or norm[j].startswith(t)):
                r.err(f"補遺の見出しが重複している: S{nums[i]} と S{nums[j]}「{titles[i][:44]}」")


def check_abstract(text: str, spec: Spec, r: Result) -> None:
    """要旨の語数。数え方を固定する（ハイフン語は 1 語）。"""
    if spec.abstract_head not in text:
        r.err("要旨の見出しが見つからない")
        return
    i = text.index(spec.abstract_head)
    j = text.index(spec.next_after_abstract, i)
    ab = text[i:j]
    for cut in ("**Keywords**", "**キーワード**"):
        if cut in ab:
            ab = ab[: ab.index(cut)]
    ab = ab.replace(spec.abstract_head, "").strip()
    ab = re.sub(r"\*\*([^*]+)\*\*", r"\1", ab)
    if spec.label == "en":
        n = len(re.findall(r"[A-Za-z][A-Za-z\-’']*", ab))
        if n > ABSTRACT_LIMIT:
            r.err(f"要旨が {n} words で上限 {ABSTRACT_LIMIT} を超えている")
        else:
            r.note(f"要旨 {n} words（上限 {ABSTRACT_LIMIT}）")
        bad = {ch for ch in ab if ord(ch) > 127}
        if bad:
            r.note("要旨に非 ASCII 文字がある（投稿フォームで化ける恐れ）: "
                   + " ".join(f"{ch!r}(U+{ord(ch):04X})" for ch in sorted(bad)))
    else:
        chars = len(re.sub(r'\s', '', ab))
        r.note(f'要旨 {chars:,} 字')


def check_citations(text: str, spec: Spec, r: Result) -> None:
    """引用番号が 1 から連番で、本文から全部引かれているか。"""
    if spec.refs_head not in text:
        r.err("参考文献の見出しが見つからない")
        return
    body, refs = text[: text.index(spec.refs_head)], text[text.index(spec.refs_head):]
    defined = [int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", refs, re.M)]
    if defined != list(range(1, len(defined) + 1)):
        r.err(f"参考文献の番号が 1 からの連番でない: {defined}")
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", body)}
    unused = [n for n in defined if n not in cited]
    if unused:
        r.err(f"本文から引かれていない引用がある: {unused}")
    dangling = sorted(n for n in cited if n not in defined)
    if dangling:
        r.err(f"本文が存在しない引用番号を指している: {dangling}")
    r.note(f"引用 {len(defined)} 件、すべて本文から参照")


def lint(spec: Spec) -> Result:
    r = Result()
    if not spec.path.exists():
        r.err(f"原稿が見つからない: {spec.path}")
        return r
    text = spec.path.read_text(encoding="utf-8-sig")
    body, supp = _split(text, spec)

    check_duplicate_headings(text, r)
    tables = check_numbering(body, spec.table_word, "本文の表", r, exclude={"M1"})
    figs = check_numbering(body, spec.figure_word, "本文の図", r, in_image=True)
    # 参照の実在確認には接尾つき（2c など）も含めて渡す
    check_references_exist(body, supp, spec, r, tables, figs)
    check_supplement_order(supp, spec, r)
    check_abstract(text, spec, r)
    check_citations(text, spec, r)
    return r


def main() -> int:
    only = None
    for a in sys.argv[1:]:
        if a in ("--en", "--ja"):
            only = a[2:]
    specs = [s for s in SPECS if only is None or s.label == only]

    bad = 0
    for spec in specs:
        r = lint(spec)
        print(f"=== 構造の検査: {spec.path.name} ===")
        for n in r.notes:
            print(f"  - {n}")
        if r.errors:
            bad += len(r.errors)
            print(f"\n  【要修正 {len(r.errors)} 件】")
            for e in r.errors:
                print(f"    x {e}")
        else:
            print("  構造の問題は見つからなかった")
        print()
    if bad:
        print(f"要修正 {bad} 件")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
