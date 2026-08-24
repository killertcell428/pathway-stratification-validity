"""AI 検出器のレポートが原稿のどこまでを見たか、節ごとの指摘密度はどうかを出す。

前回 GPTZero に英文原稿をかけたとき、文字数上限のため途中で解析が止まった。
どこまで見たのかを目視で追うのは無理なので、レポートのハイライト文を
原稿本文に突き合わせて節ごとに集計する。

これが要る理由は 2 つ。

1. **未検査の節を特定する。** 見ていない節を「指摘なし」と誤読すると、
   投稿後に初めて問題が出る
2. **無料枠の配分を決める。** GPTZero の無料枠は月 10,000 words で、
   本原稿は 9,649 words。全文を 1 回かけると枠を使い切り、
   修正後の再検査ができない。橙率の高い節から順にかける必要がある

使い方:
  pixi run detector-coverage
  pixi run python -m tools.detector_coverage <pdf> <manuscript.md>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 色の判定は tools/read_detector_report.py に既にあり、検証済み。
# ここで書き直すと閾値がずれる（実際に緑を g > 0.85 と誤って書き、
# 実際の緑 (0.31, 0.79, 0.56) を全部取りこぼした）。必ず再利用する。
from tools.read_detector_report import extract, join_wrapped

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "docs" / "レビュー結果英語.pdf"
DEFAULT_MD = ROOT / "manuscript" / "04-preprint-v1-en.md"

# ハイライトの色。橙 = High AI Impact、緑 = High Human Impact。
ORANGE, GREEN = "橙", "緑"

# 散文でない節。検出器にかけても意味がないので貼り付け用から外す。
NON_PROSE = re.compile(r"^(References|Data and code availability)")


def normalize(s: str) -> str:
    """突合用に正規化する。PDF は行末でハイフン分割や空白の潰れが起きる。"""
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def manuscript_sections(md: Path) -> list[tuple[str, str]]:
    t = md.read_text(encoding="utf-8-sig")
    heads = [(m.start(), m.group(0).lstrip("# ").strip())
             for m in re.finditer(r"^#{1,3} .+$", t, re.M)]
    heads.append((len(t), ""))
    return [(h, t[s:e]) for (s, h), (e, _) in zip(heads, heads[1:]) if h]


def strip_markup(body: str) -> str:
    """検出器に貼る用に、見出し記号・表・強調を落として散文だけにする。

    表や Markdown 記号を貼ると検出器がそこを解析対象外にしたり、
    語数を無駄に消費したりする。無料枠は月 10,000 words しかないので削る。
    """
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s or s.startswith("|") or s.startswith("#") or s.startswith("---"):
            continue
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        s = re.sub(r"\*(.+?)\*", r"\1", s)
        s = re.sub(r"`(.+?)`", r"\1", s)
        lines.append(s)
    return "\n".join(lines)


def emit_chunks(sections: list[tuple[str, str]], names: list[str],
                out_dir: Path, max_chars: int) -> None:
    """指定した節を、1 ファイルが max_chars 未満になるように束ねて書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("chunk-*.txt"):
        f.unlink()

    wanted = [(h, b) for h, b in sections if h in names]
    chunks: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    size = 0
    for h, body in wanted:
        text = strip_markup(body)
        if not text.strip():
            continue
        if size + len(text) > max_chars and cur:
            chunks.append(cur)
            cur, size = [], 0
        cur.append((h, text))
        size += len(text)
    if cur:
        chunks.append(cur)

    print(f"\n=== 検出器に貼る用のファイル（1 件 {max_chars:,} 字未満）===")
    for i, group in enumerate(chunks, 1):
        text = "\n\n".join(t for _, t in group)
        p = out_dir / f"chunk-{i:02d}.txt"
        p.write_text(text, encoding="utf-8")
        heads = " / ".join(h[:34] for h, _ in group)
        print(f"  {p.name}: {len(text.split()):>4} words  {len(text):>6,} chars  {heads}")
    total = sum(len(t.split()) for g in chunks for _, t in g)
    print(f"  合計 {total:,} words（GPTZero 無料枠は月 10,000 words）")


def main() -> int:
    ap = argparse.ArgumentParser()
    # 原稿を分割してかけると 1 節につき 1 レポートになるので、複数を合算できる必要がある。
    ap.add_argument("pdf", nargs="*", default=[str(DEFAULT_PDF)],
                    help="検出器レポート PDF（複数指定で合算する）")
    ap.add_argument("--md", default=str(DEFAULT_MD))
    ap.add_argument("--emit-unseen", metavar="DIR",
                    help="未検査と判定された節を、貼り付け用テキストに書き出す")
    ap.add_argument("--max-chars", type=int, default=8000,
                    help="1 ファイルの上限文字数（既定 8000。無料枠は 1 回 10,000 字）")
    a = ap.parse_args()

    flagged: list[tuple[str, str]] = []
    for pdf in a.pdf:
        rows = join_wrapped(extract(Path(pdf)))
        got = [(c, t) for _, c, t in rows if c in (ORANGE, GREEN)]
        n = sum(1 for c, _ in got if c == ORANGE)
        print(f"  {Path(pdf).name}: 橙 {n:,} / 緑 {len(got) - n:,}")
        flagged.extend(got)
    n_o = sum(1 for c, _ in flagged if c == ORANGE)
    print(f"合計 {len(flagged):,} 行（橙 {n_o:,} / 緑 {len(flagged) - n_o:,} "
          f"= 橙率 {n_o / len(flagged):.0%}）")

    sections = manuscript_sections(Path(a.md))
    # 節本文を正規化して連結し、ハイライト行がどの節に入るかを探す
    norm_sections = [(h, normalize(body), len(body.split())) for h, body in sections]

    hits: dict[str, dict[str, int]] = {h: {ORANGE: 0, GREEN: 0} for h, _, _ in norm_sections}
    unmatched = 0
    for colour, text in flagged:
        n = normalize(text)
        if len(n) < 25:                       # 短すぎる行は誤突合しやすいので捨てる
            continue
        for h, body, _ in norm_sections:
            if n in body:
                hits[h][colour] += 1
                break
        else:
            unmatched += 1

    print(f"原稿に突合できなかった行: {unmatched:,}（表紙・凡例・幻覚スキャン等）\n")
    print(f"{'節':<52}{'words':>7}{'橙':>6}{'緑':>6}{'橙率':>7}  判定")
    print("-" * 88)
    unseen = []
    for h, _, words in norm_sections:
        o, g = hits[h][ORANGE], hits[h][GREEN]
        if o + g == 0:
            verdict = "★未検査"
            unseen.append((h, words))
            rate = "-"
        else:
            rate = f"{o / (o + g):.0%}"
            verdict = "橙が多い" if o / (o + g) >= 0.7 else ""
        print(f"{h[:50]:<52}{words:>7,}{o:>6}{g:>6}{rate:>7}  {verdict}")

    # 語数のわりにハイライトが極端に少ない節は、解析が途中で止まった疑いがある。
    # 前回は §4.3 の途中で文字数上限に達しており、そこから先は指摘なしではなく未解析。
    density = [(h, w, hits[h][ORANGE] + hits[h][GREEN])
               for h, _, w in norm_sections if w >= 150]
    med = sorted((n / w for _, w, n in density))[len(density) // 2] if density else 0
    truncated = [(h, w, n) for h, w, n in density if w >= 250 and n / w < med / 4]

    print()
    if truncated:
        print("★ 語数に対してハイライトが極端に少ない節（解析が途中で止まった疑い）:")
        for h, w, n in truncated:
            print(f"    {h[:56]} ({w:,} words / ハイライト {n} 行)")
    if unseen:
        total = sum(w for _, w in unseen)
        print(f"★ ハイライトが 1 行もない節: {len(unseen)} 件 / 計 {total:,} words")
        for h, w in unseen:
            print(f"    {h[:60]} ({w:,} words)")

    if a.emit_unseen:
        names = [h for h, _ in unseen] + [h for h, _, _ in truncated]
        # 文献一覧は散文ではないので検出器にかけても意味がなく、無料枠を無駄に消費する。
        # 「ハイライトが 0 行」なのは解析漏れではなく、そもそも対象外だからである。
        names = [h for h in names if not NON_PROSE.match(h)]
        emit_chunks(sections, names, Path(a.emit_unseen), a.max_chars)
    return 0


if __name__ == "__main__":
    sys.exit(main())
