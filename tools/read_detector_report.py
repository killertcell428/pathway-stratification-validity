"""AI 検出器（GPTZero 等）が出力した PDF から、指摘箇所を色ごとに取り出す。

検出器のレポートは「どの文が AI 生成らしいか」をハイライトの色で区別している。
色を無視して本文だけ読むと、どの文が指摘されたのか分からない。

GPTZero の PDF は注釈ではなく、文字の背後に塗りつぶし矩形を描いて色を付ける。
そのため注釈 API では取れない。ここでは塗りつぶし矩形と本文行の重なりから、
1 文ごとに色を割り当てる。

使い方:
  pixi run python -m tools.read_detector_report docs/レビュー結果.pdf
  pixi run python -m tools.read_detector_report docs/レビュー結果.pdf --color 緑
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

# 検出器が使う塗り色。丸めた RGB で引く。
PALETTE = {
    (0.96, 0.75, 0.31): "橙",
    (0.31, 0.79, 0.56): "緑",
}
IGNORE = {(1.0, 1.0, 1.0), (0.97, 0.97, 0.97), (0.0, 0.0, 0.0), (0.23, 0.23, 0.23)}


def page_color_rects(page) -> list[tuple[str, pymupdf.Rect]]:
    out = []
    for d in page.get_drawings():
        f = d.get("fill")
        if not f:
            continue
        key = tuple(round(v, 2) for v in f)
        if key in IGNORE:
            continue
        name = PALETTE.get(key)
        if name is None:
            continue
        out.append((name, d["rect"]))
    return out


def line_color(rects: list[tuple[str, pymupdf.Rect]], bbox: pymupdf.Rect) -> str:
    """行に掛かっている塗り色を返す。掛かっていなければ「なし」。

    塗り矩形は行の高さ（16pt）で敷き詰められており、行の bbox とは上下に少しずれる。
    面積の重なりで判定すると、隣の行の矩形を拾って色が入れ替わる。行の中心点が
    どの矩形の内側にあるかで決めるほうが確実。
    行の途中で色が変わる場合があるので、水平方向は複数点を見て多数決にする。
    """
    cy = (bbox.y0 + bbox.y1) / 2
    votes: Counter[str] = Counter()
    for k in range(1, 10):
        px = bbox.x0 + (bbox.x1 - bbox.x0) * k / 10
        for name, r in rects:
            if r.x0 <= px <= r.x1 and r.y0 <= cy <= r.y1:
                votes[name] += 1
                break
    return votes.most_common(1)[0][0] if votes else "なし"


def extract(pdf: Path, skip_first: int = 1) -> list[tuple[int, str, str]]:
    """(ページ番号, 色, 文) の一覧。先頭のサマリーページは飛ばす。"""
    doc = pymupdf.open(pdf)
    rows: list[tuple[int, str, str]] = []
    for i in range(skip_first, doc.page_count):
        page = doc[i]
        rects = page_color_rects(page)
        if not rects:
            continue
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line["spans"]).strip()
                if not text:
                    continue
                bbox = pymupdf.Rect(line["bbox"])
                rows.append((i + 1, line_color(rects, bbox), text))
    return rows


def join_wrapped(rows: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    """行末のハイフンで折り返された行を 1 文につなぐ。"""
    out: list[list] = []
    for page, color, text in rows:
        if out and out[-1][2].endswith("-"):
            out[-1][2] = out[-1][2][:-1] + text
            if out[-1][1] == "なし":
                out[-1][1] = color
        else:
            out.append([page, color, text])
    return [(p, c, t) for p, c, t in out]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--color", default=None, help="この色の文だけ出す（橙 / 緑 / なし）")
    ap.add_argument("--min-chars", type=int, default=0, help="この文字数未満の行は出さない")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    rows = join_wrapped(extract(args.pdf))
    # ヘッダ・フッタなど本文でない行を落とす
    noise = re.compile(r"^(Version |02-投稿原稿|Page \d|FAQs$|GPTZero)")
    rows = [r for r in rows if not noise.match(r[2])]

    tally = Counter(c for _, c, _ in rows)
    print(f"=== {args.pdf.name}: 文の色分け ===")
    for name, n in tally.most_common():
        print(f"  {name:4s} {n:4d} 文  ({n / len(rows):.0%})")
    print()

    shown = [r for r in rows
             if (args.color is None or r[1] == args.color) and len(r[2]) >= args.min_chars]
    for page, color, text in shown:
        print(f"[p.{page:>2} {color}] {text}")
    print(f"\n表示 {len(shown)} / 全 {len(rows)} 文")
    return 0


if __name__ == "__main__":
    sys.exit(main())
