"""リポジトリを公開する前に、公開してはいけないものが混ざっていないかを機械で調べる。

なぜ要るか
  この研究は本業のクライアント案件（ORI の FPP 研究）から派生している。派生元の
  機密情報、勤務先・顧客の名前、内部文書、生データ、ローカルの絶対パス、資格情報が
  1 つでも混ざったまま public に切り替えると、取り消せない。GitHub は公開後の
  コミット履歴がフォークやキャッシュに残るため、あとで消しても消えない。

  目視は当てにならない。ファイル数が多く、履歴も含めて見る必要があるので、
  公開切替の前に必ずこれを通す。**1 件でも出たら公開しない。**

  なお読み取り専用である。このスクリプトは何も書き換えず、何も送信しない。

使い方:
  pixi run prepub-scan
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 検出したいもの。(分類, 正規表現, なぜ駄目か)
# 「たまたま一致する」を避けるため、語としての境界を意識して書く。
PATTERNS: list[tuple[str, str, str]] = [
    ("クライアント",
     r"ORI研究所|ORI\s*研究所|\bFPP\b|myo-?inositol|ミオイノシトール|イノシトール",
     "派生元のクライアント案件を特定できる語。公開データのみの研究として独立させている"),
    ("クライアント",
     r"クライアント|顧客|受託案件|請負|コンサル案件",
     "案件由来であることを示す語"),
    ("勤務先",
     r"株式会社シトラ|シトラ|Cytra(?!Ueda)|CYTRA",
     "勤務先名。所属は独立研究者で出すので原稿にもコードにも要らない"),
    ("勤務先",
     r"勤務先|上司|本業|社内|弊社|自社",
     "雇用関係を示す語。利益相反の開示は原稿の該当節だけで完結させる"),
    ("個人情報",
     r"年収|給与|副業届|ココナラ|coconala",
     "研究と無関係な個人情報"),
    ("ローカルパス",
     r"[A-Za-z]:\\\\Users\\\\|[A-Za-z]:\\Users\\|/home/[a-z]+/|/c/Users/",
     "実行環境の絶対パス。ユーザー名が露出し、再現もできない"),
    ("資格情報",
     r"(?i)(api[_-]?key|secret[_-]?key|password|passwd|bearer\s+[A-Za-z0-9._-]{16,})\s*[:=]",
     "資格情報らしき代入"),
    ("資格情報",
     r"gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}",
     "トークンの形をした文字列"),
]

# 走査しない場所。生成物・環境・キャッシュ。
SKIP_DIRS = {".pixi", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
             "node_modules", "raw", "interim"}
# 中身を読まないファイル（二値）。存在自体は別に報告する。
BINARY_SUFFIX = {".png", ".svg", ".pdf", ".docx", ".xlsx", ".parquet", ".gz",
                 ".zip", ".h5ad", ".rds", ".mtx", ".ico", ".woff", ".woff2"}
# 公開してはいけない実データの置き場
DATA_DIRS = ["data/raw", "data/interim"]


# 一次記録に含まれる誤検出。PubMed から取った抄録に検出語が出てくるだけで、
# 削れば検索の記録が壊れる。ファイル単位で除外する。
FALSE_POSITIVE_FILES = {
    "results/tables/systematic_search_records.csv":
        "PubMed の抄録に検出語が現れるだけ。検索の一次記録なので改変しない",
}


def _ignored(rel: str, rules: list[str]) -> bool:
    """.gitignore の単純な書き方だけを解釈して、公開対象外かを返す。

    完全な gitignore 実装はしない。このリポジトリで使っているのは
    「ディレクトリ/」「拡張子のワイルドカード」「パスそのもの」「末尾の *」だけなので、
    それだけを見る。判定を強めるより、判定が緩んだときに気づける形にしてある
    （公開対象と判定されたファイルは全部走査されるので、緩い側に倒しても
     検出漏れにはならない）。
    """
    from fnmatch import fnmatch

    def _hit(rel: str, r: str) -> bool:
        if r.endswith("/"):
            return rel == r[:-1] or rel.startswith(r)
        if fnmatch(rel, r) or fnmatch(rel, r + "*"):
            return True
        # ディレクトリ配下をまとめて指す書き方
        return "/" not in r and any(fnmatch(part, r) for part in rel.split("/"))

    # git と同じく、最後に一致した規則が勝つ。`!` は除外の打ち消し。
    # ここを実装しないと `docs/*` + `!docs/01-*` を全部除外と読んで、
    # 実際には公開されるファイルを走査しないまま「検出なし」と報告してしまう。
    # 一度それをやった。検査が緩い側に外れるのが最悪なので、ここは正確に書く。
    ignored = False
    for raw in rules:
        r = raw.strip()
        if not r or r.startswith("#"):
            continue
        neg = r.startswith("!")
        if neg:
            r = r[1:]
        if _hit(rel, r):
            ignored = not neg
    return ignored


def _publishable(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """(公開対象, ローカルのみ) に分ける。"""
    gi = ROOT / ".gitignore"
    rules = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    pub, local = [], []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        (local if _ignored(rel, rules) else pub).append(f)
    return pub, local


def _walk() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        out.append(p)
    return out


def main() -> int:
    all_files = _walk()
    files, local_only = _publishable(all_files)
    compiled = [(cat, re.compile(rx), why) for cat, rx, why in PATTERNS]
    hits: list[tuple[str, str, int, str, str]] = []
    scanned = skipped_binary = 0

    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        if f.suffix.lower() in BINARY_SUFFIX:
            skipped_binary += 1
            continue
        try:
            text = f.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            skipped_binary += 1
            continue
        scanned += 1
        # 自分自身は検出語の一覧を持っているので除く
        if rel == "tools/prepublication_scan.py":
            continue
        if rel in FALSE_POSITIVE_FILES:
            continue
        for i, line in enumerate(text.split("\n"), 1):
            for cat, rx, why in compiled:
                m = rx.search(line)
                if m:
                    hits.append((cat, rel, i, m.group(0)[:40], line.strip()[:110]))

    print("=== 公開前スキャン ===")
    print(f"公開対象 {len(files)} ファイル / ローカルのみ {len(local_only)} ファイル"
          f"（.gitignore で除外）")
    print(f"うち本文を走査 {scanned} 件"
          f"（二値・読めないもの {skipped_binary} 件は本文を見ていない）")
    if FALSE_POSITIVE_FILES:
        print("誤検出として除外したファイル:")
        for k, why in FALSE_POSITIVE_FILES.items():
            print(f"  {k} — {why}")

    # 生データが公開対象に入っていないかを、宿題ではなく判定として出す
    print("\n--- 実データの置き場 ---")
    gi = ROOT / ".gitignore"
    rules = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    data_bad = 0
    for d in DATA_DIRS:
        dp = ROOT / d
        n = sum(1 for _ in dp.rglob("*") if _.is_file()) if dp.exists() else 0
        if not n:
            print(f"  {d}: 空")
            continue
        exposed = [x for x in dp.rglob("*") if x.is_file()
                   and not _ignored(x.relative_to(ROOT).as_posix(), rules)]
        if exposed:
            data_bad += len(exposed)
            print(f"  {d}: {n} ファイルのうち **{len(exposed)} 件が公開対象**")
            for x in exposed[:5]:
                print(f"      {x.relative_to(ROOT).as_posix()}")
        else:
            print(f"  {d}: {n} ファイル（全件 .gitignore で除外済み）")

    print("\n--- 検出語 ---")
    if not hits:
        print("  なし")
    else:
        by_cat: dict[str, list] = {}
        for cat, rel, i, tok, line in hits:
            by_cat.setdefault(cat, []).append((rel, i, tok, line))
        for cat, rows in by_cat.items():
            why = next(w for c, _, w in PATTERNS if c == cat)
            print(f"\n  [{cat}] {len(rows)} 件 — {why}")
            for rel, i, tok, line in rows[:12]:
                print(f"    {rel}:{i}  «{tok}»")
                print(f"      {line}")
            if len(rows) > 12:
                print(f"    ... ほか {len(rows) - 12} 件")

    print()
    if hits or data_bad:
        print(f"★ 検出語 {len(hits)} 件 / 公開対象に入った実データ {data_bad} 件。**公開しない。**"
              f"文言を直すか、.gitignore で公開対象から外すか、"
              f"誤検出なら FALSE_POSITIVE_FILES に理由を書いて除外する。")
        return 1
    print("検出なし。内容面の公開条件は満たしている。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
