"""遺伝子セット（GMT）を Enrichr から取得し、取得日と sha256 を記録する。

遺伝子セットは版が変わると結果が変わるため、取得日の記録が再現性の必須条件になる。
Enrichr 経由であること（MSigDB 原本ではないこと）は論文の Methods に明記する。
"""

from __future__ import annotations

import sys

import requests

from ..common import RAW, load_config, record_provenance


def parse_gmt(text: str) -> dict[str, list[str]]:
    """GMT 文字列を {set_name: [genes]} にする。2 列目の説明は捨てる。"""
    sets: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        name, genes = parts[0], parts[2:]
        # Enrichr の GMT は "GENE,1.0" 形式の重み付き行を含むことがある
        cleaned = sorted({g.split(",")[0].strip().upper() for g in genes if g.strip()})
        if cleaned:
            sets[name] = cleaned
    return sets


def main() -> int:
    cfg = load_config("gene_sets")
    tmpl = cfg["url_template"]
    out_dir = RAW / "gene_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for family, spec in cfg["families"].items():
        lib = spec.get("library")
        if not lib:
            print(f"[{family}] 外部ファイルなし（データから導出する）")
            continue
        url = tmpl.format(library=lib)
        dest = out_dir / f"{family}__{lib}.gmt"
        if dest.exists():
            print(f"[{family}] {lib}: 既に取得済み")
            continue
        print(f"[{family}] {lib} <- {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        dest.write_text(r.text, encoding="utf-8")
        sets = parse_gmt(r.text)
        entry = record_provenance(dest, url, {"n_sets": len(sets), "family": family})
        print(f"         {len(sets)} sets, sha256={entry['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
