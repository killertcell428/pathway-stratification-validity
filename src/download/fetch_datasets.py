"""config/datasets.yml に書かれた公開データを取得し、来歴を記録する。

データ本体は Git に載せない。載せるのは「どこから・いつ・どのハッシュのものを取ったか」だけ。
既に同じサイズのファイルがあれば再取得しない（サイズ一致でも中身が違う可能性は sha256 で検出する）。
"""

from __future__ import annotations

import sys

import json

import requests

from ..common import CHECKSUMS, RAW, load_config, record_provenance


def already_recorded(dest_name: str) -> bool:
    ledger = CHECKSUMS / "provenance.jsonl"
    if not ledger.exists():
        return False
    with ledger.open(encoding="utf-8") as f:
        return any(json.loads(line).get("file") == dest_name for line in f if line.strip())


def fetch(url: str, dest_name: str) -> None:
    dest = RAW / dest_name
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    remote_size = int(head.headers.get("content-length", 0))

    if dest.exists() and remote_size and dest.stat().st_size == remote_size:
        print(f"  [skip] {dest_name} ({remote_size/1e6:.1f}MB, 既に取得済み)")
        # 手で置いたファイルにも来歴を残す（記録がないと第三者が同一物か確認できない）
        if not already_recorded(dest_name):
            entry = record_provenance(dest, url, {"note": "既存ファイルに来歴を後付け"})
            print(f"         sha256={entry['sha256'][:16]}... を記録した")
        return

    print(f"  [get ] {dest_name} ({remote_size/1e6:.1f}MB) <- {url}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    entry = record_provenance(dest, url)
    print(f"         sha256={entry['sha256'][:16]}...")


def main() -> int:
    cfg = load_config("datasets")
    for section in ("discovery", "validation"):
        for key, ds in (cfg.get(section) or {}).items():
            if str(ds.get("status", "")).startswith("未組み込み"):
                print(f"[{section}/{key}] {ds['accession']}: 未組み込みのためスキップ")
                continue
            print(f"[{section}/{key}] {ds['accession']}")
            for label, url in ds["urls"].items():
                fetch(url, url.rsplit("/", 1)[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
