"""パス・設定・乱数・チェックサムの共通処理。

設定値をコード側にハードコードしないための唯一の入口。config/*.yml を読むのは
このモジュール経由に限る。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"

# データセット名前空間（WP1: RNA-seq 再現用）。
# 環境変数 T26_DATASET を設定すると、interim / metadata / tables / figures が
# その名前のサブディレクトリに切り替わる。解析コード側は common の定数を使うだけ
# なので、同じコードを別コホートの入力で再実行できる（コード変更ゼロが WP1 の要件）。
_NS = os.environ.get("T26_DATASET", "").strip()


def _ns(base: Path) -> Path:
    return base / _NS if _NS else base


INTERIM = _ns(DATA / "interim")
METADATA = _ns(DATA / "metadata")
CHECKSUMS = DATA / "checksums"          # 来歴はコホート横断で 1 冊
RESULTS = ROOT / "results"
TABLES = _ns(RESULTS / "tables")
FIGURES = _ns(RESULTS / "figures")

for _d in (RAW, INTERIM, METADATA, CHECKSUMS, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# 発現行列の接尾辞（発現フィルタ閾値の感度分析用）。
# T26_MATRIX_SUFFIX=_cpm0.5 のとき expr_NI_cpm0.5.parquet を読み書きする。
# 名前空間と同じ理由でここに置く: 閾値を変えた再走行で解析コードを触らないため。
MATRIX_SUFFIX = os.environ.get("T26_MATRIX_SUFFIX", "").strip()


def expr_path(cond: str) -> Path:
    """条件別の gene x individual 行列のパス。"""
    return INTERIM / f"expr_{cond}{MATRIX_SUFFIX}.parquet"


def gene_mean_path() -> Path:
    """安静時の平均発現量（対照のマッチングに使う）のパス。"""
    return INTERIM / f"gene_expression_naive{MATRIX_SUFFIX}.csv"


def modules_path() -> Path:
    """データ由来共発現モジュールの GMT のパス。"""
    return INTERIM / f"data_derived_modules{MATRIX_SUFFIX}.gmt"


def load_config(name: str) -> dict:
    """config/<name>.yml を読む。

    名前空間が有効で config/<name>_<namespace>.yml が存在する場合はそちらを優先する
    （例: T26_DATASET=gse81046 のとき analysis → analysis_gse81046.yml）。
    """
    if _NS:
        override = CONFIG / f"{name}_{_NS}.yml"
        if override.exists():
            with override.open(encoding="utf-8") as f:
                return yaml.safe_load(f)
    path = CONFIG / f"{name}.yml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def rng(offset: int = 0) -> np.random.Generator:
    """解析全体で共有する乱数生成器。offset で用途ごとに独立な系列にする。"""
    seed = load_config("analysis")["seed"]
    return np.random.default_rng(seed + offset)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def record_provenance(path: Path, url: str, extra: dict | None = None) -> dict:
    """取得したファイルの取得元・取得日・sha256 を data/checksums に追記する。

    データ本体はリポジトリに置かない方針なので、第三者が同一物を取得できたか
    確認する手段としてこの記録が唯一の担保になる。
    """
    entry = {
        "file": path.name,
        "url": url,
        "retrieved": date.today().isoformat(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if extra:
        entry.update(extra)

    ledger = CHECKSUMS / "provenance.jsonl"
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
