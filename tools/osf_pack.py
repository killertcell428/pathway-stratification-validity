"""OSF 事前登録に添付する一式を集め、digest つきの一覧を作る。

なぜ要るか
  事前登録は本文で「この登録と一緒に公開する」と 6 箇所で約束している。
  約束したものが登録時に存在しない、あるいは登録後に中身が変わっていると、
  検証可能性を売りにした文書がその一点で崩れる。集約と digest 記録を
  手作業にせず、再実行できる形にしておく。

原本はリポジトリ側。ここが作るのは登録用のコピーで、内容を変えたら再実行する。

出力: docs/計画と運用/osf添付/（ファイル一式 + README.md）
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "計画と運用" / "osf添付"

# (原本, 何が入っているか) 事前登録が名前で指定しているものを漏らさない
ITEMS: list[tuple[str, str]] = [
    ("results/tables/wp2_venet_outcome.csv",
     "目的変数（Loi OS / Loi RFS / NKI RFS）。被覆率・log-rank p・補正前後の HR と p・meta-PCNA 相関・比"),
    ("results/tables/wp2_tcga_outcome.csv",
     "同じ量を TCGA-BRCA で計算したもの（感度分析 b と Declaration 3 の根拠）"),
    ("results/tables/wp2_tcga_endpoint_grid.csv",
     "終点 4 種 x コホート 5 種の全数評価（Declaration 3 の 20 通り）"),
    ("results/tables/wp2_tcga_coverage.csv",
     "シグネチャごとの被覆率（注釈段階と発現フィルタ後）と除外判定"),
    ("results/tables/wp2_target_reproducibility.csv",
     "目的変数の候補ごとのコホート間再現性（差し替えの根拠）"),
    ("data/checksums/provenance.jsonl",
     "入力ファイルの取得元・取得日・バイト数・SHA-256"),
    ("src/reliability/wp2_null_model.py",
     "帰無モデル（交換可能性検定）。登録前に凍結"),
    ("tests/test_wp2_null_model.py",
     "帰無モデルの検査 11 件（合成データのみで走行）"),
    ("tools/audit_wp2_prereg.py",
     "事前登録の数値を CSV と機械照合する"),
    ("tools/wp2_target_reproducibility.py",
     "再現性の表を再生成する"),
    ("config/gene_sets.yml",
     "被覆率・サイズのフィルタ規則"),
]

SKIPPED = [
    ("発現行列（TCGA 317 MB、Venet 87 MB）",
     "公開アクセッションなので取得元と SHA-256 を provenance.jsonl で示すほうが確実で、容量も現実的でない"),
    ("説明変数の値",
     "登録時点で存在しない。それがこの登録の要点である"),
    ("docs/ 配下の検討記録",
     "経緯は登録本文の Knowledge of Data に必要な範囲で書いてある"),
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    missing = [rel for rel, _ in ITEMS if not (ROOT / rel).exists()]
    if missing:
        print("原本が見つからない:")
        for m in missing:
            print(f"  x {m}")
        return 1

    rows = []
    for rel, why in ITEMS:
        src = ROOT / rel
        shutil.copy2(src, DEST / src.name)
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        rows.append((src.name, src.stat().st_size, digest, rel, why))
        print(f"  {src.name:<38} {src.stat().st_size:>8} bytes")

    md = [
        "# OSF 事前登録に添付するファイル",
        "",
        "`pixi run osf-pack` が集めたもの。**原本はリポジトリ側で、ここは登録用のコピー。**",
        "原本を変えたら再実行する。digest は原本に対して取っている。",
        "",
        "| ファイル | バイト | SHA-256 (先頭 16) | 原本 | 何が入っているか |",
        "|---|---|---|---|---|",
    ]
    md += [f"| `{n}` | {s:,} | `{d[:16]}` | `{r}` | {w} |" for n, s, d, r, w in rows]
    md += ["", "## 添付しないもの", ""]
    md += [f"- **{what}** — {why}" for what, why in SKIPPED]
    md += [""]
    (DEST / "README.md").write_bytes(b"\xef\xbb\xbf" + "\n".join(md).encode("utf-8"))
    print(f"\n{len(rows)} 件を集約、README.md に digest 一覧を作成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
