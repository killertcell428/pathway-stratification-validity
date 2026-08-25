"""表と図キャプションに出てくる数値が、解析出力に裏付けられているかを調べる。

なぜ audit_numbers とは別に要るか
  audit_numbers は「CSV から計算した値が本文にあるか」を見る。向きが 1 方向なので、
  本文の表にだけ載っていて誰も照合していない数値を見逃す。実際に表 6 は
  数値そのものは正しかったが、行ラベルと数値の対応を誰も確認していなかったため、
  本文と矛盾して見える状態で 335 件の照合を全部通してしまった。

  こちらは逆向きに見る。表と図キャプションの数値を全部拾い、audit_numbers が
  計算している値のどれかに一致するかを調べる。一致しないものは
  「機械的な裏付けがない数値」であり、次のどれかである。
    (a) audit に登録し忘れている（登録する）
    (b) 解析出力ではない構造的な数値（閾値・サイズ帯の境界・文献番号など。除外表に入れる）
    (c) 古い値が残っている（直す）

  (c) を見つけるのが目的で、(a) を洗い出すのが副産物として効く。

使い方:
  pixi run qc-tables
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from tools.audit_numbers import DEFAULT_MDS, checks

# 解析出力ではない数値。表に出てきても裏付けを求めない。
# 「なんとなく出てくる数」を無制限に許すと検査が意味を失うので、理由を添えて列挙する。
STRUCTURAL: dict[float, str] = {
    0.05: "FDR の水準",
    0.5: "ICC の慣例的な閾値 / CPM の下限",
    2.0: "標準偏差の本数 / CPM の上限",
    5.0: "パーセンタイルの下側",
    95.0: "パーセンタイルの上側",
    0.3: "被覆率フィルタの水準",
    -7.0: "採血日のラベル（接種 7 日前）",
    0.02: "図 3 の共変動区間の境界",
    0.1: "図 3 の共変動区間の境界",
    1.0: "図 3 の共変動区間の上端",
    40.0: "発現フィルタの分位",
    50.0: "発現フィルタの分位（採用）",
    189.0: "GSE35846 の検体数",
    755.0: "GTEx 全血の検体数",
    803.0: "GTEx 骨格筋の検体数",
    18.0: "Scheid らの評価可能な群数",
    0.6: "被覆率フィルタの水準（採用）",
    0.8: "被覆率フィルタの水準",
    3.0: "サイズ帯の境界",
    10.0: "サイズ帯の境界",
    11.0: "サイズ帯の境界",
    25.0: "サイズ帯の境界",
    26.0: "サイズ帯の境界",
    60.0: "サイズ帯の境界",
    61.0: "サイズ帯の境界",
    200.0: "サイズ帯の境界",
    6.0: "サイズ帯の境界",
    20.0: "対照数の条件",
    100.0: "対照数の条件",
    500.0: "対照数の条件",
    2000.0: "対照数の条件",
    10000.0: "対照数の条件（採用）",
    50000.0: "対照数の条件",
    1.0: "順位相関の上限 / 完全一致",
    0.0: "ゼロ",
    7.0: "採血の間隔（日）",
    56.0: "反復測定コホートの人数",
    42.0: "表現型コホートの人数",
    2.5: "分位の下側（%）",
    97.5: "分位の上側（%）",
}


SECTION_REF = re.compile(
    r"\d+\.\d+(?:\s*[-–~〜]\s*\d+\.\d+)?\s*(?:節|sections?)"
    r"|(?:節|sections?)\s*\d+\.\d+(?:\s*[-–~〜]\s*\d+\.\d+)?")


def _numbers(line: str) -> list[tuple[str, float]]:
    """行から数値トークンを拾う。表記のまま（丸めの桁を保つため）と値の組で返す。"""
    out = []
    # 「3.5 節」「sections 3.1-3.10」は数値ではないので先に落とす
    line = SECTION_REF.sub(" ", line)
    # 1.23e-4 / 4.5 × 10⁻⁹ のような指数表記はここでは扱わない（別に照合している）
    for m in re.finditer(r"(?<![\w.])(-|−)?\d[\d,]*(?:\.\d+)?%?", line):
        s = m.group(0)
        v = s.replace(",", "").replace("−", "-").rstrip("%")
        try:
            f = float(v)
        except ValueError:
            continue
        out.append((s, f))
    return out


def _tables(text: str) -> list[tuple[str, list[str]]]:
    """(直前の見出し・キャプション, 表の行) の一覧。図キャプションも 1 行の表扱いで拾う。"""
    lines = text.split("\n")
    blocks: list[tuple[str, list[str]]] = []
    i = 0
    caption = ""
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"\*\*(表|Table|図|Figure) ?[0-9a-c]+\*\*", ln.strip())
        if m:
            caption = ln.strip()[:70]
        if ln.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            blocks.append((caption or "（見出しなし）", rows))
            continue
        if ln.strip().startswith("!["):
            blocks.append((ln.strip()[:70], [ln]))
        i += 1
    return blocks


def main() -> int:
    computed = [(label, v, nd) for label, v, nd, _req in checks()]
    rc = 0
    for md in DEFAULT_MDS:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8-sig")
        total = backed = 0
        unbacked: dict[str, list[str]] = {}
        for caption, rows in _tables(text):
            for row in rows:
                if set(row.strip()) <= set("|-: "):  # 区切り行
                    continue
                for s, f in _numbers(row):
                    total += 1
                    if f in STRUCTURAL:
                        backed += 1
                        continue
                    # 表記の桁数から許容差を決める。0.147 なら 5e-4。
                    # 表記の桁で丸めて一致を見る。許容差にすると
                    # 桁の端で取りこぼす（0.5656 と 0.565 が外れる）。
                    dec = len(s.split(".")[1].rstrip("%")) if "." in s else 0
                    if any(round(v, dec) == f for _lab, v, _nd in computed):
                        backed += 1
                    else:
                        unbacked.setdefault(caption, []).append(s)
        print(f"=== 表・図の数値の裏付け: {md.name} ===")
        print(f"{backed} / {total} が解析出力または構造的な数値に一致")
        if unbacked:
            rc = 1
            print("\n【裏付けが見つからない数値】audit 未登録か、古い値が残っている")
            for caption, vals in unbacked.items():
                # 同じ値の繰り返しは 1 度だけ出す
                uniq = sorted(set(vals), key=vals.index)
                print(f"  {caption}")
                print(f"    {', '.join(uniq[:14])}"
                      + (f" ほか {len(uniq) - 14} 種" if len(uniq) > 14 else ""))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
