"""帰属解析の推定値に不確実性の幅を付ける（ブートストラップ）。

なぜ要るか
  論文は主成分の説明率を点推定だけで報告している（GTEx の血球組成 1.510、
  GSE35846 の 1.053、精製単球のチップ 0.899 など）。読者はコホート間の差
  （1.053 対 1.510）が実在するのか推定誤差の範囲なのかを判断できない。
  レビューでこれが最大の統計上の指摘（Reject Risk R2）だった。
  詳細は docs/09-論文レビュー_Evidence-Limited_Stop.md。

やり方: 復元なしの部分抽出を、記述統計として使う
  検体の 70% を復元なしで抜き、標準化・特異値分解・R^2・並べ替え帰無水準までを
  毎回やり直す。報告するのは部分標本の推定値そのものの 2.5-97.5 パーセンタイルで
  あり、「部分標本間で推定値がどこまで動くか」を示す記述的な範囲である。

  **これは信頼区間ではない。** 当初は Politis-Romano の中心化区間（分位を
  sqrt(m/n) で換算）として実装したが撤回した。部分抽出の標準理論は m/n -> 0 を
  仮定しており m = 0.7n は設定外で、被覆率を実測すると名目 95% に対し 85-90%
  しか出なかった（subsampling_coverage.py）。推論的な補正をかけたまま
  「信頼区間ではない」と書くのは中途半端なので、補正を外した。

  **並べ替え検定は推定値の分散を与えない。** ラベルを並べ替えて得られるのは
  「その因子が無関係だったときの水準」であって、推定値そのもののばらつきでは
  ないため、幅を出すには別に再標本が要る。ここを混同すると
  「p が小さいから推定値も安定している」という誤読になる。

なぜ復元抽出（通常のブートストラップ）を使わないか — 実測で棄却した
  最初は復元抽出で実装した。重複検体が生じても、同じ再標本の上で帰無水準も
  計算し直せば観測側と帰無側の両方が同じだけ上振れして相殺されると考えたが、
  **これは誤りだった。** GTEx で技術要因の点推定 0.594 に対し、
  復元抽出の再標本は 0.899-0.979 に偏った。

  理由: 重複した検体は必ず同じバッチ・同じ採取施設に属するので、観測側では
  群内分散が下がり R^2 が上がる。一方、並べ替え側では重複検体に別のラベルが
  割り当たるため群内分散は下がらない。したがって上振れは観測側にだけ乗る。
  技術要因は群が多く各群が小さいため影響が最大になり（群サイズ 2 に重複が
  1 つ入るだけで群内分散がほぼ消える）、血球組成のような 5 分位（各群 151 検体）
  ではほとんど動かない。実測でも血球組成は 1.510 に対し再標本 1.456-1.587 と
  整合していた。復元なしならこの経路が閉じる。

計算量の都合
  再標本ごとに特異値分解が要るため、再標本内の並べ替え回数は点推定時
  （200 回）より減らす。区間の推定に効くのは再標本の数であり、
  各再標本内の帰無水準の誤差は再標本をまたいで平均されるため。

対象コホート
  論文で直接比較されるのは GTEx 全血の 1.510 と GSE35846 全血の 1.053 であり、
  「この差は実在するのか」がレビューでの指摘だった。したがって両方を同じ
  手続きで測る。集計規則（分散重み付き、対象 PC の寄与率合計で正規化）も
  点推定側と同一にする。

使い方
  pixi run wp1-gtex-uncertainty              GTEx 全血
  pixi run wp1-tech-uncertainty              GSE35846 全血
  pixi run python -m src.reliability.attribution_uncertainty --cohort gtex --n-boot 200
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import yaml

from ..common import CONFIG, DATA, RAW, RESULTS, rng
from ..download.fetch_gene_sets import parse_gmt
from ..scoring.methods import _standardize_rows
from .gtex_attribution import discretize, r2_oneway
from .technical_axes_check import BIOLOGICAL, COMPOSITION_SETS, CONTINUOUS, TECHNICAL

# 名前空間つきの定数（INTERIM/TABLES）は T26_DATASET に追随するが、
# 本スクリプトは 2 コホートを 1 プロセスで扱えるようにするため、
# 参照先をコホートごとに明示する。
INTERIM_ROOT = DATA / "interim"
META_ROOT = DATA / "metadata"
TABLES_ROOT = RESULTS / "tables"
CONFIG_DIR = CONFIG


def load_gse35846() -> tuple[np.ndarray, list[tuple[str, str, np.ndarray]], int]:
    """GSE35846 全血。因子の定義は technical_axes_check の定数をそのまま使う。

    定数を再宣言せず import しているのは、点推定側と因子集合がずれると
    区間が点推定に対応しなくなるためである。
    """
    expr = pd.read_parquet(INTERIM_ROOT / "tech_expr_wholeblood.parquet")
    meta = pd.read_csv(META_ROOT / "tech_samples.csv", index_col=0)
    meta = meta.reindex(expr.columns)
    x = expr.to_numpy(dtype=np.float64)
    z = np.nan_to_num(_standardize_rows(x))

    gmt = RAW / "gene_sets" / "celltype__PanglaoDB_Augmented_2021.gmt"
    sets = parse_gmt(gmt.read_text(encoding="utf-8")) if gmt.exists() else {}
    index = {g: i for i, g in enumerate(expr.index)}

    out: list[tuple[str, str, np.ndarray]] = []
    for c in TECHNICAL:
        if c not in meta.columns:
            continue
        v = meta[c]
        lab = (pd.qcut(pd.to_numeric(v, errors="coerce"), 5, labels=False,
                       duplicates="drop").fillna(-1).to_numpy()
               if c in CONTINUOUS else pd.factorize(v.astype(str))[0])
        out.append((c, "技術", lab))
    for c in BIOLOGICAL:
        if c not in meta.columns:
            continue
        v = meta[c]
        lab = (pd.qcut(pd.to_numeric(v, errors="coerce"), 5, labels=False,
                       duplicates="drop").fillna(-1).to_numpy()
               if c in CONTINUOUS else pd.factorize(v.astype(str))[0])
        out.append((c, "生物学", lab))
    for full in COMPOSITION_SETS:
        name = full.split("|", 1)[1]
        genes = [g for g in sets.get(name, []) if g in index]
        if len(genes) >= 10:
            score = np.nanmean(z[[index[g] for g in genes]], axis=0)
            lab = pd.qcut(pd.Series(score), 5, labels=False,
                          duplicates="drop").to_numpy()
            out.append((f"組成:{name}", "血球組成", lab))
    return x, out, 5


def canonical_point(cohort: str) -> dict[str, float]:
    """論文が報告している点推定を、既存の結果表から読む。

    区間は公表値を中心に置く必要がある。ここで推定し直した値を中心にすると、
    並べ替えの乱数が違うぶんだけ論文の数字とずれた区間になってしまう
    （実測で GSE35846 は 1.053 に対し再計算が 1.051 だった）。
    """
    if cohort == "gtex":
        df = pd.read_csv(TABLES_ROOT / "gtex_blood" / "pc_attribution.csv")
        return df.groupby("kind").weighted.sum().to_dict()
    df = pd.read_csv(TABLES_ROOT / "technical_axes.csv")
    w = df.drop_duplicates("pc").set_index("pc").var_ratio.sum()
    name = {"composition": "血球組成", "technical": "技術", "biological": "生物学"}
    out: dict[str, float] = {}
    for kind, g in df.groupby("kind"):
        out[name.get(kind, kind)] = float((g.r2_excess * g.var_ratio).sum() / w)
    return out


def build_factors(cov: pd.DataFrame, z: np.ndarray, index: dict[str, int],
                  att: dict) -> dict[str, tuple[str, np.ndarray]]:
    """点推定側（gtex_attribution.main）と同じ規則で因子を組む。

    因子の作り方がずれると区間が点推定に対応しなくなるため、
    離散化と採用条件は同じものを使う。
    """
    gmt = RAW / "gene_sets" / "celltype__PanglaoDB_Augmented_2021.gmt"
    sets = parse_gmt(gmt.read_text(encoding="utf-8")) if gmt.exists() else {}

    factors: dict[str, tuple[str, np.ndarray]] = {}
    for c in att["technical"]:
        if c in cov.columns and cov[c].notna().sum() > len(cov) * 0.5:
            factors[c] = ("技術", discretize(cov[c]))
    for c in att["biological"]:
        if c in cov.columns and cov[c].notna().sum() > len(cov) * 0.5:
            factors[c] = ("生物学", discretize(cov[c]))
    for full in att["composition_sets"]:
        name = full.split("|", 1)[1]
        genes = [g for g in sets.get(name, []) if g in index]
        if len(genes) >= 5:
            score = z[np.array([index[g] for g in genes])].mean(axis=0)
            factors[f"組成:{name}"] = ("血球組成", discretize(pd.Series(score)))
    return factors


def weighted_by_kind(x: np.ndarray, cols: list[str], att: dict, gen,
                     factors_meta: list[tuple[str, str]], n_pc: int,
                     nperm: int) -> dict[str, float]:
    """1 つの標本について、分類別の分散重み付き超過説明率を返す。

    cols は使用する検体の位置添字。再標本では重複を含む。
    """
    z = np.nan_to_num(_standardize_rows(x[:, cols]))
    _, s, vt = np.linalg.svd(z - z.mean(axis=1, keepdims=True), full_matrices=False)
    var_ratio = (s ** 2) / float((s ** 2).sum())
    total_w = float(var_ratio[:n_pc].sum())

    out: dict[str, float] = {}
    for k in range(n_pc):
        pc = vt[k]
        for name, kind, lab_full in factors_meta:
            lab = np.asarray(lab_full)[cols]
            ok = ~pd.isna(lab)
            if ok.sum() < 10 or len(np.unique(lab[ok])) < 2:
                continue
            obs = r2_oneway(pc[ok], lab[ok])
            perm = np.array([r2_oneway(pc[ok], gen.permutation(lab[ok]))
                             for _ in range(nperm)])
            excess = obs - float(np.nanmean(perm))
            out[kind] = out.get(kind, 0.0) + excess * float(var_ratio[k]) / total_w
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="部分標本の数（既定 1000。両端 2.5% を 25 標本で決める）")
    ap.add_argument("--n-perm", type=int, default=50,
                    help="部分標本内の並べ替え回数（既定 50。点推定は 200）")
    ap.add_argument("--frac", type=float, default=0.7,
                    help="1 標本あたりの検体割合 m/n（既定 0.7、復元なし）")
    ap.add_argument("--cohort", choices=["gtex", "gse35846"], default="gtex")
    a = ap.parse_args()

    # attribution セクションはコホート別の設定ファイルにしかない。
    # T26_DATASET に依存させると、pixi タスク側で環境変数を付け忘れたときに
    # 黙って既定値へ落ちる（実際に GTEx 側がそれで落ちた）。ここで明示的に読む。
    if a.cohort == "gtex":
        with (CONFIG_DIR / "analysis_gtex_blood.yml").open(encoding="utf-8") as f:
            att = yaml.safe_load(f)["attribution"]
    else:
        # GSE35846 は点推定側（technical_axes_check）が定数で持っているため、
        # ここでは並べ替え回数と主成分数だけを揃える。
        att = {"n_permutations": 200, "n_pcs": 5}
    gen = rng(97)

    if a.cohort == "gtex":
        n_pc = int(att["n_pcs"])
        expr = pd.read_parquet(INTERIM_ROOT / "gtex_blood" / "expr_blood.parquet")
        cov = (pd.read_csv(META_ROOT / "gtex_blood" / "covariates.csv")
               .set_index("SAMPID").reindex(expr.columns))
        x = expr.to_numpy(dtype=np.float64)
        z0 = np.nan_to_num(_standardize_rows(x))
        index = {g: i for i, g in enumerate(expr.index)}
        factors = build_factors(cov, z0, index, att)
        factors_meta = [(name, kind, lab) for name, (kind, lab) in factors.items()]
        out_dir = TABLES_ROOT / "gtex_blood"
    else:
        x, factors_meta, n_pc = load_gse35846()
        out_dir = TABLES_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    n = x.shape[1]
    print(f"コホート {a.cohort} / 検体 {n} / 遺伝子 {x.shape[0]:,}")
    print(f"因子 {len(factors_meta)} 個 / 主成分 PC1-PC{n_pc}")

    print("[1/2] 論文の点推定を既存の結果表から読む（区間はこれを中心に置く）")
    point = canonical_point(a.cohort)
    recomputed = weighted_by_kind(x, list(range(n)), att, rng(31), factors_meta,
                                  n_pc, int(att["n_permutations"]))
    for k in sorted(point, key=lambda k: -point[k]):
        r = recomputed.get(k, float("nan"))
        print(f"  {k:8s} 公表 {point[k]:6.3f}   再計算 {r:6.3f}   差 {r - point[k]:+.3f}")
    print("  ※ 差は並べ替えによる帰無水準の推定誤差。区間には公表値を使う")

    m = int(round(n * a.frac))
    print(f"[2/2] 部分抽出 {a.n_boot} 回（1 回 {m}/{n} 検体・復元なし、"
          f"各回 並べ替え {a.n_perm} 回）")
    reps: list[dict[str, float]] = []
    for b in range(a.n_boot):
        cols = list(gen.choice(n, size=m, replace=False))
        reps.append(weighted_by_kind(x, cols, att, gen, factors_meta, n_pc, a.n_perm))
        if (b + 1) % 20 == 0:
            print(f"  {b+1}/{a.n_boot}")

    # 部分標本の推定値そのものの 2.5-97.5 パーセンタイルを報告する。
    #
    # 当初は Politis-Romano の中心化区間（分位を sqrt(m/n) で換算）を使ったが、
    # 撤回した。部分抽出の標準理論は m/n -> 0 を仮定しており m = 0.7n は設定外で、
    # 被覆率を実測すると名目 95% に対し 85-90% しか出ない（subsampling_coverage.py）。
    # 推論的な補正をかけたまま「信頼区間ではない」と書くのは中途半端なので、
    # 補正を外し、記述統計として「部分標本間で推定値がどこまで動くか」だけを示す。
    rep_df = pd.DataFrame(reps)
    rows = []
    for kind in sorted(point, key=lambda k: -point[k]):
        vals = rep_df[kind].dropna().to_numpy()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rows.append({
            "分類": kind,
            "点推定": round(point[kind], 3),
            "部分標本 2.5%": round(float(lo), 3),
            "部分標本 97.5%": round(float(hi), 3),
            "部分標本中央値": round(float(np.median(vals)), 3),
            "標本数": int(vals.size),
            "m/n": a.frac,
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "pc_attribution_ci.csv", index=False, encoding="utf-8")

    print("\n=== 分散重み付き超過説明率と 95% 区間（中心化部分抽出）===")
    print(out.to_string(index=False))
    print(f"\n-> {out_dir / 'pc_attribution_ci.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
