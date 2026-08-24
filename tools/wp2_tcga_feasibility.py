"""WP2 の採点コホートに TCGA-BRCA を使えるかを、事前登録の前に確かめる。

3 回続けて「登録用の設計を書いたあとに成立しないことが分かった」ので、
設計を確定する前に、成立性の検査を機械で通してから書く。詳細は
docs/08-WP2再設計の論点.md。ここで確認する項目は同文書 6 節の #2 と #3。

#2 検体の絞り込み: 正常隣接組織と転移を落とし、1 患者 1 検体にしたときの検体数。
    乳がん予後シグネチャを正常乳腺で採点しても意味がないので、腫瘍だけを残す。
    同一患者の複数検体を残すと「個人間」の相関に同一個人の重複が混じる。

#3 対照生成が壊れないか: T26 の陰性対照は、遺伝子ごとに「同じ発現量十分位」から
    引き直して作る（src/reliability/metrics.py の matched_random_sets）。
    十分位は pd.qcut(gene_mean, 10, duplicates="drop") で切っている。
    log2(FPKM-UQ+1) には 0 が多いため、下位の境界値が 0 で重複すると
    duplicates="drop" が働いてビンが 10 個未満に縮退し、
    「同じ発現量から引いた」という対照の前提が崩れる。
    縮退するかどうかは分布依存なので、実データで測るしかない。

#1 再測定（被覆率）: 先に測った被覆率は全 59,427 記号に対する値だった。
    実際に採点できるのは発現フィルタを通った遺伝子だけなので、
    フィルタ後の遺伝子集合に対して測り直さないと実効 n が出ない。
    T26 の被覆率下限は 0.6（config/gene_sets.yml）。

出力: results/tables/wp2_tcga_feasibility.csv（閾値ごとの判定）
      results/tables/wp2_tcga_samples.csv（採用した検体の一覧）
      results/tables/wp2_tcga_coverage.csv（シグネチャごとの被覆率）
      data/interim/tcga_brca/expr_tumor.parquet（採用閾値の行列）
"""

from __future__ import annotations

import gzip
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import rdata

from src.common import DATA, RAW, RESULTS

TABLES = RESULTS / "tables"
MATRIX = RAW / "TCGA-BRCA.star_fpkm-uq.tsv.gz"
PROBEMAP = RAW / "gencode.v36.annotation.gtf.gene.probemap"
SIGNATURES = (RAW / "venet2011" / "ploscb-venet-dumont-detours"
              / "data" / "signatures.Rda")
INTERIM = DATA / "interim" / "tcga_brca"

# 採用する発現フィルタ閾値。残る遺伝子数が T26 の他コホート
# （マイクロアレイ 10,754 / GSE81046 RNA-seq 11,623）に最も近い水準を選ぶ。
# 実測では 1.0 で 12,585 遺伝子。0.0 だと 31,516 遺伝子まで残り、
# 未発現遺伝子の測定ノイズを「共発現しない」と誤読する危険が戻る（論文 §4.5 R3）。
CHOSEN_THRESHOLD = 1.0
MIN_COVERAGE = 0.6      # config/gene_sets.yml と同じ
MIN_GENES, MAX_GENES = 3, 200

# TCGA バーコードの 4 番目（sample type）。01 = 原発腫瘍。
# 11 = 正常隣接、06 = 転移。今回は原発腫瘍だけを残す。
TUMOR_CODES = ("01",)
# 同一患者に複数検体があるときの優先順位。vial A を優先する。
VIAL_ORDER = ("A", "B", "C", "D")

# 発現フィルタの候補閾値。単位は log2(FPKM-UQ+1)。
# FPKM-UQ は CPM ではないので、T26 の「CPM >= 1」をそのまま持ち込めない。
# 分布を見て決める必要があるため、複数水準を並べて遺伝子数と十分位の挙動を比べる。
CANDIDATE_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0)
MIN_FRACTION = 0.5      # T26 と同じ「過半数の個人で検出」規則


def load_matrix() -> pd.DataFrame:
    """log2(FPKM-UQ+1) 行列を読む。

    非 ASCII を含む Windows パスで壊れる読み取り実装があったため、
    ファイルオブジェクトを渡す形で開く。
    float32 にするのは 60,660 x 1,226 を float64 で持つと 600 MB 近くになるため。
    """
    print(f"[1/7] 行列を読む: {MATRIX.name}")
    with gzip.open(MATRIX, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", index_col=0, dtype={0: str})
    df = df.astype(np.float32)
    print(f"  {df.shape[0]:,} 遺伝子 x {df.shape[1]:,} 検体")
    return df


def to_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensembl ID を遺伝子記号に畳み込む（T26 と同じ max_mean 規則）。"""
    print("[2/7] Ensembl ID を遺伝子記号に畳み込む（max_mean）")
    pm = pd.read_csv(PROBEMAP, sep="\t")
    mapping = pm.set_index("id")["gene"]
    sym = df.index.map(mapping)
    ok = sym.notna()
    print(f"  記号に対応づけできた: {int(ok.sum()):,} / {len(df):,}")
    df = df[ok]
    df.index = pd.Index(sym[ok], name="gene")

    # 同一記号に複数 ID があるときは平均発現が最大の行を残す。
    # ここは必ず位置指定（iloc）で並べ替える。df.loc にラベル配列を渡すと、
    # 重複ラベルを持つ行列では 1 ラベルにつき全該当行が返り、
    # 畳み込みが無言で効かなくなる（実測で 60,660 → 60,660 のまま通過した）。
    order = np.argsort(-df.mean(axis=1).to_numpy(), kind="stable")
    df = df.iloc[order]
    df = df[~df.index.duplicated(keep="first")]
    assert df.index.is_unique, "記号の畳み込みに失敗している"
    print(f"  畳み込み後: {df.shape[0]:,} 記号")
    return df


def select_samples(cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """原発腫瘍のみ・1 患者 1 検体に絞る（#2）。"""
    print("[3/7] 検体を絞る（原発腫瘍のみ・1 患者 1 検体）")
    rows = []
    for c in cols:
        parts = c.split("-")
        if len(parts) < 4:
            rows.append({"sample": c, "patient": c, "type": "?", "vial": "?",
                         "keep": False, "reason": "バーコード形式が想定外"})
            continue
        patient = "-".join(parts[:3])
        code, vial = parts[3][:2], parts[3][2:3]
        rows.append({"sample": c, "patient": patient, "type": code, "vial": vial,
                     "keep": code in TUMOR_CODES,
                     "reason": "" if code in TUMOR_CODES else f"検体タイプ {code}"})
    meta = pd.DataFrame(rows)

    counts = meta.groupby("type").size().sort_values(ascending=False)
    print("  検体タイプ別:", ", ".join(f"{t}={n}" for t, n in counts.items()))

    tumor = meta[meta.keep].copy()
    tumor["vial_rank"] = tumor.vial.map({v: i for i, v in enumerate(VIAL_ORDER)}).fillna(9)
    tumor = tumor.sort_values(["patient", "vial_rank", "sample"])
    dup = tumor.patient.duplicated(keep="first")
    meta.loc[tumor.index[dup], "keep"] = False
    meta.loc[tumor.index[dup], "reason"] = "同一患者の 2 検体目"

    kept = meta.loc[meta.keep, "sample"].tolist()
    print(f"  原発腫瘍 {int(tumor.shape[0]):,} 検体 → 患者重複 {int(dup.sum())} 件を除いて "
          f"{len(kept):,} 検体（= 患者数）")
    return kept, meta


def decile_health(values: pd.Series, label: str) -> dict:
    """qcut が 10 ビンに切れるか、ビンの大きさが偏っていないかを測る（#3）。"""
    d = pd.qcut(values, 10, labels=False, duplicates="drop")
    sizes = d.value_counts().sort_index()
    n_bins = int(sizes.shape[0])
    pools: dict[int, list] = defaultdict(list)
    for g, b in d.items():
        pools[int(b)].append(g)
    return {
        f"{label}_ビン数": n_bins,
        f"{label}_最小ビン": int(sizes.min()),
        f"{label}_最大ビン": int(sizes.max()),
        f"{label}_縮退": "縮退" if n_bins < 10 else "正常",
    }


def load_signatures() -> dict[str, tuple[str, list[str]]]:
    """Venet ら (2011) の補足データから遺伝子記号のリストを取り出す。

    構造は名前 -> {sig, name, id, symb, pmid}。symb が記号のベクタ。
    採点対象（公表乳がん予後シグネチャ 48 件と meta-PCNA）と、
    比較の参照になる MSigDB c2（1,892 件）を同じ形で返す。
    """
    parsed = rdata.parser.parse_file(SIGNATURES)
    obj = rdata.conversion.convert(parsed)

    out: dict[str, tuple[str, list[str]]] = {}
    groups = {
        "cancer.signatures": "乳がん予後",
        "prolif.metagene": "meta-PCNA",
        "prolif.signatures": "増殖",
        "noncancer.signatures": "非がん",
        "mSigDB.c2": "MSigDB c2",
    }
    for key, family in groups.items():
        for name, entry in obj[key].items():
            symb = entry.get("symb")
            genes = sorted({str(s) for s in np.asarray(symb).ravel()
                            if s is not None and str(s) not in ("", "nan", "None")})
            out[f"{family}|{name}"] = (family, genes)
    return out


def main() -> int:
    df = load_matrix()
    df = to_symbols(df)
    keep, meta = select_samples(list(df.columns))
    meta.to_csv(TABLES / "wp2_tcga_samples.csv", index=False, encoding="utf-8")
    X = df[keep]

    print("[4/7] 値の分布を見る（ゼロ膨張の程度）")
    flat_zero = float((X.to_numpy() == 0).mean())
    all_zero = int((X.max(axis=1) == 0).sum())
    gene_mean_all = X.mean(axis=1)
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]
    pct = np.percentile(gene_mean_all, qs)
    print(f"  行列全体で値がちょうど 0 のマス: {flat_zero:.1%}")
    print(f"  全検体で 0 の遺伝子: {all_zero:,} / {X.shape[0]:,}")
    print("  遺伝子平均 log2(FPKM-UQ+1) の分位:")
    print("   ", ", ".join(f"{q}%={v:.2f}" for q, v in zip(qs, pct)))

    print("[5/7] 閾値ごとに遺伝子数と十分位の健全性を測る")
    rows = []
    for thr in CANDIDATE_THRESHOLDS:
        detected = (X > thr).mean(axis=1) >= MIN_FRACTION
        sub = X[detected]
        gene_mean = sub.mean(axis=1)
        gene_var = sub.var(axis=1)
        row = {
            "閾値 log2(FPKM-UQ+1)": thr,
            "残る遺伝子数": int(sub.shape[0]),
            "残った遺伝子のうち 0 のマス": round(float((sub.to_numpy() == 0).mean()), 4),
            "平均が同値で並ぶ遺伝子": int(gene_mean.duplicated().sum()),
        }
        row.update(decile_health(gene_mean, "発現量"))
        row.update(decile_health(gene_var, "分散"))
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(TABLES / "wp2_tcga_feasibility.csv", index=False, encoding="utf-8")

    ok = out[(out["発現量_縮退"] == "正常") & (out["分散_縮退"] == "正常")]
    if ok.empty:
        print("  × どの閾値でも十分位が縮退する。対照生成をこのまま持ち込めない。")
    else:
        print(f"  十分位が 10 ビンに切れる閾値: "
              f"{', '.join(str(v) for v in ok['閾値 log2(FPKM-UQ+1)'])}")

    print(f"[6/7] 採用閾値 {CHOSEN_THRESHOLD} の行列を保存する")
    detected = (X > CHOSEN_THRESHOLD).mean(axis=1) >= MIN_FRACTION
    final = X[detected]
    INTERIM.mkdir(parents=True, exist_ok=True)
    final.to_parquet(INTERIM / "expr_tumor.parquet")
    print(f"  {final.shape[0]:,} 遺伝子 x {final.shape[1]:,} 患者 -> "
          f"{INTERIM / 'expr_tumor.parquet'}")

    print("[7/7] 発現フィルタ後の被覆率を測る（#1 の再測定）")
    # 被覆率の損失を 2 段階に分ける。
    #   注釈段階: Venet の記号（2007 年の HUGO）が gencode v36 に存在しない
    #             → 記号の世代差。別名表があれば回収できる性質のもの
    #   発現段階: 記号は存在するが、この組織で発現していないためフィルタで落ちた
    #             → データの性質。回収できない
    # どちらで落ちたかで、対処（別名表を作るか、除外するか）が変わる。
    annotated = set(pd.read_csv(PROBEMAP, sep="\t")["gene"].astype(str))
    present = set(final.index)

    sigs = load_signatures()
    rows = []
    for key, (family, genes) in sigs.items():
        n = len(genes)
        n_ann = sum(g in annotated for g in genes)
        hit = [g for g in genes if g in present]
        cov = len(hit) / n if n else 0.0
        rows.append({
            "family": family, "signature": key.split("|", 1)[1],
            "n_genes": n,
            "n_annotated": n_ann,
            "coverage_annotation": round(n_ann / n, 4) if n else 0.0,
            "n_present": len(hit),
            "coverage": round(cov, 4),
            "pass_coverage": cov >= MIN_COVERAGE,
            "pass_min_size": len(hit) >= MIN_GENES,
            "pass_max_size": len(hit) <= MAX_GENES,
        })
    cov_df = pd.DataFrame(rows)
    # 主解析はサイズ上限を課さない。対照がサイズをマッチしているため、
    # 200 遺伝子の上限は対照超過 z には不要（docs/08 §4-4）。
    cov_df["主解析"] = cov_df.pass_coverage & cov_df.pass_min_size
    cov_df["感度分析"] = cov_df["主解析"] & cov_df.pass_max_size
    cov_df.to_csv(TABLES / "wp2_tcga_coverage.csv", index=False, encoding="utf-8")

    for family, g in cov_df.groupby("family", sort=False):
        print(f"  [{family}] {len(g)} 件 → 主解析 {int(g['主解析'].sum())} 件 / "
              f"感度分析 {int(g['感度分析'].sum())} 件"
              f"（被覆率 中央値 {g.coverage.median():.1%}）")

    target = cov_df[cov_df.family == "乳がん予後"].copy()
    n_main = int(target["主解析"].sum())
    n_sens = int(target["感度分析"].sum())
    print(f"\n  === 主対象（乳がん予後 {len(target)} 件）===")
    print(f"  注釈段階の被覆率  中央値 {target.coverage_annotation.median():.1%} / "
          f"60% 以上 {int((target.coverage_annotation >= MIN_COVERAGE).sum())} 件")
    print(f"  発現フィルタ後    中央値 {target.coverage.median():.1%} / "
          f"60% 以上 {int(target.pass_coverage.sum())} 件")
    print(f"  >>> 主解析 n = {n_main}（全サイズ）  感度分析 n = {n_sens}（3-200 遺伝子）")

    lost = target[~target["主解析"]]
    if len(lost):
        print("\n  主解析から落ちる件（どの段階で落ちたか）:")
        for _, r in lost.sort_values("coverage").iterrows():
            stage = ("記号が当たらない" if r.coverage_annotation < MIN_COVERAGE
                     else "この組織で発現していない")
            why = []
            if not r.pass_coverage:
                why.append(f"被覆率 {r.coverage:.1%}")
            if not r.pass_min_size:
                why.append(f"検出 {r.n_present} 遺伝子（下限 {MIN_GENES}）")
            print(f"    {r.signature:<18} 全 {r.n_genes:>4} 遺伝子 → "
                  f"注釈で当たる {r.n_annotated:>4} ({r.coverage_annotation:>5.1%}) → "
                  f"フィルタ後 {r.n_present:>4} ({r.coverage:>5.1%})")
            print(f"      {' かつ '.join(why)} / 原因: {stage}")

    dropped_by_size = target[target["主解析"] & ~target["感度分析"]]
    print(f"\n  感度分析でのみ落ちる件（200 遺伝子超）: {len(dropped_by_size)} 件 — "
          f"{', '.join(dropped_by_size.signature.tolist())}")

    pcna = cov_df[cov_df.family == "meta-PCNA"]
    if len(pcna):
        r = pcna.iloc[0]
        print(f"\n  meta-PCNA: {r.n_genes} 遺伝子 → 注釈 {r.coverage_annotation:.1%} → "
              f"フィルタ後 {r.coverage:.1%}（主解析: {'○' if r['主解析'] else '×'}）")

    print("\n  === 検出力（Spearman、両側 0.05、Fisher z 近似）===")
    from scipy.stats import norm
    for label, n in (("主解析", n_main), ("感度分析", n_sens)):
        line = [f"  {label} n = {n:>2}:"]
        for rho in (0.30, 0.40, 0.50):
            z = np.arctanh(rho) * np.sqrt(n - 3)
            power = float(norm.cdf(z - norm.ppf(0.975)))
            line.append(f"ρ={rho:.2f} → {power:.2f}")
        print("  ".join(line))

    print(f"\n-> {TABLES / 'wp2_tcga_feasibility.csv'}")
    print(f"-> {TABLES / 'wp2_tcga_samples.csv'}")
    print(f"-> {TABLES / 'wp2_tcga_coverage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
