"""主要図を作る。

軸ラベルは英語で書く（論文にそのまま入れるため／和文フォント依存を避けるため）。
配色は IBM Carbon のモノトーン＋ブルー 1 色に限定する。

  fig1 条件効果と個人間整合性の乖離
  fig2 ファミリー別の内部整合性（対照との比較つき）
  fig3 手法間一致は内部整合性に依存する
  fig4 個人層別化の適格性マップ（4 象限）
"""

from __future__ import annotations

import struct
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..common import FIGURES, TABLES, load_config  # noqa: E402

INK = "#161616"
MUTED = "#525252"
SUBTLE = "#8c8c8c"
HAIRLINE = "#e0e0e0"
BLUE = "#0f62fe"

# 原稿（Word / HTML）に貼るときの図の幅。Letter・余白 1 インチの本文幅に合わせる。
# 図はこの幅で作り、保存幅がこれを大きく超えないようにする（超えた分だけ縮小され、
# 図中の文字が読めなくなる）。build_docx.py の IMG_WIDTH_IN と一致させる。
DOC_WIDTH_IN = 6.5

FAMILY_LABEL = {
    "complex": "Protein complexes\n(CORUM)",
    "pathway": "Reaction pathways\n(Reactome)",
    "signature": "Expression signatures\n(Hallmark)",
    "regulon": "TF regulons\n(TRRUST)",
    "celltype": "Cell type markers\n(PanglaoDB)",
    "data_derived": "Data-derived modules\n(this study)",
}

# 図中で使う短縮記号（ラベルの重なりを避ける）
FAMILY_CODE = {
    "complex": "C", "pathway": "P", "signature": "S",
    "regulon": "R", "celltype": "M", "data_derived": "D",
}


def coherence_pass(df: pd.DataFrame) -> pd.Series:
    """個人整合性の合格判定。2 種の対照（発現量マッチ・分散マッチ）の両方で
    FDR 0.05 を通ることを要求する。全図・全数値でこの 1 つの定義に統一する。"""
    return (df["null_q"] < 0.05) & (df["var_null_q"] < 0.05)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK, labelsize=9, width=1.0)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def _wrap(text: str, avail_in: float, size: float) -> str:
    """指定した幅（インチ）に収まる文字数へ折り返す。"""
    per_char_in = size * 0.5 / 72.0        # 比例フォントの平均字幅を 0.5 em と見積もる
    return textwrap.fill(" ".join(text.split()),
                         width=max(40, int(avail_in / per_char_in)))


def _save(fig, name: str, cfg: dict, footnote: str | None = None,
          laid_out: bool = False, top: float = 1.0) -> None:
    """図を保存する。保存幅は必ず figsize（= 本文幅）に一致させる。

    以前は bbox_inches="tight" で保存していた。この指定は軸の外に置いた要素
    ——長い y 軸ラベル、軸下のグレーの補足文、軸右の凡例——まで含めて保存範囲を
    広げるため、保存幅が figsize より大きくなる。その画像を原稿で本文幅にそろえると
    広がった分だけ図全体が縮み、軸内の文字が読めなくなる。実際に fig4 は保存幅
    13.7 インチまで広がり、原稿上で 45% に縮小されていた（8pt の文字が 3.6pt 相当）。

    そこで tight_layout で全要素を figsize の内側に収め、bbox は指定せずに保存する。
    補足文は軸座標ではなく図座標に置き、その行数ぶんの余白を下に確保する。
    """
    # laid_out=True は呼び出し側がすでに余白を確保している場合。ここで
    # tight_layout を呼び直すと、その余白が上書きされて要素が重なる。
    if laid_out:
        pass
    elif footnote:
        wrapped = _wrap(footnote, fig.get_figwidth() - 0.25, 8.0)
        n_lines = wrapped.count("\n") + 1
        # 1 行あたりの高さを図の高さに対する比で見積もり、下端に余白を取る
        line_frac = 8.0 * 1.5 / 72.0 / fig.get_figheight()
        bottom = min(0.30, line_frac * n_lines + 0.03)
        fig.tight_layout(rect=(0, bottom, 1, top))
        fig.text(0.012, bottom - 0.01, wrapped, fontsize=8, color=SUBTLE,
                 va="top", linespacing=1.5)
    else:
        fig.tight_layout(rect=(0, 0, 1, top))

    for ext in cfg["output"]["figure_format"]:
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, dpi=cfg["output"]["dpi"], facecolor="white")
        print(f"  {path.name}")
    # 保存幅を検査する。原稿では幅を DOC_WIDTH_IN にそろえて貼るので、
    # 保存幅がそれを超えるほど図中の文字が小さくなる。
    png = FIGURES / f"{name}.png"
    if png.exists():
        w_in = png_width_inches(png, cfg["output"]["dpi"])
        scale = DOC_WIDTH_IN / w_in
        flag = "" if scale >= 0.98 else f"  ★縮小率 {scale:.0%}"
        print(f"    実寸 {w_in:.2f} in → 原稿では {scale:.0%}{flag}")
    plt.close(fig)


def png_width_inches(path: Path, default_dpi: int) -> float:
    """PNG のヘッダから幅（インチ）を読む。pHYs があればその DPI を使う。"""
    b = path.read_bytes()
    w = struct.unpack(">I", b[16:20])[0]
    dpi = default_dpi
    i = b.find(b"pHYs")
    if i > 0:
        px, _, unit = struct.unpack(">IIB", b[i + 4:i + 13])
        if unit == 1:
            dpi = round(px * 0.0254)
    return w / dpi


def fig1_dissociation(df: pd.DataFrame, cfg: dict) -> None:
    """条件効果があるのに個人間では共変動しない、という乖離を 1 枚で示す。"""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    d = df.dropna(subset=["cohens_d", "internal_consistency"])
    x = d["cohens_d"].abs()
    y = d["internal_consistency"]
    passes = coherence_pass(d)

    ax.axhspan(
        float(np.nanpercentile(d["null_mean"], 5)),
        float(np.nanpercentile(d["null_mean"], 95)),
        color=HAIRLINE, zorder=0,
    )
    ax.scatter(x[~passes], y[~passes], s=6, c=SUBTLE, alpha=0.45, linewidths=0,
               label=f"fails coherence controls (n={int((~passes).sum())})")
    ax.scatter(x[passes], y[passes], s=10, c=INK, alpha=0.85, linewidths=0,
               label=f"passes coherence controls (n={int(passes.sum())})")

    anchors = df[df["family"] == "anchor"]
    for _, r in anchors.iterrows():
        label = r["set"].split("|")[1]
        ax.scatter([abs(r["cohens_d"])], [r["internal_consistency"]], s=70,
                   facecolors="none", edgecolors=BLUE, linewidths=1.6, zorder=5)
        ax.annotate(label, (abs(r["cohens_d"]), r["internal_consistency"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=BLUE)

    ax.axvline(0.5, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.set_xlabel("Condition effect  |Cohen's d|  (resting -> LPS 24 h, paired)", fontsize=10, color=INK)
    ax.set_ylabel("Individual-level coherence\n(mean inter-gene Spearman rho across donors)",
                  fontsize=10, color=INK)
    fig.text(0.012, 0.965, "Condition effects are near-universal; "
                           "individual-level coherence is rare",
             fontsize=11, color=INK, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    _save(fig, "fig1_dissociation", cfg, top=0.93,
          footnote="Grey band: 5-95th percentile of expression-matched random gene sets. "
                         f"n = {len(d)} gene sets, 207 held-out donors, CD14+ monocytes.")



def fig2_by_family(df: pd.DataFrame, cfg: dict) -> None:
    """ファミリー別の内部整合性。教科書由来かデータ由来かで結果が分かれる。"""
    fams = [f for f in FAMILY_LABEL if f in set(df["family"])]
    order = sorted(fams, key=lambda f: df.loc[df.family == f, "internal_consistency"].median())
    data = [df.loc[df.family == f, "internal_consistency"].dropna().values for f in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, showfliers=False,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#f4f4f4")
        patch.set_edgecolor(INK)
        patch.set_linewidth(1.0)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set_color(MUTED)
            line.set_linewidth(1.0)
    for line in bp["medians"]:
        line.set_color(INK)
        line.set_linewidth(2.0)

    # n の注記を入れる余白を軸の内側に作る。軸の右外に置くと保存幅が広がる。
    x0, x1 = ax.get_xlim()
    ax.set_xlim(x0, x1 + (x1 - x0) * 0.10)
    xmax = x1
    for i, f in enumerate(order, 1):
        sub = df[df.family == f]
        ax.plot([sub["null_mean"].median()] * 2, [i - 0.28, i + 0.28],
                color=BLUE, linewidth=1.8, zorder=5)
        ax.text(xmax, i, f" n={len(sub)}", ha="left", va="center", fontsize=8, color=SUBTLE)

    # ラベルは 2 行のまま使う。1 行に伸ばすと軸が右に押され、表題と x 軸ラベルが
    # 図の右端で切れる（幅を figsize に固定しているため、はみ出し分は救済されない）。
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in order], fontsize=8.5, color=INK)
    ax.set_xlabel("Individual-level coherence (mean inter-gene Spearman rho)",
                  fontsize=10, color=INK)
    # 表題は軸ではなく図の左端に置く。軸に付けると y 軸ラベルの幅ぶん右へずれる。
    fig.text(0.012, 0.965, "Only data-derived modules and cell type markers exceed "
                           "random controls", fontsize=11, color=INK, va="top")
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.8)
    _style(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, "fig2_by_family", cfg, top=0.93,
          footnote="Blue segment: median of expression-matched random control sets for that family. "
                   "Boxes: median and IQR across sets; whiskers 1.5 x IQR.")



def fig3_method_agreement(df: pd.DataFrame, cfg: dict) -> None:
    """手法を変えると個人の順位が変わるのは、セットがまとまっていないときに限る。"""
    bins = [(-1, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.3), (0.3, 1.0)]
    labels, mean_v, min_v, ns = [], [], [], []
    for lo, hi in bins:
        m = (df.internal_consistency >= lo) & (df.internal_consistency < hi)
        if not m.any():
            continue
        labels.append(f"< {hi:.2f}" if lo < 0 else f"{lo:.2f}–{hi:.2f}")
        mean_v.append(df.loc[m, "method_agreement_mean"].median())
        min_v.append(df.loc[m, "method_agreement_min"].median())
        ns.append(int(m.sum()))

    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(xs - 0.19, mean_v, width=0.36, color=MUTED, edgecolor=INK, linewidth=0.8,
           label="mean over all method pairs")
    ax.bar(xs + 0.19, min_v, width=0.36, color="#f4f4f4", edgecolor=INK, linewidth=0.8,
           label="worst method pair")
    ax.axhline(0, color=INK, linewidth=1.0)
    for x, n in zip(xs, ns):
        ax.text(x, ax.get_ylim()[1], f"n={n}", ha="center", va="bottom", fontsize=8, color=SUBTLE)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("Individual-level coherence (mean rho)", fontsize=10, color=INK)
    ax.set_ylabel("Agreement of donor rankings\nbetween scoring methods (Spearman rho)",
                  fontsize=10, color=INK)
    fig.text(0.012, 0.965, "Scoring method choice changes who ranks high "
                           "— only for incoherent sets",
             fontsize=11, color=INK, va="top")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    _save(fig, "fig3_method_agreement", cfg, top=0.93,
          footnote="Methods compared: z-mean, singscore, PLAGE, median-rank. "
                         "Median across sets in each bin.")



def fig4_qualification_map(df: pd.DataFrame, cfg: dict) -> None:
    """適格性マップ。条件効果と個人整合性の 2 軸で 4 象限に置く。"""
    d = df.dropna(subset=["cohens_d", "null_q", "var_null_q"]).copy()
    d["x"] = d["cohens_d"].abs()
    # 縦軸は q 値にする。
    #
    # 以前は「厳しい側の z」を縦軸に置き、合格セットの z の最小値を水平線として
    # 「BH-FDR 0.05 の境界」と書いていた。これは成り立たない。BH の閾値は順位に
    # 依存し、しかも各セットの帰無分布は形が違うので、全セット共通の z が
    # BH-FDR 0.05 の境界になることはない。実際に描いていたのは合格集合の
    # 下端であって、判定境界ではなかった（正規近似時代の設計の名残）。
    #
    # 2 つの対照の q の大きい側（＝厳しい側）を取り、-log10 にすると、
    # 水平線 -log10(0.05) = 1.301 がそのまま合格境界になる。
    # coherence_pass（両方の q < 0.05）と図が厳密に一致する。
    d["q_worse"] = d[["null_q", "var_null_q"]].max(axis=1)
    d["y"] = -np.log10(d["q_worse"])
    d["pass_y"] = coherence_pass(d)
    x_thr = 0.5
    y_thr = -np.log10(0.05)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(d["x"], d["y"], s=6, c=SUBTLE, alpha=0.35, linewidths=0)

    legend_rows = []
    for fam, sub in d[d.family.isin(FAMILY_LABEL)].groupby("family"):
        mx, my = sub["x"].median(), sub["y"].median()
        ax.scatter([mx], [my], s=110, c=INK, marker="s", zorder=6)
        ax.annotate(FAMILY_CODE[fam], (mx, my), color="white", fontsize=8.5,
                    ha="center", va="center", zorder=7)
        legend_rows.append((FAMILY_CODE[fam], FAMILY_LABEL[fam].replace("\n", " "), my))

    ax.axvline(x_thr, color=INK, linewidth=1.0)
    ax.axhline(y_thr, color=INK, linewidth=1.0)

    quad = {
        "usable for both": ((d.x >= x_thr) & d.pass_y).sum(),
        "condition only": ((d.x >= x_thr) & ~d.pass_y).sum(),
        "individual only": ((d.x < x_thr) & d.pass_y).sum(),
        "neither": ((d.x < x_thr) & ~d.pass_y).sum(),
    }
    # 象限ラベルは軸の隅に置く。合格側は上、不合格側は下。
    # 上端に余白を作らないと「individual only」の箱が点群に食い込む。
    xmax = d["x"].max()
    ybot, ytop = float(d["y"].min()), float(d["y"].max())
    # 下端にも余白を作る。ファミリー中央値の四角（C・R）が y ≈ 0.1-0.2 に来るので、
    # 不合格側の箱をその下に逃がさないと重なる。
    ax.set_ylim(ybot - 0.52, ytop + 0.45)
    y_hi, y_lo = ytop + 0.40, ybot - 0.06
    positions = [
        (xmax * 0.60, y_hi), (xmax * 0.60, y_lo),
        (0.02, y_hi), (0.02, y_lo),
    ]
    for (label, n), (px, py) in zip(quad.items(), positions):
        ax.text(px, py, f"{label}\nn={n} ({n/len(d):.0%})", fontsize=9, color=INK,
                va="top", bbox=dict(facecolor="white", edgecolor=HAIRLINE, linewidth=0.8, pad=4))

    # 凡例は軸の中に置く。軸の外（x > 1）に置くと bbox_inches="tight" が保存幅を
    # 広げ、原稿に貼るときの縮小率が上がって図中の文字が読めなくなる。
    text = "\n".join(f"{c}  {name}" for c, name, _ in sorted(legend_rows, key=lambda r: -r[2]))
    # 枠は右上に寄せるが、行そのものは左揃えにする（ma="left"）。
    # 右揃えのままだと記号 D/M/S/P/C/R の列がそろわない。
    ax.text(0.995, 0.985, text, transform=ax.transAxes, fontsize=8, color=INK,
            va="top", ha="right", ma="left", family="monospace",
            bbox=dict(facecolor="white", edgecolor=HAIRLINE, linewidth=0.8, pad=4))

    ax.set_xlabel("Condition effect  |Cohen's d|", fontsize=10, color=INK)
    # 縦軸ラベルは 2 行に抑える。3 行にすると図の上端まで届いて表題に当たる。
    ax.set_ylabel("Individual-level coherence\n−log₁₀ q (weaker control)",
                  fontsize=10, color=INK)
    fig.text(0.012, 0.965, "Qualification map for individual stratification",
             fontsize=11, color=INK, va="top")
    _style(ax)
    _save(fig, "fig4_qualification_map", cfg, top=0.93,
          footnote="The vertical axis is the BH-FDR q-value from the empirical p-values "
              "against 10,000 matched random sets, taken as the weaker of the two controls "
              f"and plotted as −log₁₀. The horizontal line is therefore the qualification "
              f"boundary itself, q = 0.05 (−log₁₀ q = {y_thr:.3f}). The vertical line "
              f"|d| = {x_thr} is provisional and is not used in the decision. "
              "Squares mark family medians. Test-retest reliability is absent: the cohort "
              "has no repeated measurement of the same condition.")



def fig5_retest(cfg: dict) -> None:
    """反復測定信頼性は、ほぼ全部が非特異的な床である。"""
    path = TABLES / "retest_metrics.csv"
    if not path.exists():
        print("  [skip] retest_metrics.csv がない（pixi run retest を実行する）")
        return
    d = pd.read_csv(path).dropna(subset=["icc", "icc_null_mean"])
    passes = d["icc_q"] < 0.05

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    lim = (-0.05, 0.85)
    ax.plot(lim, lim, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(d.loc[~passes, "icc_null_mean"], d.loc[~passes, "icc"], s=7, c=SUBTLE,
               alpha=0.45, linewidths=0, label=f"indistinguishable from controls (n={int((~passes).sum())})")
    ax.scatter(d.loc[passes, "icc_null_mean"], d.loc[passes, "icc"], s=26, c=INK,
               linewidths=0, label=f"exceeds matched controls (n={int(passes.sum())})")
    # ラベルは全部まとめて右側の空白へ引き出す。
    #
    # 点群は x ≈ 0.30-0.46 の帯に集まり、その右（x > 0.5）は破線しかない空白である。
    # 以前は上位 2 点だけ左へ出していたが、この 2 点は ICC 0.817 と 0.808 で
    # ほぼ同座標にあるため 2 行が詰まって読めず、左へ長く伸ばすと y 軸ラベルに
    # 突き抜けた。右側に一列で積めば、行間を確保しつつ
    # "Binding of TCF LEF CTNNB1 to Target Gene Promoters" も省略せずに収まる。
    # 長いセット名は 2 行に折り返す。x 軸を伸ばして横幅を稼ぐと点群が左に潰れ、
    # 「注釈セットが対照より上にしか出ない」という図の主眼が読めなくなる。
    top = d[passes].nlargest(5, "icc").reset_index(drop=True)
    label_x = 0.50                      # 引き出し先の x（点群の右の空白）
    # 上 2 件は 2 行に折り返されるので、その分だけ間隔を広く取る
    label_y = [0.830, 0.725, 0.650, 0.588, 0.526]
    for i, r in top.iterrows():
        ax.annotate(textwrap.fill(r["set"].split("|")[1], width=26),
                    xy=(r["icc_null_mean"], r["icc"]),
                    xytext=(label_x, label_y[i]),
                    fontsize=8.5, color=BLUE, ha="left", va="center",
                    linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.7,
                                    shrinkA=2, shrinkB=3))

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("ICC of size- and expression-matched random gene sets", fontsize=10, color=INK)
    ax.set_ylabel("ICC of the annotated gene set\n(day -7 vs day 0, both pre-vaccination)",
                  fontsize=10, color=INK)
    fig.text(0.012, 0.965, "Test-retest reliability is almost entirely non-specific",
             fontsize=11, color=INK, va="top")
    ax.legend(frameon=False, fontsize=9, loc="lower right")  # 左上は注釈と重なる
    _style(ax)
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.8)
    _save(fig, "fig5_retest_nonspecific", cfg, top=0.93,
          footnote=f"n = {len(d)} gene sets, 56 donors, PBMC (GSE47353). Dashed line: annotated = random. "
              "Points on the line have reliability that any random gene set of the same size "
              "reproduces.")



def fig6_phenotype_rerandomized(cfg: dict) -> None:
    """どのセットが表現型を予測するかは、測定回ごとに引き直される。"""
    path = TABLES / "phenotype_metrics.csv"
    if not path.exists():
        print("  [skip] phenotype_metrics.csv がない（pixi run phenotype を実行する）")
        return
    d = pd.read_csv(path).dropna(subset=["abs_rho_z", "abs_rho_z_day-7"])
    icc_pass = d["icc_q"] < 0.05
    thr = 2.0
    n0 = int((d.abs_rho_z > thr).sum())
    n7 = int((d["abs_rho_z_day-7"] > thr).sum())
    both = int(((d.abs_rho_z > thr) & (d["abs_rho_z_day-7"] > thr)).sum())
    expected = n0 * n7 / len(d)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.axvline(thr, color=INK, linewidth=1.0)
    ax.axhline(thr, color=INK, linewidth=1.0)
    ax.scatter(d.loc[~icc_pass, "abs_rho_z"], d.loc[~icc_pass, "abs_rho_z_day-7"],
               s=7, c=SUBTLE, alpha=0.45, linewidths=0, label=f"other sets (n={int((~icc_pass).sum())})")
    ax.scatter(d.loc[icc_pass, "abs_rho_z"], d.loc[icc_pass, "abs_rho_z_day-7"],
               s=30, c=INK, linewidths=0,
               label=f"passed the reliability criterion (n={int(icc_pass.sum())})")

    # 区画のラベルは「day 0 only / day -7 only」で対にする。
    # 以前は左上を "beats controls at day 0 only" にしていたが、右へ張り出して
    # both draws の注釈と重なった。上の見出しで既に「対照を上回る」と言っている。
    ax.text(thr + 0.3, ax.get_ylim()[1], f"day 0 only\nn={n0 - both}",
            fontsize=9, color=INK, va="top",
            bbox=dict(facecolor="white", edgecolor=HAIRLINE, linewidth=0.8, pad=4))
    ax.text(ax.get_xlim()[0] + 0.2, thr - 0.4, f"day -7 only\nn={n7 - both}",
            fontsize=9, color=INK, va="top",
            bbox=dict(facecolor="white", edgecolor=HAIRLINE, linewidth=0.8, pad=4))
    # both draws の注釈は右下から引く。右上に置くと "day 0 only" の箱に当たる。
    ax.annotate(f"both draws\nn={both} (chance: {expected:.0f})",
                xy=(thr + 0.15, thr + 0.15), xytext=(thr + 1.5, thr + 1.5),
                fontsize=9, color=BLUE, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=0.9,
                                shrinkA=2, shrinkB=2))

    ax.set_xlabel("Excess association with vaccine response at day 0  (z vs matched random sets)",
                  fontsize=10, color=INK)
    ax.set_ylabel("Same quantity at day -7\n(same 42 donors, one week earlier)",
                  fontsize=10, color=INK)
    fig.text(0.012, 0.965, "Which gene sets appear to predict the phenotype is redrawn "
                           "at each blood draw",
             fontsize=11, color=INK, va="top")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _style(ax)
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.8)
    _save(fig, "fig6_phenotype_rerandomized", cfg, top=0.93,
          footnote="Both draws are pre-vaccination. n = 42 donors with a known response class, "
              f"{len(d)} gene sets. Sets above z = 2 beat their own size- and "
              "expression-matched controls.")



def fig7_cross_cohort_attribution(cfg: dict) -> None:
    """4 コホートで「第 1 主成分を最も説明する要因」が入れ替わることを 1 枚で示す。

    各コホートについて PC1 の超過説明率（偶然水準を引いた R^2）の上位を並べ、
    分類（細胞組成・技術・個人属性）で塗り分ける。個人属性がどのコホートでも
    最下位に沈むことが主眼。
    """
    rows = []

    # 精製単球（マイクロアレイ）: batch_check.csv
    p = TABLES / "batch_check.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.pc == "PC1"]
        for _, r in d.iterrows():
            kind = "technical"
            label = {"chip": "BeadChip", "position": "chip position", "batch": "processing batch"}.get(r.factor, r.factor)
            rows.append(("Purified monocytes\nmicroarray, n=207", label, kind, r.r2_excess))

    # 全血（マイクロアレイ）: technical_axes.csv
    p = TABLES / "technical_axes.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.pc == "PC1"].sort_values("r2_excess", ascending=False)
        kmap = {"composition": "cell composition", "technical": "technical", "biological": "individual attribute"}
        lmap = {"T Cells Naive": "naive T cells", "plate id": "plate", "ethnicity": "ethnicity"}
        for kind_en in ("composition", "technical", "biological"):
            sub = d[d.kind == kind_en]
            if len(sub):
                r = sub.iloc[0]
                rows.append(("Whole blood\nmicroarray, n=189", lmap.get(r.factor, r.factor),
                             kmap[kind_en], r.r2_excess))

    # マクロファージ（RNA-seq）: 祖先集団のみ測定
    p = TABLES / "gse81046" / "ancestry_attribution.csv"
    if p.exists():
        d = pd.read_csv(p)
        r = d[d.pc == "PC1"].iloc[0]
        rows.append(("Macrophages\nRNA-seq, n=79", "genetic ancestry", "individual attribute", r.excess))

    # 全血（RNA-seq, GTEx）
    p = TABLES / "gtex_blood" / "pc_attribution.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.pc == "PC1"].sort_values("excess", ascending=False)
        lmap = {"組成:Neutrophils": "neutrophils", "SMTSISCH": "ischemic time", "AGE": "age",
                "DTHHRDY": "death classification", "SMRIN": "RIN",
                "SMCENTER": "collection center", "SMGEBTCH": "expression batch"}
        # 虚血時間と死因分類は「採取時の状況」であって、安定した個人特性でも
        # 純粋な測定技術でもない。本文の主張と分類を一致させるため独立させる。
        cat = {"SMTSISCH": "collection circumstance", "DTHHRDY": "collection circumstance",
               "AGE": "individual attribute", "SEX": "individual attribute"}
        for _, r in d.iterrows():
            k = cat.get(r.factor)
            if k is None:
                k = "cell composition" if r.kind == "血球組成" else "technical"
            rows.append(("Whole blood (GTEx)\nRNA-seq, n=755", lmap.get(r.factor, r.factor), k, r.excess))

    if not rows:
        print("  [skip] fig7: 帰属検証の集計表がない")
        return

    df = pd.DataFrame(rows, columns=["cohort", "factor", "kind", "excess"])
    df = df[df.excess > -0.05]
    # 分類ごとに最大の要因だけを残す（コホート内で最大 4 本）
    df = df.sort_values("excess", ascending=False).groupby(["cohort", "kind"], as_index=False).head(1)
    color = {"cell composition": INK, "collection circumstance": MUTED,
             "technical": SUBTLE, "individual attribute": "#c6c6c6"}

    # 並び順は論文での登場順に固定する（groupby の結果に依存させない）
    order = [
        "Purified monocytes\nmicroarray, n=207",
        "Whole blood\nmicroarray, n=189",
        "Macrophages\nRNA-seq, n=79",
        "Whole blood (GTEx)\nRNA-seq, n=755",
    ]
    known = [c for c in order if c in set(df.cohort)]
    cohorts = known + [c for c in dict.fromkeys(df.cohort) if c not in order]
    # 4 コホートを横 1 列に並べると幅 12.6 インチになり、本文幅 6.5 インチでは
    # 52% に縮小されて軸ラベルが読めない。2 段組みにして幅を本文幅に収める。
    ncol = 2 if len(cohorts) > 2 else len(cohorts)
    nrow = -(-len(cohorts) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for extra in axes[len(cohorts):]:
        extra.set_visible(False)
    for ax, coh in zip(axes, cohorts):
        sub = df[df.cohort == coh].sort_values("excess")
        y = np.arange(len(sub))
        ax.barh(y, sub.excess, color=[color[k] for k in sub.kind],
                edgecolor=INK, linewidth=0.8, height=0.62)
        # 値を添える。祖先集団のようにほぼゼロの棒が「欠測」に見えるのを防ぐ
        for yy, v in zip(y, sub.excess):
            ax.text(max(v, 0) + 0.02, yy, f"{v:+.3f}", va="center", ha="left",
                    fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(sub.factor, fontsize=9, color=INK)
        ax.set_xlim(min(0, sub.excess.min()) - 0.03, max(0.35, sub.excess.max()) + 0.24)
        # コホートによって測れる分類の数が違う（精製単球には血球組成の共変量がない）。
        # 全パネルを 4 枠に固定すると、棒が 1 本のパネルが「3 本欠測」に見えるため、
        # 実際の本数に合わせる。棒の太さはパネル間で変わるが、x 軸は元々別尺度である。
        ax.set_ylim(-0.7, max(len(sub) - 0.3, 1.7))
        ax.set_title(coh, fontsize=10, color=INK, pad=8)
        ax.axvline(0, color=INK, linewidth=0.8)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK)
        ax.tick_params(colors=INK, labelsize=8.5, width=1.0)
        ax.grid(axis="x", color=HAIRLINE, linewidth=0.8)
        ax.set_axisbelow(True)

    # 表題・凡例・補足文はすべて図の内側に置く。図の外（y > 1 や y < 0）に置くと、
    # 保存時にその分だけ範囲が広がり、原稿で本文幅にそろえたときに全体が縮む。
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=INK, linewidth=0.8)
               for c in color.values()]
    footnote = _wrap("x-axis: excess R2 for PC1 (observed minus mean of 200 label "
                     "permutations). The top factor per category is shown for each cohort.",
                     fig.get_figwidth() - 0.25, 8.0)
    n_lines = footnote.count("\n") + 1
    line_frac = 8.0 * 1.5 / 72.0 / fig.get_figheight()
    bottom = line_frac * n_lines + 0.105      # 補足文 + 凡例 1 行 + 余裕
    fig.tight_layout(rect=(0, bottom, 1, 0.93))
    fig.suptitle("The factor that best explains PC1 changes from cohort to cohort",
                 fontsize=11.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.legend(handles, list(color), frameon=False, fontsize=8.5, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, bottom - 0.012))
    fig.text(0.012, bottom - 0.055, footnote, fontsize=8, color=SUBTLE,
             va="top", linespacing=1.5)
    _save(fig, "fig7_cross_cohort_attribution", cfg, laid_out=True)


def main() -> int:
    cfg = load_config("analysis")
    path = TABLES / "gene_set_metrics.csv"
    if not path.exists():
        print("gene_set_metrics.csv がない。pixi run analyze を先に実行する")
        return 1
    df = pd.read_csv(path)
    print(f"{len(df)} sets を読んだ")
    fig1_dissociation(df, cfg)   # パネル B の素材。本文からは参照しない
    from .fig1_overview import build as build_fig1_overview
    build_fig1_overview(cfg)     # 本文の図 1（4 パネルの入口図）
    fig2_by_family(df, cfg)
    fig3_method_agreement(df, cfg)
    fig4_qualification_map(df, cfg)
    fig5_retest(cfg)
    fig6_phenotype_rerandomized(cfg)
    fig7_cross_cohort_attribution(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
