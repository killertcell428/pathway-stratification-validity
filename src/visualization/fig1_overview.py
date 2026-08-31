"""Figure 1 を「論文の入口」として 4 パネルで作る。

なぜ作り直すか
  旧 Figure 1 は条件効果 x 個人間共変動の散布図 1 枚だった。図としては正しいが、
  査読者が最初に見たときに「で、これの何が意外なのか」が本文を読まないと分からない。
  結果が増えた（ファミリー・サイズ・手法・帰属・正規化・反復測定・表現型）ので、
  読み進めるうちに最初の発見のインパクトが薄まる危険がある。
  そこで Figure 1 を「何を問い、何が分かり、別データでも起きたか、だから何を提案するか」
  の 4 段に組み替える。

  図の総数は増やさない。旧 Figure 1 の散布図はパネル B に吸収するので、
  論文全体は Figure 1-7 のままである（投稿先の図数制限に触れない）。

パネルの役割
  A  なぜ問題なのか。同じセットを半分に割ると同じ 5 名の順位が一致しないことを示す
     模式図（データは使わない）
  B  主コホートの実測（旧 Figure 1 の散布図を C と同じ 4 区画に整理し、86.7% を大きく出す）
  C  2 コホートの区画割合（1 コホートの偶然ではないことを示す）

  **B と C は同じ 4 区画で描く。**B を「条件効果なし / 条件効果のみ / 両方」の 3 色に
  していた時期があり、C が 4 区画だったため同じ図の中で分割が食い違っていた。
  区画の定義は _regions() の 1 箇所に集約し、B の凡例が C の凡例も兼ねる。

  **4 番目のパネルは置かない。**一度は「要件を順に課すと何件残るか」
  （2,195 -> 1,992 -> 90）を入れたが、90 件は B の 86.7% の裏返しであって
  新しい情報が反復測定の 17 件だけだった。その 17 件は 3.9 節と反復測定の図が
  正面から扱うので、入口図で先出しする必要がない。
  提案する 4 項目は 4.2 節と Figure 4 に置く。A で「なぜ問題か」、
  B で「実際にそうなっている」、C で「1 コホートの偶然ではない」まで言えば、
  初読者が「何を問い、何が分かった論文か」を取れる。

レイアウトで気をつけたこと
  A は左右対比なので横幅が要る。C は縦積み棒 2 本なので半幅で足りる。
  そこで A を上段全幅、B と C を下段に置き、読む順と配置の順を一致させた。

  **C には凡例を置かない。**色はパネル B と共通で、B 側に凡例がある。
  C の軸の外に凡例を出すと保存幅が本文幅（6.5 インチ）を超え、原稿に貼るときに
  図全体が縮んで軸内の文字が読めなくなる（figures._save のコメント参照）。

  文字はすべて textwrap でパネル幅に収める。データ座標 10 単位が何インチに
  当たるかは余白設定で決まるので、幅を変えたら折り返し幅も見直す。

出力: results/figures/fig1_overview.{png,svg}
"""

from __future__ import annotations

import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..common import TABLES, load_config  # noqa: E402
from .figures import (BLUE, DOC_WIDTH_IN, HAIRLINE, INK, MUTED, SUBTLE,  # noqa: E402
                      _save, _style, coherence_pass)

# 4 区画の色。パネル B と C で共通に使う。条件効果のみ（最大区画）を最も濃く出す。
C_ONLY = INK           # 条件効果あり・共変動なし
C_BOTH = BLUE          # 両方
C_COH = SUBTLE         # 共変動のみ
C_NEITHER = "#d0d0d0"  # どちらもなし


def _regions(df: pd.DataFrame, common_null: bool = True) -> dict[str, float]:
    """4 区画の割合（%）。合計は 100 になる。

    `common_null=True` では条件効果も**同じ対照への超過**で判定する。
    ゼロ帰無（delta_q）と対照超過（cond_null_q）を交差させると、2 つの判定が
    別の帰無を持つため「条件効果のみ」が大きく出る。それは注釈セット固有の
    乖離の大きさではなく判定基準の非対称性を映すので、本文の図は共通対照で描く。
    ゼロ帰無版は補遺の図に回す。
    """
    col = "cond_null_q" if common_null and "cond_null_q" in df.columns else "delta_q"
    cond = df[col] < 0.05
    coh = coherence_pass(df)
    return {
        "cond_only": 100 * float((cond & ~coh).mean()),
        "both": 100 * float((cond & coh).mean()),
        "coh_only": 100 * float((~cond & coh).mean()),
        "neither": 100 * float((~cond & ~coh).mean()),
        "n": len(df),
        "cond": 100 * float(cond.mean()),
        "coh": 100 * float(coh.mean()),
    }


def _panel_a(ax) -> None:
    """概念図: 同じ用途に見えて要求が違うこと、そして 2 番目が満たされないと何が壊れるか。

    左は争いのない側である。構成遺伝子が摂動で同じ向きに動けばスコアは動くので、
    群としての応答は検出できる。細い線が遺伝子、太い線がセットスコア。

    右がこのパネルの主眼である。同じセットを 2 つに割って同じ 5 名を採点すると、
    順位が一致しない。**「遺伝子が共変動しない」という言い方では
    「個人差はノイズが多い」で流されてしまう。**壊れ方を出す方が伝わる。
    どの遺伝子を使うかという任意の選択で「誰が高いか」が変わるなら、
    その順位は個人の性質ではない。3.5 節（手法間で順位が食い違う）と
    折半法の実測が、この模式図に対応する測定結果である。

    2 本の順位リストを線でつないだ図（傾斜図）はデータに見えないので、
    模式図であることが形から分かる。念のため schematic とも書く。
    """
    ax.set_xlim(0, 10)
    # 上端は見出し 2 行ぶんを取る。右の見出しは半幅に収まらないので折り返す。
    ax.set_ylim(0.20, 4.35)
    ax.axis("off")

    # --- 左: 争いのない側。遺伝子が同じ向きに動けばスコアは動く ---
    ax.text(1.70, 4.26, "What group comparison needs", fontsize=8.4, color=INK,
            ha="center", va="top", weight="bold")
    xs = [0.95, 2.35]
    for base, top in ((1.42, 2.30), (1.28, 2.16), (1.56, 2.44)):
        ax.plot(xs, [base, top], color=SUBTLE, lw=0.9, zorder=1)
    ax.plot(xs, [1.42, 2.30], color=INK, lw=2.2, marker="o", ms=4.2, zorder=2)
    ax.text(xs[1] + 0.12, 2.30, "set\nscore", fontsize=6.8, color=INK,
            va="center", linespacing=1.3)
    for x, lab in zip(xs, ["rest", "stimulus"]):
        ax.text(x, 1.06, lab, fontsize=7.2, color=MUTED, ha="center", va="top")
    ax.text(1.55, 0.72, "member genes move the same way\n"
                        "-> the response is detectable",
            fontsize=7.4, color=INK, ha="center", va="top", linespacing=1.5)

    # --- 右: 同じセットを半分に割ると、同じ 5 名の順位が一致しない ---
    ax.text(6.85, 4.26,
            "What ranking individuals additionally requires"
            "\n- and how it can fail",
            fontsize=8.4, color=INK, ha="center", va="top", weight="bold",
            linespacing=1.35)
    left_x, right_x = 5.55, 8.15
    rank_y = [2.62, 2.24, 1.86, 1.48, 1.10]
    half_a = ["P3", "P1", "P5", "P2", "P4"]
    half_b = ["P2", "P4", "P1", "P3", "P5"]
    # 順位が最も大きく動く 2 名を青で追跡できるようにする
    track = {"P3": BLUE, "P2": BLUE}

    # 見出しは 1 行にする。2 行にすると最上位の順位の箱と重なる。
    ax.text(left_x, 2.95, "half the set", fontsize=7.2, color=MUTED,
            ha="center", va="center")
    ax.text(right_x, 2.95, "the other half", fontsize=7.2, color=MUTED,
            ha="center", va="center")
    ax.annotate("", xy=(right_x - 0.68, 2.95), xytext=(left_x + 0.62, 2.95),
                arrowprops=dict(arrowstyle="-|>", color=HAIRLINE, lw=1.2))

    for i, p in enumerate(half_a):
        j = half_b.index(p)
        col = track.get(p, "#b8b8b8")
        ax.plot([left_x + 0.28, right_x - 0.28], [rank_y[i], rank_y[j]],
                color=col, lw=1.6 if p in track else 1.0,
                zorder=3 if p in track else 2)
    for x, order in ((left_x, half_a), (right_x, half_b)):
        for i, p in enumerate(order):
            col = track.get(p, MUTED)
            ax.text(x, rank_y[i], p, fontsize=7.6, color=col,
                    weight="bold" if p in track else "normal",
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor=HAIRLINE, lw=0.7))
    ax.text(left_x - 0.62, rank_y[0], "high", fontsize=6.8, color=SUBTLE,
            ha="right", va="center")
    ax.text(left_x - 0.62, rank_y[-1], "low", fontsize=6.8, color=SUBTLE,
            ha="right", va="center")

    ax.text(6.85, 0.72, "the same gene set, split in two: the ordering disagrees\n"
                        "-> who is 'high' depends on which genes you use",
            fontsize=7.4, color=INK, ha="center", va="top", linespacing=1.5)

    # 模式図であることを明示する。タイトルと重ならない左下に置く。
    ax.text(-0.070, 0.905, "schematic", transform=ax.transAxes, fontsize=7.0,
            color=SUBTLE, style="italic", ha="left", va="top")
    ax.text(0.42, 1.86, "expression", fontsize=6.8, color=SUBTLE, rotation=90,
            ha="center", va="center")
    ax.annotate("", xy=(0.66, 2.46), xytext=(0.66, 1.26),
                arrowprops=dict(arrowstyle="-|>", color=SUBTLE, lw=0.9))
    ax.plot([3.95, 3.95], [0.50, 3.20], color=HAIRLINE, lw=1.2)


def _panel_b(ax, df: pd.DataFrame, reg: dict[str, float]) -> None:
    """主コホートの散布図。旧 Figure 1 を C と同じ 4 区画に整理して吸収する。"""
    # 両軸を「同じ 10,000 対照に対する超過の q 値」に取る。生の |d| と生の共変動を
    # 軸にすると、2 つの量が別の帰無で判定されていることが図から見えない。
    d = df.dropna(subset=["cond_null_q", "null_q", "var_null_q"]).copy()
    eps = 1e-6
    d["x"] = -np.log10(d["cond_null_q"].clip(lower=eps))
    d["y"] = -np.log10(d[["null_q", "var_null_q"]].max(axis=1).clip(lower=eps))
    x, y = d["x"], d["y"]
    cond = d["cond_null_q"] < 0.05
    coh = coherence_pass(d)
    thr = -np.log10(0.05)
    ax.axvline(thr, color=INK, linewidth=0.9, zorder=1)
    ax.axhline(thr, color=INK, linewidth=0.9, zorder=1)
    # パネル C と同じ 4 区画に揃える。
    # 以前は「条件効果なし」を 1 色にまとめていたが、C は 4 区画に分けていたので、
    # 同じ図の中で分割が食い違っていた。共変動のみの区画は主コホートで 0.5%
    # （11 セット）しかないが、区画として存在するなら図でも独立させる。
    h_nei = ax.scatter(x[~cond & ~coh], y[~cond & ~coh], s=4, c=C_NEITHER,
                       alpha=0.55, linewidths=0, label="neither")
    h_con = ax.scatter(x[cond & ~coh], y[cond & ~coh], s=5, c=C_ONLY, alpha=0.38,
                       linewidths=0, label="condition effect only")
    h_coh = ax.scatter(x[~cond & coh], y[~cond & coh], s=15, c=C_COH, alpha=0.95,
                       linewidths=0, label="coherence only")
    h_bot = ax.scatter(x[cond & coh], y[cond & coh], s=12, c=C_BOTH, alpha=0.95,
                       linewidths=0, label="condition effect + coherence")

    # 最大区画の割合を図中に大きく出す。ここが入口図の要点である。
    ax.text(0.70, 0.74, f"{reg['neither']:.1f}%", transform=ax.transAxes,
            fontsize=18, color=MUTED, ha="right", va="top", weight="bold")
    ax.text(0.70, 0.625, "clear neither set of\nmatched controls",
            transform=ax.transAxes, fontsize=7.2, color=MUTED, ha="right", va="top",
            linespacing=1.45)

    for _, r in d[d["family"] == "anchor"].iterrows():
        ax.scatter([r["x"]], [r["y"]], s=44,
                   facecolors="none", edgecolors=BLUE, linewidths=1.2, zorder=5)
        ax.annotate(r["set"].split("|")[1].replace("_", " "),
                    (r["x"], r["y"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=6.8,
                    color=BLUE, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_xlabel("Condition effect vs matched controls  (−log₁₀ q)", fontsize=8.8, color=INK)
    ax.set_ylabel("Coherence vs matched controls  (−log₁₀ q)", fontsize=8.8, color=INK)
    # 凡例の並びは C の積み上げ順（下から 条件のみ / 両方 / 共変動のみ / どちらも）に
    # そろえる。描画順は重なりの都合で別なので、ここで並べ替える。
    ax.legend(handles=[h_con, h_bot, h_coh, h_nei],
              frameon=True, facecolor="white", edgecolor=HAIRLINE, framealpha=0.92,
              fontsize=6.6, loc="upper left", handletextpad=0.3,
              borderpad=0.15, labelspacing=0.25, markerscale=1.8)
    _style(ax)


def _panel_c(ax, regs: list[tuple[str, dict[str, float]]]) -> None:
    """2 コホートの区画割合。縦積み棒 2 本なので半幅で収まる。

    凡例は置かない（パネル B と同色。理由はモジュール冒頭）。
    """
    order = [("cond_only", C_ONLY), ("both", C_BOTH),
             ("coh_only", C_COH), ("neither", C_NEITHER)]
    xs = [0.26, 0.80]
    half = 0.14
    for (label, reg), xx in zip(regs, xs):
        bottom = 0.0
        outside: list[tuple[float, float, str]] = []
        for key, col in order:
            v = reg[key]
            ax.bar(xx, v, bottom=bottom, width=2 * half, color=col, linewidth=0)
            # 4 区画すべての値を出す。パネル B の凡例に 4 区画あるのに C で
            # 一部の数値が消えていると、同じ分割で描いていることが伝わらない。
            # 帯に収まらない区画は棒の右に引き出す。
            if v >= 7:
                ax.text(xx, bottom + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8.2,
                        weight="bold" if key == "neither" else "normal",
                        color="white" if col in (C_ONLY, C_BOTH) else INK)
            else:
                outside.append((bottom + v / 2, v, col))
            bottom += v

        # 引き出したラベルどうしが重ならないよう、下から最小間隔をあけて置き直す。
        # 主コホートでは 4.1% と 0.5% が 2.3 ポイントしか離れておらず、
        # そのまま書くと重なる。
        gap, prev = 6.0, -99.0
        for y_true, v, col in outside:
            y = max(y_true, prev + gap)
            prev = y
            ax.plot([xx + half, xx + half + 0.05], [y_true, y], color=col, lw=0.7,
                    solid_capstyle="butt", clip_on=False)
            ax.text(xx + half + 0.07, y, f"{v:.1f}%", ha="left", va="center",
                    fontsize=7.0, color=col)
        ax.text(xx, -2.5, label, fontsize=7.4, color=INK,
                ha="center", va="top", linespacing=1.45)

    ax.set_xlim(0, 1.30)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_ylabel("Percentage of gene sets", fontsize=8.8, color=INK)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=8.2)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def build(cfg: dict) -> None:
    main_df = pd.read_csv(TABLES / "gene_set_metrics.csv")
    rna_path = TABLES / "gse81046" / "gene_set_metrics.csv"
    reg_main = _regions(main_df)
    # ラベルは 2 行に分ける。1 行にすると 2 本の棒のあいだで文字が重なる。
    regs = [("Monocytes\narray, 207", reg_main)]
    if rna_path.exists():
        regs.append(("Macrophages\nRNA-seq, 79",
                     _regions(pd.read_csv(rna_path))))

    fig = plt.figure(figsize=(DOC_WIDTH_IN, 6.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.14, 1.70],
                          width_ratios=[1.42, 1.0],
                          hspace=0.30, wspace=0.38,
                          left=0.100, right=0.980, top=0.905, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    _panel_a(ax_a)
    _panel_b(ax_b, main_df, reg_main)
    _panel_c(ax_c, regs)

    for ax, letter, dx, dy in ((ax_a, "A", -0.072, 0.98), (ax_b, "B", -0.130, 1.02),
                               (ax_c, "C", -0.185, 1.02)):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12, color=INK,
                weight="bold", va="bottom", ha="left")

    fig.text(0.010, 0.988,
             "Against matched controls, most gene sets clear neither requirement",
             fontsize=10.4, color=INK, va="top", weight="bold")

    _save(fig, "fig1_overview", cfg, laid_out=True)


def main() -> int:
    cfg = load_config("analysis")
    if not (TABLES / "gene_set_metrics.csv").exists():
        print("gene_set_metrics.csv がない。pixi run analyze を先に実行する")
        return 1
    build(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
