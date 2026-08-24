"""感度分析が「どの腕のファイルを読んでいるか」を取り違えないことを確かめる。

被覆率フィルタの集計は results/tables/gene_set_metrics*.csv を走査していた。
接尾辞のないファイル（正本）と、被覆率の腕（_cov0.3 など）以外にも
発現フィルタ分位の腕（_p40 / _p60）が同じ名前で始まるため、
「_cov に一致しなければ既定」という判定では _p60 が正本の行を上書きしていた。
表 6 の既定行が 2,195 セットではなく 1,713 セットになる、という形で実測した。
腕が増えるたびに同じ事故が起きうるので、名前の判定をここで固定する。
"""

from __future__ import annotations

import pytest

from src.reliability.coverage_sensitivity import label_for


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("gene_set_metrics", "0.6 (既定)"),      # 正本
        ("gene_set_metrics_cov0.3", "0.3"),
        ("gene_set_metrics_cov0.5", "0.5"),
        ("gene_set_metrics_cov0.8", "0.8"),
        ("gene_set_metrics_p40", None),          # 発現フィルタ分位の腕。対象外
        ("gene_set_metrics_p60", None),          # ここが既定を上書きしていた
        ("gene_set_metrics_cpm0.5", None),       # RNA-seq 側の腕
        ("gene_set_metrics_tpm", None),
        ("gene_set_metrics_quantile", None),
    ],
)
def test_label_for(stem: str, expected: str | None) -> None:
    assert label_for(stem) == expected


def test_default_label_is_unique() -> None:
    """既定ラベルを返すのは接尾辞なしの 1 件だけである。

    複数が既定ラベルを返すと、辞書代入で静かに上書きが起きる。
    """
    stems = ["gene_set_metrics", "gene_set_metrics_cov0.3", "gene_set_metrics_p40",
             "gene_set_metrics_p60", "gene_set_metrics_cpm2.0"]
    assert sum(label_for(s) == "0.6 (既定)" for s in stems) == 1
