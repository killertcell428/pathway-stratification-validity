"""BH の独立性仮定を外した検査（Benjamini-Yekutieli）の性質を確かめる。

ここで守りたいのは 3 つ。
  1. BY の通過件数は BH の通過件数を上回らない（上回ったら向きの解釈が壊れる）。
  2. BY の閾値の厳しさは Σ_{i=1..m} 1/i に等しい（原稿に 8.27 倍と書いている）。
  3. 表の列と原稿の主張が対応している（コホートと量の組が揃っている）。

実データを読む検査は原稿の数値そのものなので audit_numbers に任せる。
ここは手続きの性質だけを合成データで見る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.reliability.dependency_fdr import FDR, TARGETS, _dissociation, _rows


def _n_pass(p: np.ndarray, method: str) -> int:
    return int((multipletests(p, alpha=FDR, method=method)[1] < FDR).sum())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_by_never_passes_more_than_bh(seed: int) -> None:
    """BY は BH より厳しいので、通過件数が増えることはない。"""
    rng = np.random.default_rng(seed)
    # 帰無 8 割 + 対立 2 割。対立は 0 に寄せる。
    p = np.concatenate([rng.uniform(size=800), rng.uniform(size=200) ** 8])
    assert _n_pass(p, "fdr_by") <= _n_pass(p, "fdr_bh")


def test_penalty_is_the_harmonic_sum() -> None:
    """閾値の厳しさは Σ 1/i。原稿が 8.27 倍と書いている根拠。"""
    for m in (2093, 2195, 2514):
        pen = float(np.sum(1.0 / np.arange(1, m + 1)))
        # BY の q 値は BH の q 値をちょうどこの倍率で持ち上げる
        rng = np.random.default_rng(m)
        # 一様な p だけだと q が全部 1 に切り上げられて倍率が見えない。
        # 小さい p を混ぜて、切り上げ前の領域を作る。
        p = np.concatenate([rng.uniform(size=m - 200),
                            rng.uniform(size=200) * 1e-6])
        q_bh = multipletests(p, alpha=FDR, method="fdr_bh")[1]
        q_by = multipletests(p, alpha=FDR, method="fdr_by")[1]
        free = q_by < 1.0  # 1 で切り上げられていない要素だけ比べる
        assert free.sum() > 0
        assert np.allclose(q_by[free] / q_bh[free], pen, rtol=1e-9)
    assert 8.2 < float(np.sum(1.0 / np.arange(1, 2196))) < 8.3


def test_by_is_conservative_on_pure_null() -> None:
    """一様な p だけを与えたら BY は 1 件も通さない。"""
    rng = np.random.default_rng(7)
    p = rng.uniform(size=2000)
    assert _n_pass(p, "fdr_by") == 0


def test_targets_cover_both_directions() -> None:
    """向きの注記が「多い」「少ない」の 2 値で、両方が入っていること。

    「少ない」しか無ければ BY は自動的に安全側なので検査の意味がない。
    条件効果（多いことが主張）が対象に入っているかを構造として固定する。
    """
    dirs = {d for _, _, _, d in TARGETS}
    assert dirs == {"多い", "少ない"}
    assert sum(1 for _, _, _, d in TARGETS if d == "多い") == 2


def test_rows_and_regions_agree_on_pass_counts() -> None:
    """行ごとの通過件数と、区画の割合が同じ判定から出ていること。

    _rows は件数、_dissociation は割合を出す。別々に計算しているので、
    条件効果の通過率が一致するかを突き合わせる。ずれたらどちらかが古い。
    """
    rows = pd.DataFrame(_rows())
    regs = pd.DataFrame(_dissociation())
    if rows.empty or regs.empty:
        pytest.skip("解析出力がまだない")
    for tag, label in (("主コホート", "主コホート 条件効果"),
                       ("GSE81046", "GSE81046 条件効果")):
        r = rows[rows["対象"] == label]
        g = regs[(regs["コホート"] == tag) & (regs["量"] == "条件効果あり(%)")]
        if not len(r) or not len(g):
            continue
        assert 100 * float(r.iloc[0]["BY 通過率"]) == pytest.approx(
            float(g.iloc[0]["BY"]), abs=0.05)
        assert 100 * float(r.iloc[0]["BH 通過率"]) == pytest.approx(
            float(g.iloc[0]["BH"]), abs=0.05)
