"""高解像度の対照計算が、既存の 1 件ずつの実装と同じ値を出すことを確かめる。

対照を 20 個から 10,000 個に増やすために、平均ペア相関・ICC・Spearman を
「対照方向にまとめて計算する」実装に置き換えた。速さのために式を書き換えているので、
既存実装と一致しなければ論文の数字が静かに変わる。ここが本体の検査になる。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reliability.metrics import (icc_two_way, icc_two_way_batch,
                                     mean_pairwise_rho,
                                     mean_pairwise_rho_batch,
                                     mean_pairwise_rho_fast, rank_consistency,
                                     spearman_batch)
from src.scoring.methods import _rank_rows, _standardize_rows


def _S(n_genes: int, n_donors: int, seed: int) -> np.ndarray:
    """順位を取って標準化した行列（本番と同じ作り方）。各行の分散は厳密に 1 になる。"""
    x = np.random.default_rng(seed).normal(size=(n_genes, n_donors))
    return _standardize_rows(_rank_rows(x))


@pytest.mark.parametrize("g", [3, 5, 17, 60, 200])
def test_mean_pairwise_rho_fast_matches_reference(g: int) -> None:
    """行和による O(gn) 版が、行列積による実装と一致する。

    S の各行は分散 1 に標準化されているので c = S S^T / n の対角は 1 であり、
    平均ペア相関 = (||行和||^2 / n - g) / (g(g-1)) が成り立つ。この同値性が
    崩れると、対照 10,000 個の計算は全部無意味になる。
    """
    S = _S(400, 90, seed=g)
    rng = np.random.default_rng(g + 1)
    for _ in range(20):
        idx = rng.choice(S.shape[0], size=g, replace=True)
        assert mean_pairwise_rho_fast(S, idx) == pytest.approx(
            mean_pairwise_rho(S[idx]), abs=1e-11
        )


def test_mean_pairwise_rho_batch_matches_reference() -> None:
    """(対照, 位置) の行列版が、1 行ずつ計算した結果と一致する。"""
    S = _S(300, 70, seed=7)
    rng = np.random.default_rng(8)
    idx = rng.choice(S.shape[0], size=(50, 23))
    got = mean_pairwise_rho_batch(S, idx)
    want = np.array([mean_pairwise_rho(S[row]) for row in idx])
    assert np.allclose(got, want, atol=1e-11)


def test_icc_batch_matches_reference() -> None:
    """まとめ計算した ICC(2,1) が metrics.icc_two_way と一致する。"""
    rng = np.random.default_rng(11)
    y = rng.normal(size=(40, 56, 2))
    y[:, :, 1] += 0.6 * y[:, :, 0]  # 反復間に相関を持たせる
    got = icc_two_way_batch(y)
    want = np.array([icc_two_way(y[i]) for i in range(y.shape[0])])
    assert np.allclose(got, want, atol=1e-12)


def test_icc_is_scale_invariant() -> None:
    """ICC(2,1) は定数倍で不変。だから遺伝子数で割らず和のまま渡してよい。"""
    rng = np.random.default_rng(12)
    y = rng.normal(size=(8, 40, 2))
    assert np.allclose(icc_two_way_batch(y), icc_two_way_batch(y * 17.0), atol=1e-11)


def test_spearman_batch_matches_reference() -> None:
    """まとめ計算した Spearman が rank_consistency の rho と一致する。"""
    rng = np.random.default_rng(13)
    a = rng.normal(size=(60, 56))
    b = 0.4 * a + rng.normal(size=(60, 56))
    got = spearman_batch(a, b)
    want = np.array([rank_consistency(a[i], b[i])["rho"] for i in range(a.shape[0])])
    assert np.allclose(got, want, atol=1e-11)


def test_empirical_p_has_expected_floor() -> None:
    """経験 p の下限は 1/(B+1)。対照 20 個では 0.048 で、BH-FDR を通せない。

    この式が動機なので、境界の値を固定して残す。
    """
    assert (0 + 1) / (20 + 1) == pytest.approx(0.047619, abs=1e-6)
    assert (0 + 1) / (10_000 + 1) == pytest.approx(9.999e-5, rel=1e-3)


def test_disattenuation_matches_pairwise_definition() -> None:
    """減衰補正後の平均ペア相関が、定義どおりのペア計算と一致する。

    古典テスト理論の補正は r_ij / sqrt(rel_i * rel_j) である。
    実装は T_i = S_i / sqrt(rel_i) の行和で計算しており（対角が 1/rel_i になる）、
    式変形を間違えると補正量が静かにずれる。ここで定義に突き合わせる。
    """
    from src.reliability.attenuation_correction import disattenuated_mean_rho

    rng = np.random.default_rng(31)
    S = _S(200, 60, seed=31)
    rel = rng.uniform(0.3, 0.9, size=S.shape[0])
    inv_sqrt, inv = 1 / np.sqrt(rel), 1 / rel
    n = S.shape[1]
    for g in (3, 8, 25):
        idx = rng.choice(S.shape[0], size=g, replace=False)
        got = disattenuated_mean_rho(S, inv_sqrt, inv, idx)
        # 定義どおり: ペアごとに r_ij を出して sqrt(rel_i rel_j) で割り、平均する
        vals = []
        for x in range(g):
            for y in range(g):
                if x == y:
                    continue
                i, j = idx[x], idx[y]
                r = float(S[i] @ S[j]) / n
                vals.append(r / np.sqrt(rel[i] * rel[j]))
        assert got == pytest.approx(float(np.mean(vals)), abs=1e-11)


def test_disattenuation_is_identity_when_reliability_is_one() -> None:
    """信頼性が 1 なら補正は何もしない（平均ペア相関そのもの）。"""
    from src.reliability.attenuation_correction import disattenuated_mean_rho

    S = _S(150, 50, seed=33)
    one = np.ones(S.shape[0])
    idx = np.random.default_rng(34).choice(S.shape[0], size=17, replace=False)
    assert disattenuated_mean_rho(S, one, one, idx) == pytest.approx(
        mean_pairwise_rho(S[idx]), abs=1e-11
    )
