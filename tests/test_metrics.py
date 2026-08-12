"""合成データで指標の妥当性を確かめる。

本研究の主張は「条件効果は出るのに個人層別化には使えないセットがある」なので、
その 2 状況を作り分けた合成データで、指標が両者を区別できることを先に保証する。
実データで同じパターンが出たときに、それが実装の副作用でないと言えるようにする。

  module    : 個人ごとの潜在因子を共有する遺伝子群（個人層別化に使えるべき）
  incoherent: 個人間では独立だが、摂動で全遺伝子が同方向に動く遺伝子群
              （条件効果は出るが個人層別化には使えない）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reliability.metrics import (
    condition_effect,
    cronbach_alpha,
    direction_concordance,
    empirical_null,
    icc_two_way,
    mean_pairwise_rho,
    rank_consistency,
    split_half_reliability,
)
from src.scoring.methods import ScoringContext, score_set

N_IND = 200
N_GENE = 10
SHIFT = 1.0


@pytest.fixture(scope="module")
def synthetic():
    rs = np.random.default_rng(0)
    latent = rs.normal(size=N_IND)

    module_rest = np.array([0.9 * latent + 0.4 * rs.normal(size=N_IND) for _ in range(N_GENE)])
    incoherent_rest = rs.normal(size=(N_GENE, N_IND))

    # 摂動: どちらのセットも全遺伝子が同じ向きに動く（条件効果は同等）
    module_pert = module_rest + SHIFT + 0.2 * rs.normal(size=module_rest.shape)
    incoherent_pert = incoherent_rest + SHIFT + 0.2 * rs.normal(size=incoherent_rest.shape)

    return {
        "module_rest": module_rest,
        "module_pert": module_pert,
        "incoherent_rest": incoherent_rest,
        "incoherent_pert": incoherent_pert,
    }


def _std_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, x.shape[1] + 1, dtype=float)
    return (ranks - ranks.mean(axis=1, keepdims=True)) / ranks.std(axis=1, keepdims=True)


def test_internal_consistency_separates_module_from_incoherent(synthetic):
    rho_mod = mean_pairwise_rho(_std_ranks(synthetic["module_rest"]))
    rho_inc = mean_pairwise_rho(_std_ranks(synthetic["incoherent_rest"]))
    assert rho_mod > 0.5, f"共変動する遺伝子群で低すぎる: {rho_mod}"
    assert abs(rho_inc) < 0.1, f"独立な遺伝子群で高すぎる: {rho_inc}"


def test_cronbach_alpha_separates(synthetic):
    a_mod = cronbach_alpha(_std_ranks(synthetic["module_rest"]))
    a_inc = cronbach_alpha(_std_ranks(synthetic["incoherent_rest"]))
    assert a_mod > 0.8
    assert a_inc < 0.3


def test_split_half_separates(synthetic):
    gen = np.random.default_rng(1)
    r_mod, sb_mod = split_half_reliability(_std_ranks(synthetic["module_rest"]), 50, gen)
    r_inc, sb_inc = split_half_reliability(_std_ranks(synthetic["incoherent_rest"]), 50, gen)
    assert r_mod > 0.5 and sb_mod > r_mod
    assert abs(r_inc) < 0.15


def test_condition_effect_is_similar_for_both(synthetic):
    """ここが本研究の要点。条件効果だけ見ると 2 つのセットは区別できない。"""
    def eff(rest, pert):
        pooled = np.hstack([rest, pert])
        z = (pooled - pooled.mean(axis=1, keepdims=True)) / pooled.std(axis=1, keepdims=True)
        n = rest.shape[1]
        return condition_effect(z[:, :n].mean(axis=0), z[:, n:].mean(axis=0))

    e_mod = eff(synthetic["module_rest"], synthetic["module_pert"])
    e_inc = eff(synthetic["incoherent_rest"], synthetic["incoherent_pert"])
    assert e_mod["delta_p"] < 1e-10 and e_inc["delta_p"] < 1e-10
    assert e_mod["delta_mean"] > 0.5 and e_inc["delta_mean"] > 0.5


def test_direction_concordance_high_for_both(synthetic):
    d_mod = direction_concordance(synthetic["module_rest"], synthetic["module_pert"])
    d_inc = direction_concordance(synthetic["incoherent_rest"], synthetic["incoherent_pert"])
    assert d_mod["frac_same_direction"] == 1.0
    assert d_inc["frac_same_direction"] == 1.0


def test_rank_consistency_bounds():
    x = np.random.default_rng(2).normal(size=100)
    same = rank_consistency(x, x)
    assert same["rho"] == pytest.approx(1.0)
    assert same["tertile_swap"] == 0.0
    flipped = rank_consistency(x, -x)
    assert flipped["rho"] == pytest.approx(-1.0)


def test_empirical_null_p_is_bounded():
    out = empirical_null(0.5, list(np.random.default_rng(3).normal(0, 0.1, 100)))
    assert 0 < out["null_p"] <= 1
    assert out["null_z"] > 3


def test_scoring_methods_agree_on_module(synthetic):
    """共変動するセットでは 4 手法が概ね一致する（手法差は主張の交絡にならない）。"""
    expr = pd.DataFrame(
        synthetic["module_rest"],
        index=[f"G{i}" for i in range(N_GENE)],
        columns=[f"S{j}" for j in range(N_IND)],
    )
    # スコア対象外の背景遺伝子を足す（singscore はサンプル内順位を使うため必要）
    bg = pd.DataFrame(
        np.random.default_rng(4).normal(size=(500, N_IND)),
        index=[f"BG{i}" for i in range(500)],
        columns=expr.columns,
    )
    ctx = ScoringContext(pd.concat([expr, bg]))
    genes = list(expr.index)
    scores = {m: score_set(ctx, genes, m) for m in ("zmean", "singscore", "plage", "meanrank")}
    base = scores["zmean"]
    for m, s in scores.items():
        r = np.corrcoef(base, s)[0, 1]
        assert r > 0.8, f"{m} が zmean と乖離しすぎる: r={r:.3f}"


def test_icc_separates_stable_from_unstable():
    """ICC は「同一個人を測り直したとき順位も水準も保たれるか」を測る。"""
    rs = np.random.default_rng(6)
    trait = rs.normal(size=80)
    stable = np.column_stack([trait + 0.2 * rs.normal(size=80), trait + 0.2 * rs.normal(size=80)])
    unstable = rs.normal(size=(80, 2))
    assert icc_two_way(stable) > 0.85
    assert abs(icc_two_way(unstable)) < 0.25


def test_icc_penalizes_level_shift():
    """順位が完全に一致していても、2 回目が全体的にずれていれば ICC は下がる。"""
    rs = np.random.default_rng(7)
    trait = rs.normal(size=60)
    same = np.column_stack([trait, trait])
    shifted = np.column_stack([trait, trait + 2.0])
    assert icc_two_way(same) > 0.99
    assert icc_two_way(shifted) < icc_two_way(same)
    # 順位相関は水準のずれを検出できない（だから ICC を併記する必要がある）
    assert rank_consistency(shifted[:, 0], shifted[:, 1])["rho"] == pytest.approx(1.0)


def test_scoring_returns_nan_for_missing_genes():
    expr = pd.DataFrame(
        np.random.default_rng(5).normal(size=(5, 20)),
        index=[f"G{i}" for i in range(5)],
        columns=[f"S{j}" for j in range(20)],
    )
    ctx = ScoringContext(expr)
    assert np.isnan(score_set(ctx, ["NOT_PRESENT"], "zmean")).all()
