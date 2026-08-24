"""分散分解の実装を、答えが分かっている合成データで確かめる。

実データに当てる前にここを通す理由は 2 つ。
  1. 分解は完全分割であるべき（成分の合計が全体の説明率に一致する）。
     恒等式が崩れていれば符号か重みを間違えている。
  2. 直交した要因では共有分が 0、完全に重なった要因では固有分が 0 になるべき。
     この 2 つの極を外すと、実データで「共有が多い/少ない」を読めない。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reliability.commonality import (
    code_factor,
    commonality,
    decompose,
    dummy_block,
    r2,
    set_function,
    shapley,
    subsets,
)

N = 400


def blocks_from(labels: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    return {k: [dummy_block(v)] for k, v in labels.items()}


def test_code_factor_カテゴリなら数値でも分位に潰さない():
    """plate id のように数値に見えるカテゴリを守る。ここが崩れると
    既存解析で技術要因の主役だった 17 水準のプレート番号が 5 分位に潰れる。"""
    v = list(range(1, 18)) * 10          # 17 水準のプレート番号
    lab = code_factor(v, continuous=False, max_levels=10 ** 6)
    assert len(np.unique(lab)) == 17


def test_code_factor_連続変数は分位に割る():
    v = np.linspace(0, 1, 100)
    lab = code_factor(v, continuous=True, n_quantiles=5)
    assert len(np.unique(lab)) == 5
    assert np.bincount(lab).tolist() == [20] * 5


def test_code_factor_カテゴリは上位水準とその他に丸める():
    v = ["a"] * 50 + ["b"] * 30 + [f"rare{i}" for i in range(20)]
    lab = code_factor(v, continuous=False, max_levels=3)
    # 上位 2 水準 + その他 = 3 水準
    assert len(np.unique(lab)) == 3
    assert sorted(np.bincount(lab).tolist()) == [20, 30, 50]


def test_r2_説明変数なしはゼロ():
    y = np.random.default_rng(0).normal(size=N)
    assert r2(y, None) == 0.0


def test_部分集合の数は2のk乗():
    assert len(subsets(["a", "b", "c"])) == 8


def test_直交した要因では共有分がほぼゼロ():
    """要因が直交していれば、commonality の共有成分は 0 に寄る。

    ここは乱数で 2 要因を振ってはいけない。N=400 で独立に引くと標本相関が
    1/sqrt(N) = 0.05 程度残り、それだけで共有分が 0.05 出る（実測した）。
    直交を主張する検査なので、総当たりの均衡配置で厳密に直交させる。
    """
    gen = np.random.default_rng(1)
    idx = np.arange(N)
    a = idx // 4 % 4          # 4 水準 x 4 水準を各 25 回。相関は厳密に 0
    b = idx % 4
    assert abs(np.corrcoef(a, b)[0, 1]) < 1e-12
    y = 1.0 * a + 1.0 * b + gen.normal(scale=1.0, size=N)

    res = decompose(y, blocks_from({"A": a, "B": b}), gen, nperm=60)
    uniq_a = res["commonality"][frozenset(["A"])]
    uniq_b = res["commonality"][frozenset(["B"])]
    shared = res["commonality"][frozenset(["A", "B"])]

    assert uniq_a > 0.15 and uniq_b > 0.15
    assert abs(shared) < 0.02, f"直交しているのに共有分が {shared:.3f}"
    assert abs(shared) < 0.1 * min(uniq_a, uniq_b)
    # 直交なら一元配置と Shapley はほぼ一致する（二重計上が起きていない）
    assert abs(res["oneway"]["A"] - res["shapley"]["A"]) < 0.02


def test_重なった要因では固有分が落ち共有分に移る():
    """B を A のほぼ複製にすると、固有分が消えて共有分に集まる。"""
    gen = np.random.default_rng(2)
    a = gen.integers(0, 4, N)
    b = a.copy()                      # 完全に同じ情報
    y = 1.0 * a + gen.normal(scale=1.0, size=N)

    res = decompose(y, blocks_from({"A": a, "B": b}), gen, nperm=60)
    shared = res["commonality"][frozenset(["A", "B"])]

    assert abs(res["commonality"][frozenset(["A"])]) < 0.03
    assert abs(res["commonality"][frozenset(["B"])]) < 0.03
    assert shared > 0.15, f"重複しているのに共有分が {shared:.3f}"
    # 一元配置の足し算は共有分を 2 回数える。Shapley は 1 回に直す。
    assert res["oneway"]["A"] + res["oneway"]["B"] > 1.7 * res["total"]
    assert res["shapley"]["A"] == pytest.approx(res["shapley"]["B"], abs=0.03)


def test_成分の合計は全体の超過説明率に一致する():
    """完全分割であること。k=3 で恒等式を確かめる。"""
    gen = np.random.default_rng(3)
    a = gen.integers(0, 3, N)
    b = (a + gen.integers(0, 2, N)) % 3      # A と相関する
    c = gen.integers(0, 3, N)
    y = a + 0.5 * b + 0.8 * c + gen.normal(size=N)

    res = decompose(y, blocks_from({"A": a, "B": b, "C": c}), gen, nperm=60)
    assert len(res["commonality"]) == 7      # 2^3 - 1
    assert sum(res["commonality"].values()) == pytest.approx(res["total"], abs=1e-8)
    assert sum(res["shapley"].values()) == pytest.approx(res["total"], abs=1e-8)


def test_偶然水準は説明変数を増やすと上がる():
    """並べ替え R^2 が自由度で膨らむことを確かめる。これを引かないと分解が歪む。"""
    gen = np.random.default_rng(4)
    few = gen.integers(0, 2, N)
    many = gen.integers(0, 10, N)
    y = gen.normal(size=N)               # どの要因とも無関係

    table = set_function(y, blocks_from({"F": few, "M": many}), gen, nperm=80)
    assert table[frozenset(["M"])]["r2_chance"] > table[frozenset(["F"])]["r2_chance"]
    # 無関係な要因なので、超過はどちらも 0 の近くに落ちる
    assert abs(table[frozenset(["F"])]["excess"]) < 0.05
    assert abs(table[frozenset(["M"])]["excess"]) < 0.05


def test_shapley_は要因の順序に依存しない():
    gen = np.random.default_rng(5)
    a = gen.integers(0, 3, N)
    b = gen.integers(0, 3, N)
    y = a + b + gen.normal(size=N)
    g = {s: float(len(s)) for s in subsets(["A", "B"])}   # 単純加算的な値関数
    sh = shapley(g, ["A", "B"])
    assert sh["A"] == pytest.approx(1.0) and sh["B"] == pytest.approx(1.0)
    cm = commonality(g, ["A", "B"])
    assert sum(cm.values()) == pytest.approx(2.0)
