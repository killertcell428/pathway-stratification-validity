"""主成分の説明分散を、要因群の固有分と共有分に分ける（commonality / Shapley）。

なぜ要るか
  既存の帰属解析（batch_check / technical_axes_check / gtex_attribution）は、
  要因を 1 つずつ主成分に当てた一元配置 R^2 を出し、分類ごとに足している。
  要因どおしが相関していると、共有された分散が**各要因に二重計上される**。

  たとえば全血では好中球比率と虚血時間が相関しうる。組成の合計 1.510 と
  技術の合計 0.594 を並べても、両者が同じ分散を別々に数えているなら
  「組成が技術の 2.5 倍」という読みは成立しない。論文の
  「個人属性は組成の 12 分の 1」も同じ弱点を持つ。

  ここでは要因群を同時に投入した重回帰の R^2 を全部分集合について求め、
  そこから固有分と共有分を復元する。

2 つの分解を出す
  **commonality**  2^k - 1 個の成分（固有 k 個と共有 2^k-1-k 個）に完全分割する。
    共有分そのものを見たいときに使う。成分は負になりうる（抑制効果）。
  **Shapley**      各要因群に 1 つの値を割り当てる。共有分を、投入順序の全並べ替えで
    平均して等分する。負になりにくく、合計が全体の R^2 に一致するので
    「組成 : 技術 : 個人属性」を比で語りたいときはこちらを使う。

値関数について
  R^2 は説明変数を増やすだけで上がる。既存解析と同じく、ラベルを並べ替えた
  偶然水準を引いた**超過 R^2** を値関数にする。分解はこの値関数について線形なので、
  成分の合計は「全要因を入れたときの超過 R^2」にちょうど一致する。

  偶然水準の作り方は、各要因のダミー行列の**行を独立に並べ替える**。
  ラベルを並べ替えてからダミー化するのと同値で、こちらのほうが速い。

離散化の規約（既存解析と揃えていない点があるので注記する）
  連続変数は 5 分位。カテゴリ変数は**上位 9 水準 + その他の 10 水準に丸める**。
  丸めるのは、SMNABTCH のように 335 水準ある要因を素で入れると
  重回帰が飽和して分解が意味を失うため。既存の一元配置は丸めていないので、
  比較を成立させるために、この符号化での一元配置 R^2 も同時に出す。
"""

from __future__ import annotations

from itertools import chain, combinations
from math import factorial

import numpy as np
import pandas as pd

MAX_LEVELS = 10       # カテゴリ変数の水準の上限（上位 9 + その他）
N_QUANTILES = 5       # 連続変数の分位数


def code_factor(v, continuous: bool, max_levels: int = MAX_LEVELS,
                n_quantiles: int = N_QUANTILES) -> np.ndarray:
    """要因を整数ラベルにする。連続なら分位、カテゴリなら頻度上位 + その他。

    連続かどうかは**呼び出し側が明示する**。値が数値に見えるかで推測してはいけない。
    GSE35846 の plate id は 17 枚のプレート番号で、数値に見えるが順序も間隔も
    意味を持たないカテゴリである。推測に任せると 5 分位に潰れ、
    既存解析で技術要因の主役だった plate id の寄与（超過 0.175）が消える（実測した）。
    """
    s = pd.Series(np.asarray(v).ravel())
    if continuous:
        num = pd.to_numeric(s, errors="coerce")
        if num.nunique() > n_quantiles:
            lab = pd.qcut(num.rank(method="first"), n_quantiles, labels=False)
            return lab.fillna(-1).to_numpy(dtype=np.int64)
    cat = s.astype(str)
    keep = cat.value_counts().index[:max_levels - 1]
    cat = cat.where(cat.isin(keep), "__other__")
    return pd.factorize(cat)[0].astype(np.int64)


def dummy_block(labels: np.ndarray) -> np.ndarray:
    """整数ラベルを、最初の水準を落としたダミー行列にする（切片は別に足す）。"""
    d = pd.get_dummies(pd.Series(labels), drop_first=True)
    return d.to_numpy(dtype=np.float64)


def r2(y: np.ndarray, x: np.ndarray | None) -> float:
    """切片つき最小二乗の決定係数。x が空なら 0。"""
    tss = float(((y - y.mean()) ** 2).sum())
    if tss == 0:
        return np.nan
    if x is None or x.shape[1] == 0:
        return 0.0
    a = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    rss = float(((y - a @ beta) ** 2).sum())
    return 1.0 - rss / tss


def subsets(keys: list[str]):
    """空集合を含む全部分集合を frozenset で返す。"""
    return [frozenset(c) for c in chain.from_iterable(
        combinations(keys, r) for r in range(len(keys) + 1))]


def set_function(y: np.ndarray, blocks: dict[str, list[np.ndarray]], gen,
                 nperm: int = 200) -> dict[frozenset, dict]:
    """全部分集合について、超過 R^2（観測 − 偶然水準の平均）を求める。

    blocks は {群の名前: [その群に属する要因のダミー行列, ...]}。
    偶然水準は、各ダミー行列の行を独立に並べ替えて作る。
    """
    keys = list(blocks)
    out: dict[frozenset, dict] = {}
    n = len(y)
    for sub in subsets(keys):
        mats = [m for k in sub for m in blocks[k]]
        x = np.column_stack(mats) if mats else None
        obs = r2(y, x)
        if not mats:
            out[sub] = {"r2": 0.0, "r2_chance": 0.0, "excess": 0.0, "n_cols": 0}
            continue
        perm = np.empty(nperm)
        for b in range(nperm):
            shuffled = [m[gen.permutation(n)] for m in mats]
            perm[b] = r2(y, np.column_stack(shuffled))
        chance = float(np.nanmean(perm))
        out[sub] = {"r2": obs, "r2_chance": chance, "excess": obs - chance,
                    "n_cols": int(x.shape[1])}
    return out


def shapley(g: dict[frozenset, float], keys: list[str]) -> dict[str, float]:
    """各群の Shapley 値。合計は g(全体) に一致する。"""
    k = len(keys)
    out = {}
    for i in keys:
        rest = [x for x in keys if x != i]
        total = 0.0
        for r in range(len(rest) + 1):
            w = factorial(r) * factorial(k - r - 1) / factorial(k)
            for c in combinations(rest, r):
                s = frozenset(c)
                total += w * (g[s | {i}] - g[s])
        out[i] = total
    return out


def commonality(g: dict[frozenset, float], keys: list[str]) -> dict[frozenset, float]:
    """commonality 成分。空でない全部分集合について 1 つ、合計は g(全体)。

    C(T) = − Σ_{S ⊆ T} (−1)^{|S|} g((F \\ T) ∪ S)

    k=3 の教科書的な式（U_A = R²(ABC) − R²(BC)、
    C_AB = R²(AC) + R²(BC) − R²(C) − R²(ABC) …）をそのまま一般化したもの。
    """
    F = frozenset(keys)
    out = {}
    for sub in subsets(keys):
        if not sub:
            continue
        acc = 0.0
        for r in range(len(sub) + 1):
            for c in combinations(sorted(sub), r):
                acc += (-1) ** r * g[(F - sub) | frozenset(c)]
        out[sub] = -acc
    return out


def decompose(y: np.ndarray, blocks: dict[str, list[np.ndarray]], gen,
              nperm: int = 200) -> dict:
    """1 本の主成分について、一元配置・Shapley・commonality をまとめて返す。"""
    keys = list(blocks)
    table = set_function(y, blocks, gen, nperm)
    g = {s: v["excess"] for s, v in table.items()}
    full = frozenset(keys)

    # 同じ符号化での一元配置（既存解析の足し算に相当するもの）。
    # これと Shapley の差が、二重計上されていた量である。
    oneway = {k: g[frozenset([k])] for k in keys}

    sh = shapley(g, keys)
    cm = commonality(g, keys)
    assert abs(sum(sh.values()) - g[full]) < 1e-8, "Shapley の合計が全体と一致しない"
    assert abs(sum(cm.values()) - g[full]) < 1e-8, "commonality の合計が全体と一致しない"
    return {"subsets": table, "oneway": oneway, "shapley": sh,
            "commonality": cm, "total": g[full], "n_cols_full": table[full]["n_cols"]}
