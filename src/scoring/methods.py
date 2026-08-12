"""遺伝子セットスコアの算出手法。

手法の違いで結論が変わるかを見るのが目的なので、実装は 1 ファイルに揃えて
前処理（基準化の取り方）を明示的に切り替えられるようにする。

reference の意味
  within  : 与えられた行列の中だけで基準化する。個人の順位づけに使う
  pooled  : 呼び出し側が複数条件を連結した行列を渡すことで、条件間の差を残す

4 手法
  zmean     : 遺伝子ごとに z 化してセット内平均（最も素朴で解釈しやすい）
  singscore : サンプル内で全遺伝子を順位づけし、セット内の平均順位（サンプル内基準）
  meanrank  : singscore の中央値版（外れ値に強い変種）
  plage     : 遺伝子 z 行列の第 1 主成分（共変動する方向を取り出す）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("zmean", "singscore", "plage", "meanrank")


class ScoringContext:
    """行列を 1 度だけ前処理して、各手法で共有する。

    Z : 遺伝子ごとに z 化した行列（zmean / plage 用）
    P : サンプル内の百分位順位（singscore / meanrank 用）
    S : 遺伝子ごとに順位を取って標準化した行列（個人間の相関計算用）
    """

    def __init__(self, expr: pd.DataFrame):
        self.genes = list(expr.index)
        self.samples = list(expr.columns)
        self.index = {g: i for i, g in enumerate(self.genes)}
        x = expr.to_numpy(dtype=np.float64)

        self.Z = _standardize_rows(x)
        self.P = _percentile_rank_columns(x)
        self.S = _standardize_rows(_rank_rows(x))

    def idx(self, genes: list[str]) -> np.ndarray:
        return np.array([self.index[g] for g in genes if g in self.index], dtype=int)

    def present(self, genes: list[str]) -> list[str]:
        return [g for g in genes if g in self.index]


def _standardize_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=0, keepdims=True)
    sd[sd == 0] = np.nan
    return (x - mu) / sd


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """行（遺伝子）ごとに、列（個人）方向の順位を取る。"""
    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    n = x.shape[1]
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks


def _percentile_rank_columns(x: np.ndarray) -> np.ndarray:
    """列（サンプル）ごとに、全遺伝子の中での百分位順位を取る。"""
    order = np.argsort(x, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    m = x.shape[0]
    cols = np.arange(x.shape[1])[None, :]
    ranks[order, cols] = np.arange(1, m + 1, dtype=np.float64)[:, None]
    return ranks / m


def score_set(ctx: ScoringContext, genes: list[str], method: str) -> np.ndarray:
    """1 セットのスコアを個人ごとに返す。遺伝子が足りない場合は NaN を返す。"""
    idx = ctx.idx(genes)
    if idx.size < 2:
        return np.full(len(ctx.samples), np.nan)

    if method == "zmean":
        return np.nanmean(ctx.Z[idx], axis=0)
    if method == "singscore":
        return ctx.P[idx].mean(axis=0) - 0.5
    if method == "meanrank":
        return np.median(ctx.P[idx], axis=0) - 0.5
    if method == "plage":
        sub = ctx.Z[idx]
        sub = np.nan_to_num(sub, nan=0.0)
        # 第 1 右特異ベクトル。符号は zmean と正相関する向きに固定する
        _, _, vt = np.linalg.svd(sub, full_matrices=False)
        comp = vt[0]
        ref = sub.mean(axis=0)
        if np.dot(comp, ref) < 0:
            comp = -comp
        return comp * np.sqrt(len(comp))  # 分散を他手法と揃えるためのスケール
    raise ValueError(f"未知の手法: {method}")


def score_sets(
    ctx: ScoringContext, sets: dict[str, list[str]], method: str
) -> pd.DataFrame:
    """複数セットのスコアを sets x individuals で返す。"""
    rows = {name: score_set(ctx, genes, method) for name, genes in sets.items()}
    return pd.DataFrame(rows, index=ctx.samples).T
