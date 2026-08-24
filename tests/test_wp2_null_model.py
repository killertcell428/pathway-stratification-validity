"""WP2 の帰無モデルを合成データで検査する。実データに当てる前にここを通す。

確かめること
  1. 標準化の手続きが観測側と帰無側で同一である（散らばりの非対称性が出ない）
  2. シグネチャ固有の関連があるとき、それを検出できる
  3. 対照にも同じ関連が出るとき、棄却される（項目 h の存在理由）
  4. 真に帰無のとき、偽陽性率が名目水準を守る
  5. 3 条件の判定が「どれか 1 つ欠ければ null」になっている
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reliability.wp2_null_model import (
    empirical_p,
    excess_z,
    exchangeability_null,
    range_matched_control_null,
    observed_statistic,
    pooled_correlation,
    slot_statistic,
    verdict,
)

N_SLOTS = 40          # シグネチャの枠（WP2 の n）
N_CTRL = 20           # 1 枠あたりの対照


def make_world(gen, sig_effect: float = 0.0, shared_effect: float = 0.0,
               sig_extra_sd: float = 1.0, n_ctrl: int = N_CTRL):
    """枠 40 個を作る。各枠は「シグネチャ 1 個 + 対照 n_ctrl 個」の整合性と目的変数。

    設計上の要点を 2 つ埋め込んである。

    1. **効果は枠の水準ではなく枠内の超過に載せる。** 説明変数は「自分以外の集合を
       基準にした z」なので、枠の水準（サイズ・発現量分位）は引き算で消える。
       効果を水準に載せるとどの z にも現れず、検出できない（実際にそう挙動した）。
    2. **シグネチャは対照より散らばりを広く取れる**（sig_extra_sd）。実データでも
       シグネチャの z は 12 に達するのに対し、対照は概ね標準正規に収まる。
       帰無側が同じ標準化手続きを踏まないと、この差だけで偽陽性が出る。

    sig_effect    シグネチャだけに現れる関連。H1 が真の世界
    shared_effect 対照にも同じだけ現れる関連。共分散構造の性質であって
                  シグネチャの性質ではない世界（項目 h が捕まえるべき状況）
    """
    coh_slots, y_slots = [], []
    for _ in range(N_SLOTS):
        level = gen.normal()                    # 枠に固有の水準。z からは消える
        ctrl = level + gen.normal(scale=0.5, size=n_ctrl)
        sig = level + gen.normal(scale=0.5 * sig_extra_sd) + gen.normal(scale=sig_extra_sd)
        coh = np.concatenate([[sig], ctrl])     # index 0 がシグネチャ

        # 目的変数は「その集合の枠内での超過」に連動させる。
        # 超過は excess_z と同じ定義（自分以外を基準）で計算する。
        y = np.empty(len(coh))
        for j in range(len(coh)):
            z = excess_z(coh, j)
            eff = shared_effect + (sig_effect if j == 0 else 0.0)
            y[j] = eff * (0.0 if not np.isfinite(z) else z) + gen.normal()
        coh_slots.append(coh)
        y_slots.append(y)
    return coh_slots, y_slots


def test_excess_z_は自分を基準に含めない():
    gen = np.random.default_rng(0)
    coh = np.concatenate([[8.0], gen.normal(scale=1.0, size=19)])
    z = excess_z(coh, 0)
    naive = (coh[0] - coh.mean()) / coh.std(ddof=1)
    assert z > 6, f"自分を除けば大きく出るべき: {z:.2f}"
    assert z > naive * 1.5, f"loo={z:.2f} naive={naive:.2f}"


def test_excess_z_は縮退でnan():
    assert np.isnan(excess_z(np.array([5.0] + [0.0] * 19), 0))   # 基準側の SD が 0
    assert np.isnan(excess_z(np.array([1.0, 2.0, 3.0]), 0))      # 基準側が 3 個未満


def test_観測値は帰無分布と同じ構成法から出ている():
    """全枠 index 0 を選んだ場合が観測値であることを、値の一致で確かめる。"""
    gen = np.random.default_rng(5)
    coh, y = make_world(gen)
    assert observed_statistic(coh, y) == pytest.approx(
        slot_statistic(coh, y, [0] * N_SLOTS))


def test_散らばりの非対称性だけでは偽陽性にならない():
    """シグネチャの z の幅が対照の 3 倍あっても、関連が無ければ棄却されない。

    これが最初の修正案（対照だけを leave-one-out で標準化して比べる案）で出た偏りである。
    シグネチャ固有の効果ゼロの合成データで経験的 p = 0.002 が出たため、
    観測側と帰無側の標準化手続きを揃える交換可能性検定に組み替えた。
    """
    gen = np.random.default_rng(31)
    coh, y = make_world(gen, sig_extra_sd=3.0)
    rho = observed_statistic(coh, y)
    null = exchangeability_null(coh, y, n_replicates=600, seed=2)
    p = empirical_p(rho, null)
    assert p >= 0.05, f"関連が無いのに棄却された: rho={rho:.3f} p={p:.4f}"


def test_シグネチャ固有の関連は検出できる():
    """H1 が真の世界。シグネチャにだけ関連があり、対照には無い。"""
    gen = np.random.default_rng(11)
    coh, y = make_world(gen, sig_effect=2.0)
    rho = observed_statistic(coh, y)
    null = exchangeability_null(coh, y, n_replicates=600, seed=3)
    assert abs(rho) > 0.3, f"rho={rho:.3f}"
    assert empirical_p(rho, null) < 0.05
    assert verdict(rho, (rho - 0.2, rho + 0.2), null)["supported"]


def test_交換可能性検定の限界_共有効果は捕まえられない():
    """**既知の限界を固定する検査。** 直せなかったことを記録している。

    同じ関係式 y = f(z) がシグネチャと対照の両方に成り立っていて、
    シグネチャの z の幅だけが広い世界を作ると、交換可能性検定は「シグネチャ固有」と
    誤判定する。Spearman が信号対雑音比で決まり、幅の広い側が高く出るためである。

    区間をそろえた対照の分布と比べても解けない。シグネチャの z の範囲は
    対照の範囲をほぼ包含するので、区間で切っても散らばりの差が残る。
    **区間の一致は散らばりの一致ではない。**

    20 個の対照からはこれ以上詰められないので、事前登録では限界として明記し、
    シグネチャと対照の z の分布を並べて報告する規則にした。
    この検査は、その限界が黙って変わらないように固定してある。
    """
    gen = np.random.default_rng(21)
    coh, y = make_world(gen, shared_effect=2.0)
    rho = observed_statistic(coh, y)
    assert abs(rho) > 0.4, f"主要相関は強く出る: {rho:.3f}"

    null = exchangeability_null(coh, y, n_replicates=600, seed=4)
    assert empirical_p(rho, null) < 0.05, "限界が消えたなら、この検査を更新する"

    rm, used = range_matched_control_null(coh, y, n_replicates=600, seed=5)
    assert used >= 10, f"区間内に対照が残った枠: {used}"
    assert empirical_p(rho, rm) < 0.05, "範囲そろえでも解けない（これが限界の内容）"

    # 併記はされるが判定には入らない
    v = verdict(rho, (rho - 0.2, rho + 0.2), null, range_matched_null=rm)
    assert v["empirical_p_range_matched"] is not None
    assert "範囲そろえ対照に対する p < 0.05" not in v["conditions"]


def test_シグネチャ固有なら3条件を通る():
    """H1 が真の世界。3 条件すべてを満たす。"""
    gen = np.random.default_rng(41)
    coh, y = make_world(gen, sig_effect=2.5)
    rho = observed_statistic(coh, y)
    null = exchangeability_null(coh, y, n_replicates=600, seed=6)
    v = verdict(rho, (rho - 0.2, rho + 0.2), null)
    assert v["supported"], v["conditions"]


def test_真に帰無なら偽陽性率が名目を守る():
    false_pos, trials = 0, 40
    for s in range(trials):
        gen = np.random.default_rng(100 + s)
        coh, y = make_world(gen)
        rho = observed_statistic(coh, y)
        null = exchangeability_null(coh, y, n_replicates=300, seed=s)
        if empirical_p(rho, null) < 0.05:
            false_pos += 1
    rate = false_pos / trials
    assert rate <= 0.15, f"偽陽性率が高すぎる: {rate:.0%}（名目 5%）"


def test_帰無分布の幅はn40相当になる():
    """入れ子をまとめた 840 点の相関は 1 つの値しか出ないので幅を作れない。

    事前登録の初期草案はその 1 点をシグネチャ側の CI と比べる規則だった。
    """
    gen = np.random.default_rng(7)
    coh, y = make_world(gen)
    null = exchangeability_null(coh, y, n_replicates=400, seed=1)
    spread = np.percentile(null, 97.5) - np.percentile(null, 2.5)
    assert 0.2 < spread < 0.9, f"n=40 の帰無分布の幅: {spread:.3f}"
    assert np.isfinite(pooled_correlation(coh, y))


def test_3条件のどれか1つ欠ければnull扱い():
    null = np.random.default_rng(0).normal(scale=0.16, size=2000)   # n=40 相当の帰無
    assert verdict(0.55, (0.30, 0.75), null)["supported"]
    v = verdict(0.28, (0.05, 0.50), null)
    assert not v["supported"] and v["conditions"]["|rho| >= 0.3"] is False
    v = verdict(0.55, (-0.05, 0.90), null)
    assert not v["supported"] and v["conditions"]["CI が 0 を除く"] is False
    assert not verdict(0.10, (0.02, 0.18), null)["supported"]


def test_経験的pは0にならない():
    assert empirical_p(99.0, np.zeros(100)) == pytest.approx(1 / 101)
