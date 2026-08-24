#!/usr/bin/env bash
# 正規化の選択を headline 指標まで通す（GSE81046、RNA-seq）。
#
# なぜ要るか
#   3.8 節は正規化でランダム対照の水準が 0.002 から 0.168 まで 70 倍動くと示している。
#   ところがそれを測ったのは主成分の構造と対照の水準までで、**合格率と乖離の割合まで
#   通していない**。しかも quantification: tpm と cross_sample_normalization: quantile は
#   コードに実装済みで、設定ファイルに「感度分析用」と書いてある。
#   つまみは作ってあって回していない状態だった。
#
#   3.8 節は自分を守る予測も持っている。表 4 の対照の 5-95% 幅は分位 0.050 /
#   TMM 0.055 / log2(TPM+1) 0.178 で、擁護可能な 2 通りの間では散らばりが 10% しか違わない。
#   超過 z の分母がほぼ動かないなら合格率も大きく動かない、という予測が立つ。それを実測する。
#
# 安全性
#   名前空間（T26_DATASET）は gse81046 のまま動かさない。動かすと expr_full.parquet と
#   samples.csv の場所が変わって --reuse-full が使えなくなる。
#   定量と正規化は T26_QUANTIFICATION / T26_CROSS_SAMPLE_NORM で差し替え、
#   出力は T26_MATRIX_SUFFIX で分ける。接尾辞つき走行では donor_split.json を書き換えない。
#
# 出力
#   results/tables/gse81046/gene_set_metrics_tpm.csv, _quantile.csv
set -euo pipefail
cd "$(dirname "$0")/.."

export T26_DATASET=gse81046

# 第 4 引数で --reuse-full を渡すかを切り替える。
# **定量を変える腕では使えない。** expr_full.parquet は定量済みの行列であり、
# _finish の中で quant は表示ラベルにしか使われない。再利用すると
# 「定量を変えたつもりで前の値を使う」ことになる（実際にそれで 1 本が無効になった）。
# build_rnaseq_matrix 側にも定量の不一致ガードを入れたので、誤ると例外で止まる。
run_variant () {
  local tag="$1" quant="$2" norm="$3" reuse="${4:-}"
  echo "=========================================================="
  echo "[${tag}] quantification=${quant} / cross_sample_normalization=${norm} ${reuse}"
  T26_MATRIX_SUFFIX="_${tag}" T26_QUANTIFICATION="${quant}" T26_CROSS_SAMPLE_NORM="${norm}" \
    pixi run python -m src.preprocessing.build_rnaseq_matrix ${reuse}
  T26_MATRIX_SUFFIX="_${tag}" \
    pixi run python -m src.preprocessing.derive_modules
  T26_MATRIX_SUFFIX="_${tag}" \
    pixi run python -m src.reliability.run_evaluation --suffix "_${tag}"
}

# 分位正規化。TMM と並んで擁護可能な腕であり、ここが動くかどうかが本番。
# 定量は採用値と同じ tmm_logcpm なので expr_full を再利用できる。
run_variant quantile tmm_logcpm quantile --reuse-full
# log2(TPM+1) をそのまま使う腕。3.8 節が「第 1 主成分が平均発現量と ρ = 0.977 で
# ほぼ一対一」という独立の理由で棄却している腕なので、ここが動いても結論は揺れない。
# **定量が変わるので tar から作り直す（--reuse-full を渡さない）。**
run_variant tpm tpm none
echo "=========================================================="
echo "完了"
