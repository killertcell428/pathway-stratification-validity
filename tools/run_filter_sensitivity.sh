#!/usr/bin/env bash
# 主コホートの発現フィルタ分位を振って、headline 指標が保つかを確かめる。
#
# なぜ要るか
#   2.2 節は「発現フィルタは解析の第一段である」と主張し、3.8 節は前処理で対照の水準が
#   70 倍動くと示している。それなのに headline の 80.9% を出す側のフィルタ（第 50 分位）は
#   一度も振っていなかった。RNA-seq 側は CPM 3 水準を振ってあるので非対称だった。
#
# 安全性
#   build_matrices は expr_path / gene_mean_path 経由になったので、T26_MATRIX_SUFFIX を
#   立てた走行は正本（data/interim/expr_*.parquet）に触らない。
#   各水準の出力は expr_<cond>_p<分位>.parquet に入る。
#
# 出力
#   results/tables/gene_set_metrics_p40.csv など（run_evaluation の --suffix）
set -euo pipefail
cd "$(dirname "$0")/.."

for P in 40 60; do
  echo "=========================================================="
  echo "[発現フィルタ 第 ${P} 分位] 前処理"
  T26_MATRIX_SUFFIX="_p${P}" T26_EXPR_PERCENTILE="${P}" \
    pixi run python -m src.preprocessing.build_matrices
  echo "[発現フィルタ 第 ${P} 分位] モジュール導出"
  T26_MATRIX_SUFFIX="_p${P}" pixi run python -m src.preprocessing.derive_modules
  echo "[発現フィルタ 第 ${P} 分位] 評価"
  T26_MATRIX_SUFFIX="_p${P}" \
    pixi run python -m src.reliability.run_evaluation --suffix "_p${P}"
done
echo "=========================================================="
echo "完了。正本のハッシュを確認する:"
pixi run python - <<'PY'
import hashlib, json, pathlib
canon = json.loads(pathlib.Path('/tmp/canon.json').read_text())
bad = []
for name, want in canon.items():
    f = pathlib.Path('data/interim') / name
    got = hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else 'MISSING'
    print(f"  {'OK ' if got == want else 'NG '} {name:34s} {got}")
    if got != want:
        bad.append(name)
print('正本は無傷' if not bad else f'**正本が変わった: {bad}**')
PY
