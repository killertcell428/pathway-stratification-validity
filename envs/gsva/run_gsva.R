# GSVA を実物の Bioconductor 実装で走らせる。純粋な関数として扱う。
#
# 入力  expr.tsv  遺伝子 x 検体の発現行列（1 列目が遺伝子名）
#       sets.gmt  評価するセット（GMT 形式）
# 出力  scores.tsv セット x 検体のスコア行列
#
# セットの抽出（どの 427 セットを使うか）は Python 側に一本化してある。
# ここで抽出をやり直すと ssGSEA と別のセットになり、比較の意味がなくなる。
#
# kcdf は "Gaussian"。主コホートは Illumina BeadChip の対数スケールで連続なので
# Gaussian が正しい（カウントなら "Poisson"）。ここを間違えると GSVA の
# カーネル推定が壊れるので、入力が対数スケールであることを前提に固定する。

suppressPackageStartupMessages({
  library(GSVA)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("使い方: Rscript run_gsva.R <expr.tsv> <sets.gmt> <out.tsv>")
}
expr_path <- args[1]; gmt_path <- args[2]; out_path <- args[3]

cat("GSVA", as.character(packageVersion("GSVA")), "/ R", R.version.string, "\n")

expr <- read.delim(expr_path, row.names = 1, check.names = FALSE)
mat <- as.matrix(expr)
cat("発現行列:", nrow(mat), "遺伝子 x", ncol(mat), "検体\n")

# GMT を読む。1 列目がセット名、2 列目が説明、3 列目以降が遺伝子。
lines <- readLines(gmt_path)
sets <- list()
for (ln in lines) {
  f <- strsplit(ln, "\t", fixed = TRUE)[[1]]
  if (length(f) < 3) next
  sets[[f[1]]] <- unique(f[3:length(f)])
}
cat("遺伝子セット:", length(sets), "件\n")

# GSVA 2.x の API。gsvaParam でパラメータを固めてから gsva を呼ぶ。
par <- gsvaParam(
  exprData = mat,
  geneSets = sets,
  kcdf = "Gaussian",   # 対数スケールの連続値
  minSize = 2,
  maxSize = 1e6
)
res <- gsva(par, verbose = FALSE)
cat("スコア行列:", nrow(res), "セット x", ncol(res), "検体\n")

out <- data.frame(set = rownames(res), res, check.names = FALSE)
write.table(out, out_path, sep = "\t", quote = FALSE, row.names = FALSE)
cat("書き出し:", out_path, "\n")
