# Reproduces the registry experiment (point estimates) in R from the raw extract, using the
# R package, and compares every term with the Python results.
#   Rscript validation/reproduce_registry.R
suppressPackageStartupMessages(library(thresholdbias))
root <- if (file.exists("data/raw/trials.tsv.gz")) "." else ".."
rd <- function(f) read.delim(gzfile(f), colClasses = "character", na.strings = character(0), quote = "")

trials <- rd(file.path(root, "data/raw/trials.tsv.gz"))
rows <- do.call(rbind, lapply(sort(Sys.glob(file.path(root, "data/raw/rows_*.tsv.gz"))), rd))
rows$term <- trimws(gsub("\"", "", rows$term, fixed = TRUE))
num <- function(x) suppressWarnings(as.numeric(x))
trials$thr <- num(trials$threshold)
trials$ratio <- num(trials$drug_o_risk) / num(trials$pl_o_risk)
for (v in c("xd", "nd", "xp", "np")) rows[[v]] <- num(rows[[v]])
bad <- with(rows, is.na(xd) | is.na(nd) | is.na(xp) | is.na(np) | xd > nd | xp > np | nd <= 0 | np <= 0 | xd < 0 | xp < 0)
rows <- rows[!bad, ]
rows$order <- seq_len(nrow(rows))
rows <- rows[order(-(rows$xd + rows$xp), rows$order), ]
rows <- rows[!duplicated(rows[c("nct", "type", "term")]), ]

t0 <- trials[!is.na(trials$thr) & trials$thr == 0 & trials$model != "CROSSOVER" & is.finite(trials$ratio) & trials$ratio > 0, ]
t0$stratum <- ifelse(t0$ratio < 0.9, "<0.9", ifelse(t0$ratio <= 1.1, "0.9-1.1",
                     ifelse(t0$ratio < 1.7, "1.1-1.7", ">=1.7")))
o <- merge(rows[rows$type == "O" & rows$xd + rows$xp > 0, ], t0[c("nct", "stratum")], by = "nct")
o$post <- posted_anyarm(o$xd, o$nd, o$xp, o$np, 0.05)
o$pool <- posted_pooled(o$xd, o$nd, o$xp, o$np, 0.05)
N <- o$nd + o$np
o$R <- o$xd * o$np / N
o$S <- o$xp * o$nd / N

lr <- function(R, S) ifelse(R > 0 & S > 0, log(R / S), NA_real_)
res <- list()
for (s in c("<0.9", "0.9-1.1", "1.1-1.7", ">=1.7")) {
  g <- o[o$stratum == s, ]
  cnt <- tapply(g$nct, g$term, function(v) length(unique(v)))
  terms <- sort(names(cnt)[cnt >= 10])
  g <- g[g$term %in% terms, ]
  agg <- function(m) {
    R <- tapply(g$R * m, g$term, sum)[terms]; S <- tapply(g$S * m, g$term, sum)[terms]
    lr(R, S)
  }
  res[[s]] <- data.frame(stratum = s, term = terms, lr_full = agg(1), lr_post = agg(g$post), lr_pool = agg(g$pool))
}
r <- do.call(rbind, res)
r$shift_post <- r$lr_post - r$lr_full
r$shift_pool <- r$lr_pool - r$lr_full

py <- read.csv(file.path(root, "results/registry/rethreshold_primary_terms.csv"), stringsAsFactors = FALSE, na.strings = "")
m <- merge(r, py, by = c("stratum", "term"), suffixes = c("_r", "_py"))
stopifnot("term sets differ" = nrow(m) == nrow(r) && nrow(m) == nrow(py))
d <- function(a, b) max(abs(a - b), na.rm = TRUE)
cat(sprintf("Terms compared: %d\n", nrow(m)))
cat(sprintf("Max |difference| in log RR: all rows %.2e, posted %.2e, pooled %.2e\n",
            d(m$lr_full_r, m$lr_full_py), d(m$lr_post_r, m$lr_post_py), d(m$lr_pool_r, m$lr_pool_py)))
stopifnot(identical(is.na(m$lr_post_r), is.na(m$lr_post_py)))
out <- t(sapply(split(m, m$stratum), function(x)
  c(terms = nrow(x), median_shift_R = median(x$shift_post_r, na.rm = TRUE),
    median_shift_Python = median(x$shift_post_py, na.rm = TRUE),
    median_pooled_R = median(x$shift_pool_r, na.rm = TRUE), median_pooled_Python = median(x$shift_pool_py, na.rm = TRUE))))
print(round(out[c("<0.9", "0.9-1.1", "1.1-1.7", ">=1.7"), ], 6))
stopifnot(d(m$lr_full_r, m$lr_full_py) < 1e-10, d(m$lr_post_r, m$lr_post_py) < 1e-10, d(m$lr_pool_r, m$lr_pool_py) < 1e-10)
cat("R and Python results agree (largest difference below 1e-10).\n")
