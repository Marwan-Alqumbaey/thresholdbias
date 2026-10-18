# Repeats selected cells of the simulation study in R and compares bias and coverage
# of M1 (any-arm rule) and M7 (pooled-arm rule) with the Python results.
#   Rscript validation/simulation_spot_check.R
suppressPackageStartupMessages(library(thresholdbias))
root <- if (file.exists("results/simulation/naive_methods.csv")) "." else ".."
py <- read.csv(file.path(root, "results/simulation/naive_methods.csv"), stringsAsFactors = FALSE)
set.seed(20261001)
reps <- 1000; K <- 30; N <- 300; c5 <- 0.05

one_cell <- function(a, b, pmed, rr) {
  n1 <- round(N * a / (a + b)); n0 <- N - n1
  res <- matrix(NA_real_, reps, 4, dimnames = list(NULL, c("M1", "M1cov", "M7", "M7cov")))
  for (r in seq_len(reps)) {
    p0 <- plogis(qlogis(pmed) + 0.5 * rnorm(K)); p1 <- pmin(rr * p0, 0.99)
    x1 <- rbinom(K, n1, p1); x0 <- rbinom(K, n0, p0)
    for (m in c("M1", "M7")) {
      keep <- if (m == "M1") posted_anyarm(x1, n1, x0, n0, c5) else posted_pooled(x1, n1, x0, n0, c5)
      e <- mh_rr(x1[keep], rep(n1, sum(keep)), x0[keep], rep(n0, sum(keep)))
      res[r, m] <- e[["logRR"]] - log(rr)
      res[r, paste0(m, "cov")] <- abs(e[["logRR"]] - log(rr)) <= 1.96 * e[["se"]]
    }
  }
  res
}

ok <- TRUE
for (cell in list(c(3, 1, 0.02, 1), c(2, 1, 0.03, 1), c(1, 1, 0.02, 1.5))) {
  res <- one_cell(cell[1], cell[2], cell[3], cell[4])
  for (m in c("M1", "M7")) {
    ref <- py[py$K == K & py$N == N & py$c == c5 & py$ratio == paste0(cell[1], ":", cell[2]) &
                abs(py$pmed - cell[3]) < 1e-9 & abs(py$rr - cell[4]) < 1e-9 & py$method == m, ]
    b <- res[, m]; b <- b[is.finite(b)]
    cv <- mean(res[is.finite(res[, m]), paste0(m, "cov")])
    z_bias <- (mean(b) - ref$bias) / sqrt(sd(b)^2 / length(b) + ref$mcse_bias^2)
    z_cov <- (cv - ref$cover) / sqrt(cv * (1 - cv) / length(b) + ref$mcse_cover^2)
    cat(sprintf("%g:%g, placebo %g%%, RR %g, %s: bias R %.3f, Python %.3f; coverage R %.3f, Python %.3f\n",
                cell[1], cell[2], 100 * cell[3], cell[4], m, mean(b), ref$bias, cv, ref$cover))
    ok <- ok && abs(z_bias) < 4 && abs(z_cov) < 4
  }
}
stopifnot("R and Python simulation results differ by more than Monte Carlo error" = ok)
cat("R and Python simulation results agree within Monte Carlo error.\n")
