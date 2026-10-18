test_that("weak rule at c = 0 gives the ordinary random-effects binomial MLE", {
  set.seed(42)
  K <- 60; p0 <- exp(log(0.04) + 0.5 * rnorm(K))
  x1 <- rbinom(K, 150, pmin(1.3 * p0, 0.999)); x0 <- rbinom(K, 100, p0)
  gh <- thresholdbias:::.gauss_hermite(80); z <- sqrt(2) * gh$x; wz <- gh$w / sqrt(pi)
  nll <- function(p) {
    q0 <- pmin(exp(p[2] + exp(p[3]) * z), 0.999); q1 <- pmin(exp(p[1]) * q0, 0.999)
    L <- sapply(seq_len(K), function(i) sum(dbinom(x1[i], 150, q1) * dbinom(x0[i], 100, q0) * wz))
    -sum(log(L))
  }
  ref <- optim(c(0, log(0.04), log(0.5)), nll, control = list(reltol = 1e-12, maxit = 5000))$par
  r <- fit_corrected(x1, 150, x0, 100, 0, rule = "weak", ci = "none")
  expect_equal(r$logRR, ref[1], tolerance = 1e-3)
})

test_that("corrected estimate recovers the truth under 2:1 selection", {
  set.seed(2026)
  K <- 4000; p0 <- exp(log(0.03) + 0.4 * rnorm(K))
  x1 <- rbinom(K, 200, p0); x0 <- rbinom(K, 100, p0)
  post <- which(posted_anyarm(x1, 200, x0, 100, 0.05))[1:600]
  r <- fit_corrected(x1[post], 200, x0[post], 100, 0.05)
  expect_lt(abs(r$logRR), 0.08)
  expect_true(r$lo < 0 && 0 < r$hi)
  expect_identical(r$n_posted, 600L)
  expect_lt(mh_rr(x1[post], rep(200, 600), x0[post], rep(100, 600))["logRR"], -0.08)
})

test_that("threshold given as a percentage is rejected", {
  expect_error(fit_corrected(6, 100, 3, 50, 5))
})
