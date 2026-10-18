test_that("exact conditional moments match the brute-force double sum", {
  set.seed(1)
  for (i in 1:40) {
    n1 <- sample(5:300, 1); n0 <- sample(5:300, 1)
    p0 <- exp(runif(1, log(0.002), log(0.2))); p1 <- min(p0 * exp(rnorm(1, 0, 0.5)), 0.6)
    cc <- sample(c(0, 0.02, 0.05, 0.1), 1); rule <- sample(c("strict", "weak"), 1)
    b <- brute(n1, n0, p1, p0, cc, rule)
    r <- exact_conditional(n1, n0, p1, p0, cc, rule)
    expect_equal(r$p_post, b$P, tolerance = 1e-9)
    expect_equal(r$e1, b$e1, tolerance = 1e-8)
    expect_equal(r$e0, b$e0, tolerance = 1e-8)
    expect_equal(r$rr_cond, b$mh, tolerance = 1e-8)
    k <- r$kappa
    D1 <- if (k > 0) (r$e1 - p1) / k else 0
    D0 <- if (k > 0) (r$e0 - p0) / k else 0
    expect_gte(D1, -1e-15); expect_gte(D0, -1e-15)
    expect_equal(r$rr_cond / r$rr_true, (1 + k * D1 / p1) / (1 + k * D0 / p0), tolerance = 1e-10)
  }
})

test_that("key values of the theory", {
  expect_equal(exact_conditional(200, 100, 0.03, 0.03, 0.05)$rr_cond, 0.7293, tolerance = 1e-4)
  expect_equal(exact_conditional(200, 100, 0.03, 0.03, 0.05)$p_post, 0.1177, tolerance = 1e-3)
  expect_equal(exact_conditional(100, 100, 0.02, 0.02, 0.05)$rr_cond, 1, tolerance = 1e-12)
  expect_equal(exact_conditional(100, 99, 0.02, 0.02, 0.05)$rr_cond, 0.65, tolerance = 1e-3)
  expect_equal(exact_conditional(150, 150, 0.045, 0.03, 0.05)$rr_cond, 1.7656, tolerance = 1e-4)
})

test_that("strict rule at c = 0 leaves the ratio unchanged", {
  for (d in list(c(200, 100, 0.01, 0.02), c(300, 100, 0.03, 0.01), c(57, 13, 0.1, 0.05))) {
    r <- exact_conditional(d[1], d[2], d[3], d[4], 0)
    expect_equal(r$rr_cond, d[3] / d[4], tolerance = 1e-10)
  }
})

test_that("very rare posting is computed stably (large-deviation limits)", {
  r <- exact_conditional(80000, 40000, 0.03, 0.03, 0.05)
  expect_lt(r$p_post, 1e-50)
  expect_equal(r$rr_cond, 0.6, tolerance = 0.01)
  expect_equal(exact_conditional(90000, 30000, 0.045, 0.03, 0.05)$rr_cond, 0.05 / 0.03, tolerance = 0.01)
})

test_that("pooled-arm rule: Mantel-Haenszel limit is exactly 1 when RR = 1", {
  set.seed(3)
  for (i in 1:12) {
    n1 <- sample(5:150, 1); n0 <- sample(5:150, 1); p <- exp(runif(1, log(0.005), log(0.2)))
    b <- brute(n1, n0, p, p, sample(c(0.02, 0.05, 0.1), 1), pooled = TRUE)
    if (b$P > 1e-12) expect_equal(b$mh, 1, tolerance = 1e-9)
  }
})

test_that("bias_check reduces to exact_conditional for one design", {
  for (d in list(c(200, 100, 0.03, 1), c(150, 150, 0.03, 1.5), c(40, 120, 0.01, 0.67))) {
    r <- bias_check(d[1], d[2], d[3], rr = d[4])
    e <- exact_conditional(d[1], d[2], d[4] * d[3], d[3], 0.05)
    expect_equal(r$rr_posted, e$rr_cond, tolerance = 1e-12)
    expect_equal(r$rr_complete, d[4], tolerance = 1e-12)
    expect_equal(r$p_post, e$p_post, tolerance = 1e-12)
  }
})

test_that("bias_check matches the double sum over a mix of designs", {
  n1 <- c(200, 50, 120, 33); n0 <- c(100, 50, 40, 61)
  for (rule in c("strict", "weak")) for (s in list(c(0.02, 1), c(0.03, 1.5), c(0.05, 0.67))) {
    num <- den <- 0
    for (i in seq_along(n1)) {
      b <- brute(n1[i], n0[i], s[2] * s[1], s[1], 0.05, rule)
      w <- n1[i] * n0[i] / (n1[i] + n0[i])
      num <- num + w * b$P * b$e1
      den <- den + w * b$P * b$e0
    }
    expect_equal(bias_check(n1, n0, s[1], rr = s[2], rule = rule)$rr_posted, num / den, tolerance = 1e-9)
  }
})

test_that("bias_check: equal arms give no distortion, rare posting stays finite", {
  expect_equal(bias_check(c(100, 250, 400), c(100, 250, 400), c(0.005, 0.02, 0.04))$distortion,
               rep(1, 3), tolerance = 1e-12)
  big <- bias_check(c(20000, 300), c(10000, 150), c(0.001, 0.01))
  expect_true(all(is.finite(big$rr_posted)) && all(big$rr_posted > 0))
  expect_equal(bias_check(100, 50, 0.03, rr = 1.5, tau = 0.3)$rr_complete, 1.5 * exp(0.3^2 / 2),
               tolerance = 1e-8)
  expect_error(bias_check(c(10, 20), 10, 0.02))
  expect_error(bias_check(10, 0, 0.02))
})
