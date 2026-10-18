test_that("Mantel-Haenszel reduces to the single-table risk ratio", {
  r <- mh_rr(12, 200, 5, 100)
  expect_equal(unname(r["logRR"]), log((12 / 200) / (5 / 100)))
  expect_equal(unname(r["se"])^2, 1 / 12 - 1 / 200 + 1 / 5 - 1 / 100)
})

test_that("Greenland-Robins variance by hand", {
  x1 <- c(3, 10, 0); n1 <- c(40, 120, 60); x0 <- c(1, 8, 2); n0 <- c(20, 60, 30); N <- n1 + n0
  R <- sum(x1 * n0 / N); S <- sum(x0 * n1 / N); P <- sum(n1 * n0 * (x1 + x0) / N^2 - x1 * x0 / N)
  r <- mh_rr(x1, n1, x0, n0)
  expect_equal(unname(r["logRR"]), log(R / S))
  expect_equal(unname(r["se"]), sqrt(P / (R * S)))
  expect_true(all(is.na(mh_rr(c(0, 0), c(10, 10), c(1, 2), c(10, 10)))))
})

test_that("inverse-variance and DerSimonian-Laird", {
  r <- iv_rr(0, 100, 3, 50)
  expect_equal(unname(r["logRR"]), log(0.5 / 101) - log(3.5 / 51))
  x1 <- c(3, 10, 0, 7); n1 <- c(40, 120, 60, 90); x0 <- c(1, 8, 2, 1); n0 <- c(20, 60, 30, 45)
  z <- (x1 == 0) | (x0 == 0); a <- x1 + 0.5 * z; b <- x0 + 0.5 * z; m1 <- n1 + z; m0 <- n0 + z
  y <- log(a / m1) - log(b / m0); v <- 1 / a - 1 / m1 + 1 / b - 1 / m0; w <- 1 / v
  fe <- sum(w * y) / sum(w)
  tau2 <- max(0, (sum(w * (y - fe)^2) - 3) / (sum(w) - sum(w^2) / sum(w))); ws <- 1 / (v + tau2)
  expect_equal(unname(iv_rr(x1, n1, x0, n0, method = "dl")["logRR"]), sum(ws * y) / sum(ws))
  expect_error(iv_rr(x1, n1, x0, n0, method = "reml"))
})
