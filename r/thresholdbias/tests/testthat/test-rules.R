test_that("kcross matches exact rational comparison", {
  for (n in c(1:300, 999, 1000, 1001)) {
    for (cc in c(0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 1)) {
      # x / n > c  <=>  100 x > 100 c n, done in integers for c on a 1/100 grid
      ref_s <- which((100 * (0:n)) > round(100 * cc) * n)[1] - 1
      ref_w <- which((100 * (0:n)) >= round(100 * cc) * n)[1] - 1
      expect_equal(kcross(n, cc, "strict"), if (is.na(ref_s)) n + 1L else as.integer(ref_s))
      expect_equal(kcross(n, cc, "weak"), if (is.na(ref_w)) n + 1L else as.integer(ref_w))
    }
  }
})

test_that("kcross edge cases", {
  expect_equal(kcross(100, 0.05), 6L)
  expect_equal(kcross(100, 0.05, "weak"), 5L)
  expect_equal(kcross(99, 0.05), 5L)
  expect_equal(kcross(100, 0), 1L)
  expect_equal(kcross(100, 0, "weak"), 0L)
  expect_equal(kcross(10, 1), 11L)
  expect_equal(kcross(c(20, 99, 100), 0.05), c(2L, 5L, 6L))
  expect_error(kcross(100, 5))
  expect_error(kcross(100, 0.05, "Strict"))
})

test_that("posting rules", {
  expect_true(posted_anyarm(6, 100, 0, 50, 0.05))
  expect_false(posted_anyarm(5, 100, 2, 50, 0.05))
  expect_true(posted_anyarm(0, 100, 3, 50, 0.05))
  expect_true(posted_pooled(5, 100, 3, 50, 0.05))
  expect_false(posted_pooled(5, 100, 2, 50, 0.05))
})
