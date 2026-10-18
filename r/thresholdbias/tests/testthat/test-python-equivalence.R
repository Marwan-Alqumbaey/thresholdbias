# Reference values written by validation/make_reference.py with the Python package.
skip_if_not_installed("jsonlite")
ref <- jsonlite::fromJSON(test_path("fixtures", "reference_python.json"), simplifyVector = FALSE)

test_that("kcross is identical to Python", {
  for (e in ref$kcross) expect_identical(kcross(e$n, e$c, e$rule), as.integer(e$k))
})

test_that("exact conditional moments agree with Python", {
  for (e in ref$exact) {
    r <- exact_conditional(e$n1, e$n0, e$p1, e$p0, e$c, e$rule)
    expect_equal(r$p_post, e$p_post, tolerance = 1e-12)
    expect_equal(r$e1, e$e1, tolerance = 1e-12)
    expect_equal(r$e0, e$e0, tolerance = 1e-12)
    expect_equal(r$rr_cond, e$rr_cond, tolerance = 1e-12)
  }
})

test_that("pooled estimators agree with Python", {
  for (e in ref$pool) {
    args <- lapply(e[c("x1", "n1", "x0", "n0")], unlist)
    expect_equal(unname(do.call(mh_rr, args)), unlist(e$mh), tolerance = 1e-12)
    expect_equal(unname(do.call(iv_rr, c(args, method = "fixed"))), unlist(e$fixed), tolerance = 1e-12)
    expect_equal(unname(do.call(iv_rr, c(args, method = "dl"))), unlist(e$dl), tolerance = 1e-12)
  }
})

test_that("log-likelihood values agree with Python", {
  for (e in ref$fit) {
    D <- .prep(unlist(e$x1), unlist(e$n1), unlist(e$x0), unlist(e$n0), 0.05, "strict")
    U <- .prep_unposted(list(unlist(e$un1), unlist(e$un0), 0.05), "strict")
    for (v in e$negll) {
      par <- unlist(v$par)
      expect_lt(abs(negll(par, D) - v$conditional), 1e-8)
      expect_lt(abs(negll(par, D, U) - v$full), 1e-8)
    }
  }
})

test_that("selection-corrected fits agree with Python", {
  skip_on_cran()
  for (e in ref$fit) {
    x1 <- unlist(e$x1); n1 <- unlist(e$n1); x0 <- unlist(e$x0); n0 <- unlist(e$n0)
    r <- fit_corrected(x1, n1, x0, n0, 0.05)
    py <- unlist(e$conditional)
    expect_lt(max(abs(c(r$logRR, r$lo, r$hi) - py[1:3])), 1e-5)
    g <- fit_corrected(x1, n1, x0, n0, 0.05, unposted = list(unlist(e$un1), unlist(e$un0), 0.05), ci = "none")
    expect_lt(abs(g$logRR - e$full), 1e-5)
  }
})

test_that("bias_check agrees with Python", {
  for (e in ref$bias_check) {
    r <- bias_check(unlist(e$n1), unlist(e$n0), unlist(e$p0), rr = e$rr, c = e$c, rule = e$rule,
                    sd_logit = e$sd_logit, tau = e$tau, nodes = 15)
    for (k in c("rr_complete", "rr_posted", "distortion", "p_post"))
      expect_equal(r[[k]], unlist(e[[k]]), tolerance = 1e-10)
  }
})
