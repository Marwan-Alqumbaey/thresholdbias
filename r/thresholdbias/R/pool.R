#' Mantel-Haenszel pooled risk ratio
#'
#' Mantel-Haenszel log risk ratio with the Greenland-Robins standard error.
#'
#' @inheritParams posted_anyarm
#' @return Named numeric vector \code{c(logRR, se)}; both \code{NA} when either
#'   arm has no events in total.
#' @examples
#' mh_rr(c(3, 10), c(40, 120), c(1, 8), c(20, 60))
#' @export
mh_rr <- function(x1, n1, x0, n0) {
  N <- n1 + n0
  R <- sum(x1 * n0 / N)
  S <- sum(x0 * n1 / N)
  if (R <= 0 || S <= 0) return(c(logRR = NA_real_, se = NA_real_))
  P <- sum((n1 * n0 * (x1 + x0) - x1 * x0 * N) / N^2)
  c(logRR = log(R / S), se = sqrt(P / (R * S)))
}

#' Inverse-variance pooled risk ratio
#'
#' Fixed-effect or DerSimonian-Laird random-effects pooling of log risk ratios.
#' A continuity correction \code{cc} is added to every cell of tables with a
#' zero count.
#'
#' @inheritParams posted_anyarm
#' @param cc Continuity correction.
#' @param method \code{"fixed"} or \code{"dl"}.
#' @return Named numeric vector \code{c(logRR, se)}.
#' @export
iv_rr <- function(x1, n1, x0, n0, cc = 0.5, method = c("fixed", "dl")) {
  method <- match.arg(method)
  z <- (x1 == 0) | (x0 == 0)
  a <- x1 + cc * z
  b <- x0 + cc * z
  m1 <- n1 + 2 * cc * z
  m0 <- n0 + 2 * cc * z
  y <- log(a / m1) - log(b / m0)
  v <- 1 / a - 1 / m1 + 1 / b - 1 / m0
  w <- 1 / v
  est <- sum(w * y) / sum(w)
  if (method == "fixed" || length(y) < 2) return(c(logRR = est, se = 1 / sqrt(sum(w))))
  Q <- sum(w * (y - est)^2)
  C <- sum(w) - sum(w^2) / sum(w)
  tau2 <- max(0, (Q - (length(y) - 1)) / C)
  ws <- 1 / (v + tau2)
  c(logRR = sum(ws * y) / sum(ws), se = 1 / sqrt(sum(ws)))
}
