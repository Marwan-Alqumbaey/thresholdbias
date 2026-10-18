.eps <- 1e-12

#' Smallest event count that crosses a threshold
#'
#' @param n Number(s) of participants at risk.
#' @param c Threshold as a proportion in [0, 1].
#' @param rule \code{"strict"} (\eqn{x/n > c}) or \code{"weak"} (\eqn{x/n \ge c}).
#' @return Integer vector: the smallest \eqn{x} in \eqn{0, \dots, n} that crosses,
#'   or \eqn{n + 1} if none does.
#' @examples
#' kcross(100, 0.05)          # 6: five events out of 100 do not exceed 5%
#' kcross(99, 0.05)           # 5
#' kcross(100, 0.05, "weak")  # 5
#' @export
kcross <- function(n, c, rule = "strict") {
  if (!rule %in% c("strict", "weak")) stop("rule must be 'strict' or 'weak'")
  if (any(c < 0) || any(c > 1)) stop("threshold c must be a proportion in [0, 1]")
  if (rule == "strict") {
    k <- floor(c * n + .eps) + 1
    k <- pmax(k, 1)
  } else {
    k <- ceiling(c * n - .eps)
    k <- pmax(k, 0)
  }
  as.integer(pmin(k, n + 1))
}

#' Posting rules
#'
#' \code{posted_anyarm} is the current registry rule: a row is posted when at
#' least one arm crosses the threshold. \code{posted_pooled} posts a row when both
#' arms combined cross it.
#'
#' @param x1,n1 Events and participants at risk in the drug arm.
#' @param x0,n0 Events and participants at risk in the placebo arm.
#' @inheritParams kcross
#' @return Logical vector.
#' @examples
#' posted_anyarm(0, 100, 3, 50, 0.05)  # TRUE: 3/50 exceeds 5%
#' posted_pooled(0, 100, 3, 50, 0.05)  # FALSE: 3/150 does not
#' @export
posted_anyarm <- function(x1, n1, x0, n0, c, rule = "strict") {
  (x1 >= kcross(n1, c, rule)) | (x0 >= kcross(n0, c, rule))
}

#' @rdname posted_anyarm
#' @export
posted_pooled <- function(x1, n1, x0, n0, c, rule = "strict") {
  (x1 + x0) >= kcross(n1 + n0, c, rule)
}
