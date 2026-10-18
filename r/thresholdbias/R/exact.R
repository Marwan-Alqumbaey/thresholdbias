# P(X >= k) and P(X < k) for X ~ Bin(n, p), through the regularized incomplete beta function.
.sf <- function(k, n, p) {
  p <- pmin(pmax(p, 1e-300), 1 - 1e-15)
  ifelse(k <= 0, 1, ifelse(k > n, 0, stats::pbeta(p, pmax(k, 1e-300), pmax(n - k + 1, 1e-300))))
}

.cdf_below <- function(k, n, p) {
  p <- pmin(pmax(p, 1e-300), 1 - 1e-15)
  ifelse(k <= 0, 0, ifelse(k > n, 1, stats::pbeta(1 - p, pmax(n - k + 1, 1e-300), pmax(k, 1e-300))))
}

#' Exact moments of a row given that it is posted
#'
#' For one adverse-event term in one trial, returns the probability that the row
#' is posted under the any-arm rule and the expected event proportion in each arm
#' among posted rows, \eqn{E[x_j/n_j \mid \mathrm{posted}] = p_j + \kappa D_j}.
#' Their ratio, \code{rr_cond}, is the value that a Mantel-Haenszel risk ratio
#' pooled over many posted rows of this design converges to.
#'
#' @param n1,n0 Participants at risk in the drug and placebo arms.
#' @param p1,p0 True risks in the drug and placebo arms.
#' @inheritParams kcross
#' @return A list with \code{p_post}, \code{e1}, \code{e0}, \code{rr_cond},
#'   \code{rr_true} and \code{kappa}.
#' @examples
#' exact_conditional(200, 100, 0.03, 0.03, 0.05)$rr_cond  # about 0.73
#' @export
exact_conditional <- function(n1, n0, p1, p0, c, rule = "strict") {
  k1 <- kcross(n1, c, rule)
  k0 <- kcross(n0, c, rule)
  S1 <- .sf(k1, n1, p1)
  S0 <- .sf(k0, n0, p0)
  F1 <- .cdf_below(k1, n1, p1)
  F0 <- .cdf_below(k0, n0, p0)
  T1 <- p1 * (.sf(k1 - 1, n1 - 1, p1) - S1)
  T0 <- p0 * (.sf(k0 - 1, n0 - 1, p0) - S0)
  pp <- S1 + S0 - S1 * S0
  e1 <- p1 + F0 * T1 / pp
  e0 <- p0 + F1 * T0 / pp
  list(p_post = pp, e1 = e1, e0 = e0, rr_cond = e1 / e0,
       rr_true = if (p0 > 0) p1 / p0 else NA_real_, kappa = F1 * F0 / pp)
}

# Nodes and weights (summing to 1) for expectations over a standard normal variable
# (Golub-Welsch algorithm for the probabilists' Hermite polynomials).
.gauss_hermite <- function(m) {
  if (m < 1) stop("nodes must be at least 1")
  if (m == 1) return(list(x = 0, w = 1))
  J <- matrix(0, m, m)
  off <- sqrt(seq_len(m - 1))
  J[cbind(1:(m - 1), 2:m)] <- off
  J[cbind(2:m, 1:(m - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  w <- e$vectors[1, o]^2
  list(x = e$values[o], w = w / sum(w))
}

#' Expected distortion of a pooled risk ratio when only posted rows are pooled
#'
#' For a set of trials with arm sizes \code{n1} (drug) and \code{n0} (placebo),
#' returns the value that the Mantel-Haenszel risk ratio pooled over the rows
#' posted under the any-arm rule converges to when the same event is reported by
#' many trials of this mix, and compares it with the value for complete tables.
#' The limit is the ratio of weighted conditional means of Remark S1.6 of the
#' Supplementary Material, with weights \eqn{P(\mathrm{posted})\, n_1 n_0/(n_1+n_0)}.
#' Between-trial variation in the placebo risk (normal on the logit scale, SD
#' \code{sd_logit}) and in the log risk ratio (normal, SD \code{tau}) is
#' integrated out by Gauss-Hermite quadrature.
#'
#' @param n1,n0 Participants at risk in the drug and placebo arms, one value per trial.
#' @param p0 Median placebo risk; several values give one result per value.
#' @param rr Median risk ratio (drug / placebo).
#' @inheritParams kcross
#' @param sd_logit SD of the placebo risk between trials, on the logit scale.
#' @param tau SD of the log risk ratio between trials.
#' @param nodes Quadrature nodes for each source of variation (used only when the SD is above 0).
#' @return A data frame with one row per value of \code{p0} and columns \code{p0},
#'   \code{rr_complete} (limit of the MH risk ratio over complete tables),
#'   \code{rr_posted} (limit over posted rows), \code{distortion}
#'   (\code{rr_posted / rr_complete}) and \code{p_post} (mean probability that a
#'   trial posts the event).
#' @examples
#' bias_check(n1 = c(200, 150, 400), n0 = c(100, 50, 200), p0 = c(0.01, 0.02, 0.03))
#' @export
bias_check <- function(n1, n0, p0, rr = 1, c = 0.05, rule = "strict",
                       sd_logit = 0, tau = 0, nodes = 21) {
  n1 <- as.integer(n1)
  n0 <- as.integer(n0)
  if (length(n1) != length(n0)) stop("n1 and n0 must have the same length")
  if (any(n1 < 1) || any(n0 < 1)) stop("every arm needs at least one participant")
  if (any(p0 <= 0) || any(p0 >= 1) || rr <= 0) stop("p0 must lie in (0, 1) and rr must be positive")
  gu <- if (sd_logit > 0) .gauss_hermite(nodes) else list(x = 0, w = 1)
  gv <- if (tau > 0) .gauss_hermite(nodes) else list(x = 0, w = 1)
  k1 <- kcross(n1, c, rule)
  k0 <- kcross(n0, c, rule)
  w0 <- n1 * n0 / (n1 + n0)
  res <- lapply(p0, function(p) {
    a <- b <- num <- den <- pp_sum <- 0
    for (i in seq_along(gu$x)) for (j in seq_along(gv$x)) {
      q0 <- stats::plogis(stats::qlogis(p) + sd_logit * gu$x[i])
      q1 <- min(q0 * exp(log(rr) + tau * gv$x[j]), 1)
      S1 <- .sf(k1, n1, q1); S0 <- .sf(k0, n0, q0)
      F1 <- .cdf_below(k1, n1, q1); F0 <- .cdf_below(k0, n0, q0)
      T1 <- q1 * (.sf(k1 - 1, n1 - 1, q1) - S1)
      T0 <- q0 * (.sf(k0 - 1, n0 - 1, q0) - S0)
      pp <- S1 + S0 - S1 * S0
      wij <- gu$w[i] * gv$w[j]
      a <- a + wij * sum(w0 * q1)
      b <- b + wij * sum(w0 * q0)
      num <- num + wij * sum(w0 * (pp * q1 + F0 * T1))
      den <- den + wij * sum(w0 * (pp * q0 + F1 * T0))
      pp_sum <- pp_sum + wij * mean(pp)
    }
    c(p0 = p, rr_complete = a / b, rr_posted = num / den,
      distortion = (num / den) / (a / b), p_post = pp_sum)
  })
  as.data.frame(do.call(rbind, res))
}
