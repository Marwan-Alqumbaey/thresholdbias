.pmax <- 0.999

.logsumexp <- function(m) {
  # row-wise log(sum(exp(m))) for a matrix
  mx <- apply(m, 1, max)
  mx + log(rowSums(exp(m - mx)))
}

.logaddexp <- function(a, b) {
  mx <- pmax(a, b)
  mx + log(exp(a - mx) + exp(b - mx))
}

# Gauss-Hermite nodes and weights (physicists' convention) by the Golub-Welsch method.
.gauss_hermite <- function(n) {
  J <- matrix(0, n, n)
  off <- sqrt(seq_len(n - 1) / 2)
  J[cbind(1:(n - 1), 2:n)] <- off
  J[cbind(2:n, 1:(n - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  list(x = e$values[o], w = sqrt(pi) * e$vectors[1, o]^2)
}

.quad <- local({
  gh <- .gauss_hermite(20)
  zg <- seq(-9, 9, length.out = 361)
  lzw <- -0.5 * zg^2 - 0.5 * log(2 * pi) + log(zg[2] - zg[1])
  lzw <- lzw - (max(lzw) + log(sum(exp(lzw - max(lzw)))))
  list(agh_x = gh$x, agh_w = gh$w, zg = zg, lzw = lzw,
       ltail = stats::pnorm(9, lower.tail = FALSE, log.p = TRUE))
})

.logbin <- function(x, n, p) {
  p <- pmin(pmax(p, 1e-300), 1 - 1e-15)
  lgamma(n + 1) - lgamma(x + 1) - lgamma(n - x + 1) + x * log(p) + (n - x) * log1p(-p)
}

# log E[Bin(x1; n1, theta p0) Bin(x0; n0, p0)], log p0 = mu + s Z, by adaptive Gauss-Hermite quadrature.
.lognum <- function(beta, mu, s, x1, n1, x0, n0) {
  th <- exp(beta)
  zmax <- (log(.pmax) - max(beta, 0) - mu) / s - 1e-9
  z <- pmin(pmax((log((x1 / th + x0 + 0.5) / (n1 + n0 + 1)) - mu) / s, -40), zmax)
  for (it in 1:50) {
    q0 <- exp(mu + s * z)
    q1 <- th * q0
    g <- s * (x1 + x0 - (n1 - x1) * q1 / (1 - q1) - (n0 - x0) * q0 / (1 - q0)) - z
    H <- -s * s * ((n1 - x1) * q1 / (1 - q1)^2 + (n0 - x0) * q0 / (1 - q0)^2) - 1
    zn <- pmin(pmax(z - g / H, -40), zmax)
    done <- max(abs(zn - z)) < 1e-10
    z <- zn
    if (done) break
  }
  q0 <- exp(mu + s * z)
  q1 <- th * q0
  H <- -s * s * ((n1 - x1) * q1 / (1 - q1)^2 + (n0 - x0) * q0 / (1 - q0)^2) - 1
  sd <- 1 / sqrt(-H)
  zk <- z + sqrt(2) * outer(sd, .quad$agh_x)
  p0 <- pmin(exp(mu + s * zk), .pmax)
  p1 <- pmin(th * p0, .pmax)
  lf <- .logbin(x1, n1, p1) + .logbin(x0, n0, p0) - 0.5 * zk^2 - 0.5 * log(2 * pi)
  lw <- matrix(log(.quad$agh_w) + .quad$agh_x^2, nrow = length(z), ncol = length(.quad$agh_x), byrow = TRUE) +
    log(sqrt(2) * sd)
  .logsumexp(lf + lw)
}

# Posted rows, their cut-offs, and the distinct designs.
.prep <- function(x1, n1, x0, n0, c, rule) {
  m <- length(x1)
  c <- rep_len(c, m)
  k1 <- vapply(seq_len(m), function(i) kcross(n1[i], c[i], rule), integer(1))
  k0 <- vapply(seq_len(m), function(i) kcross(n0[i], c[i], rule), integer(1))
  key <- paste(n1, n0, k1, k0)
  first <- !duplicated(key)
  inv <- match(key, key[first])
  list(x1 = as.numeric(x1), n1 = as.numeric(n1), x0 = as.numeric(x0), n0 = as.numeric(n0),
       k1 = k1, k0 = k0, first = which(first), inv = inv)
}

# Distinct unposted designs (n1, n0, k1, k0) and the number of trials with each.
.prep_unposted <- function(unposted, rule) {
  un1 <- as.numeric(unposted[[1]]); un0 <- as.numeric(unposted[[2]])
  uc <- rep_len(as.numeric(unposted[[3]]), length(un1))
  if (length(un0) != length(un1)) stop("unposted arrays must have the same length")
  if (length(un1) == 0) {
    return(list(n1 = numeric(0), n0 = numeric(0), k1 = numeric(0), k0 = numeric(0), w = numeric(0)))
  }
  uk1 <- vapply(seq_along(un1), function(i) kcross(un1[i], uc[i], rule), integer(1))
  uk0 <- vapply(seq_along(un0), function(i) kcross(un0[i], uc[i], rule), integer(1))
  key <- paste(un1, un0, uk1, uk0)
  tab <- table(factor(key, levels = unique(key)))
  fi <- match(names(tab), key)
  list(n1 = un1[fi], n0 = un0[fi], k1 = uk1[fi], k0 = uk0[fi], w = as.numeric(tab))
}

# Negative log-likelihood at par = c(log theta, mu, log sigma). U = NULL gives the
# corrected likelihood of posted rows; otherwise the full likelihood is used.
negll <- function(par, D, U = NULL) {
  beta <- par[1]; mu <- par[2]; s <- exp(par[3])
  num <- .lognum(beta, mu, s, D$x1, D$n1, D$x0, D$n0)
  p0g <- pmin(exp(mu + s * .quad$zg), .pmax)
  p1g <- pmin(exp(beta) * p0g, .pmax)
  G <- length(.quad$zg)
  lzw <- matrix(.quad$lzw, nrow = 1)
  if (is.null(U)) {
    f <- D$first
    S1 <- .sf(matrix(D$k1[f], length(f), G), matrix(D$n1[f], length(f), G), matrix(p1g, length(f), G, byrow = TRUE))
    S0 <- .sf(matrix(D$k0[f], length(f), G), matrix(D$n0[f], length(f), G), matrix(p0g, length(f), G, byrow = TRUE))
    lpost <- log(pmax(S1 + S0 - S1 * S0, 1e-300))
    den <- .logsumexp(lpost + lzw[rep(1, length(f)), , drop = FALSE])
    den <- .logaddexp(den, .quad$ltail)
    return(-sum(num - den[D$inv]))
  }
  ll <- sum(num)
  if (length(U$n1) > 0) {
    u <- length(U$n1)
    lnp <- log(pmax(.cdf_below(matrix(U$k1, u, G), matrix(U$n1, u, G), matrix(p1g, u, G, byrow = TRUE)), 1e-300)) +
      log(pmax(.cdf_below(matrix(U$k0, u, G), matrix(U$n0, u, G), matrix(p0g, u, G, byrow = TRUE)), 1e-300))
    ll <- ll + sum(U$w * .logaddexp(.logsumexp(lnp + lzw[rep(1, u), , drop = FALSE]), .quad$ltail + lnp[, 1]))
  }
  -ll
}

#' Selection-corrected estimate of a common risk ratio
#'
#' Fits the random-effects model with \eqn{\log p_{0i} \sim N(\mu, \sigma^2)} and
#' \eqn{p_{1i} = \theta p_{0i}} to rows that were posted under the any-arm rule.
#' Each posted row contributes its likelihood divided by its probability of being
#' posted. When \code{unposted} is given, the full likelihood is used instead:
#' each posted row contributes its likelihood, and each trial that monitored the
#' event but did not post it contributes its probability of not being posted.
#' Both risks are capped at 0.999.
#'
#' @inheritParams posted_anyarm
#' @param c Threshold(s) as proportions, one per row or a single value.
#' @param rule \code{"strict"} or \code{"weak"}.
#' @param unposted Optional list \code{list(n1, n0, c)} for trials that monitored
#'   the event but did not post it.
#' @param ci Type of 95\% confidence interval for the log risk ratio:
#'   \code{"profile"}, \code{"wald"} or \code{"none"}.
#' @return A list with \code{logRR}, \code{mu}, \code{sigma}, \code{converged},
#'   \code{sigma_at_bound} (true when sigma is at most 0.001 or at least 10, the
#'   limits used in the last optimization step), \code{n_posted} (number of
#'   posted rows) and, when requested, the confidence limits \code{lo} and
#'   \code{hi} (with \code{se} for Wald intervals).
#' @details With few posted rows the model is weakly identified: sigma often
#'   goes to a bound and the estimate for a single event can be far from the
#'   complete-data value. In the registry experiment of the accompanying article,
#'   the median absolute error was 0.4 to 0.7 on the log scale with fewer than 5
#'   posted rows, 0.2 with 5 to 9 and below 0.1 with 10 or more, while the
#'   profile intervals widened accordingly. Report the profile interval, and
#'   treat estimates from fewer than 5 posted rows as uninformative.
#' @examples
#' set.seed(1)
#' p0 <- exp(log(0.03) + 0.4 * rnorm(400))
#' x1 <- rbinom(400, 200, p0); x0 <- rbinom(400, 100, p0)
#' keep <- posted_anyarm(x1, 200, x0, 100, 0.05)
#' fit_corrected(x1[keep], rep(200, sum(keep)), x0[keep], rep(100, sum(keep)), 0.05, ci = "none")$logRR
#' @export
fit_corrected <- function(x1, n1, x0, n0, c, rule = "strict", unposted = NULL,
                          ci = c("profile", "wald", "none")) {
  ci <- match.arg(ci)
  m <- length(x1)
  n1 <- rep_len(n1, m); n0 <- rep_len(n0, m)
  if (!length(c) %in% c(1, m)) stop("c must have length 1 or one value per row")
  D <- .prep(x1, n1, x0, n0, c, rule)
  U <- if (is.null(unposted)) NULL else .prep_unposted(unposted, rule)
  f <- function(p) negll(p, D, U)
  best <- NULL
  for (s0 in c(0.3, 0.7, 1.5)) {
    start <- c(0, log(max(sum(D$x0) / sum(D$n0), 1e-3)) - 0.5 * s0, log(s0))
    r <- stats::optim(start, f, method = "Nelder-Mead", control = list(reltol = 1e-12, maxit = 4000))
    if (is.null(best) || r$value < best$value) best <- r
  }
  lower <- c(-10, log(1e-8), log(1e-3))
  upper <- c(10, log(.pmax), log(10))
  xs <- pmin(pmax(best$par, lower), upper)
  r <- stats::optim(xs, f, method = "L-BFGS-B", lower = lower, upper = upper)
  if (r$value > best$value) r <- best
  b <- r$par[1]; mu <- r$par[2]; ls <- r$par[3]; lmin <- r$value
  out <- list(logRR = b, mu = mu, sigma = exp(ls), converged = r$convergence == 0,
              sigma_at_bound = ls <= log(1e-3) + 1e-6 || ls >= log(10) - 1e-6,
              n_posted = length(x1))
  if (ci == "profile") {
    cut <- stats::qchisq(0.95, 1) / 2
    starts <- list(c(mu, max(ls, log(1e-3))), c(mu, log(0.5)))
    prof <- function(bb) {
      g <- function(q) f(c(bb, q[1], q[2]))
      fits <- lapply(starts, function(st) stats::optim(st, g, method = "Nelder-Mead",
                                                        control = list(reltol = 1e-10, maxit = 4000)))
      best <- fits[[which.min(vapply(fits, function(z) z$value, numeric(1)))]]
      starts[[3]] <<- best$par
      best$value - lmin - cut
    }
    res <- tryCatch({
      lo <- if (prof(b - 4) > 0) stats::uniroot(prof, c(b - 4, b), tol = 1e-8)$root else -Inf
      hi <- if (prof(b + 4) > 0) stats::uniroot(prof, c(b, b + 4), tol = 1e-8)$root else Inf
      c(lo, hi)
    }, error = function(e) c(NA_real_, NA_real_))
    out$lo <- res[1]; out$hi <- res[2]
  } else if (ci == "wald") {
    h <- 1e-3
    x <- r$par
    E <- diag(3) * h
    H <- matrix(0, 3, 3)
    for (i in 1:3) for (j in i:3) {
      H[i, j] <- H[j, i] <- (f(x + E[i, ] + E[j, ]) - f(x + E[i, ] - E[j, ]) -
                               f(x - E[i, ] + E[j, ]) + f(x - E[i, ] - E[j, ])) / (4 * h * h)
    }
    se <- tryCatch(sqrt(solve(H)[1, 1]), error = function(e) NA_real_)
    out$se <- se; out$lo <- b - 1.959964 * se; out$hi <- b + 1.959964 * se
  }
  out
}
