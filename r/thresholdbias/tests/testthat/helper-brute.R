# Brute-force reference: full double sum over the product binomial distribution.
brute <- function(n1, n0, p1, p0, c, rule = "strict", pooled = FALSE) {
  x1 <- 0:n1; x0 <- 0:n0
  f <- outer(dbinom(x1, n1, p1), dbinom(x0, n0, p0))
  if (pooled) {
    post <- outer(x1, x0, "+") >= kcross(n1 + n0, c, rule)
  } else {
    post <- outer(x1 >= kcross(n1, c, rule), x0 >= kcross(n0, c, rule), "|")
  }
  P <- sum(f[post])
  e1 <- sum((f * post) * outer(x1 / n1, rep(1, n0 + 1))) / P
  e0 <- sum((f * post) * outer(rep(1, n1 + 1), x0 / n0)) / P
  N <- n1 + n0
  mh <- sum((f * post) * outer(x1 * n0 / N, rep(1, n0 + 1))) / sum((f * post) * outer(rep(1, n1 + 1), x0 * n1 / N))
  list(P = P, e1 = e1, e0 = e0, mh = mh)
}
