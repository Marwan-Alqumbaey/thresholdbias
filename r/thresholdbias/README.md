# thresholdbias (R)

Tools for adverse-event tables posted only when an arm exceeds a frequency
threshold, as on ClinicalTrials.gov. The R package implements the same methods as the
Python package of the same name, and both give the same results (see
`tests/testthat/test-python-equivalence.R`).

```r
# install.packages("remotes")
remotes::install_github("Marwan-Alqumbaey/thresholdbias", subdir = "r/thresholdbias")

library(thresholdbias)
exact_conditional(200, 100, 0.03, 0.03, 0.05)$rr_cond   # 0.729

# Distortion expected for the arm sizes of the trials in a review, at three placebo risks
bias_check(c(200, 150, 400, 90), c(100, 50, 200, 45), c(0.01, 0.02, 0.03))$distortion   # 0.22 0.49 0.72

# Selection-corrected common log risk ratio from posted rows
set.seed(1)
p0 <- exp(log(0.03) + 0.4 * rnorm(400))
x1 <- rbinom(400, 200, p0); x0 <- rbinom(400, 100, p0)
keep <- posted_anyarm(x1, 200, x0, 100, 0.05)
fit <- fit_corrected(x1[keep], 200, x0[keep], 100, c = 0.05)
c(fit$logRR, fit$lo, fit$hi)   # 0.020 -0.076 0.115
```

| Function | Purpose |
|---|---|
| `kcross(n, c, rule)` | Smallest event count that crosses the threshold |
| `posted_anyarm()`, `posted_pooled()` | Posting rules (any-arm rule; pooled-arm rule) |
| `exact_conditional()` | Exact moments of a row given that it is posted |
| `bias_check()` | Distortion expected for a set of trials, with optional between-trial variation |
| `mh_rr()` | Mantel-Haenszel log risk ratio with Greenland-Robins standard error |
| `iv_rr()` | Inverse-variance or DerSimonian-Laird log risk ratio |
| `fit_corrected()` | Selection-corrected random-effects likelihood for a common risk ratio |

Thresholds are proportions (0.05 for 5%). The strict rule (`x/n > c`) is the
default; `rule = "weak"` gives `x/n >= c`.
