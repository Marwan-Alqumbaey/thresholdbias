#' thresholdbias: bias from adverse-event reporting thresholds
#'
#' Exact theory, pooled estimators and a selection-corrected likelihood for
#' adverse-event tables posted only when an arm exceeds a frequency threshold.
#' Arm 1 is the drug arm (\code{x1} events among \code{n1}) and arm 0 is the
#' placebo arm (\code{x0} among \code{n0}). Thresholds are proportions
#' (0.05 for 5\%). Under the strict rule an arm crosses the threshold when
#' \eqn{x/n > c}, as in 42 CFR 11.48; under the weak rule it crosses when
#' \eqn{x/n \ge c}.
#'
#' @keywords internal
"_PACKAGE"
