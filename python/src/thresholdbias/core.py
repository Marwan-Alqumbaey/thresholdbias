"""Posting rules, exact conditional moments and pooled estimators.

Notation: arm 1 is the drug arm (x1 events among n1 participants) and arm 0 is
the placebo arm (x0 among n0). A threshold ``c`` is a proportion (0.05 for 5%).
Under the strict rule an arm crosses the threshold when x/n > c, as in
42 CFR 11.48; under the weak rule it crosses when x/n >= c.
"""

import numpy as np
from scipy.special import betainc

__all__ = ["kcross", "posted_anyarm", "posted_pooled", "exact_conditional", "bias_check", "mh_rr", "iv_rr"]

_EPS = 1e-12


def kcross(n, c, rule="strict"):
    """Smallest event count that crosses the threshold.

    Parameters
    ----------
    n : int or array of int
        Number of participants at risk.
    c : float
        Threshold as a proportion in [0, 1].
    rule : {"strict", "weak"}
        "strict": x/n > c. "weak": x/n >= c.

    Returns
    -------
    int or array of int
        The smallest x in 0..n that crosses, or n + 1 if no count crosses.
    """
    n = np.asarray(n)
    if rule not in ("strict", "weak"):
        raise ValueError("rule must be 'strict' or 'weak'")
    if np.any(np.asarray(c) < 0) or np.any(np.asarray(c) > 1):
        raise ValueError("threshold c must be a proportion in [0, 1]")
    if rule == "strict":
        k = np.floor(c * n + _EPS) + 1
    else:
        k = np.ceil(c * n - _EPS)
    k = np.maximum(k, 0 if rule == "weak" else 1)
    return np.minimum(k, n + 1).astype(int)


def posted_anyarm(x1, n1, x0, n0, c, rule="strict"):
    """True where at least one arm crosses the threshold (any-arm rule, the current registry rule).

    Parameters
    ----------
    x1, n1 : array_like
        Events and participants at risk in the drug arm.
    x0, n0 : array_like
        Events and participants at risk in the placebo arm.
    c : float
        Threshold as a proportion.
    rule : {"strict", "weak"}
        Crossing rule, as in :func:`kcross`.
    """
    x1, n1, x0, n0 = (np.asarray(a) for a in (x1, n1, x0, n0))
    return (x1 >= kcross(n1, c, rule)) | (x0 >= kcross(n0, c, rule))


def posted_pooled(x1, n1, x0, n0, c, rule="strict"):
    """True where both arms combined cross the threshold (pooled-arm rule).

    Parameters are those of :func:`posted_anyarm`.
    """
    x1, n1, x0, n0 = (np.asarray(a) for a in (x1, n1, x0, n0))
    return (x1 + x0) >= kcross(n1 + n0, c, rule)


def _sf(k, n, p):
    """P(X >= k) for X ~ Bin(n, p)."""
    p = np.clip(p, 1e-300, 1 - 1e-15)
    k = np.asarray(k)
    n = np.asarray(n)
    return np.where(k <= 0, 1.0, np.where(k > n, 0.0, betainc(np.maximum(k, 1e-300), np.maximum(n - k + 1, 1e-300), p)))


def _cdf_below(k, n, p):
    """P(X < k) for X ~ Bin(n, p), accurate in the lower tail."""
    p = np.clip(p, 1e-300, 1 - 1e-15)
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    return np.where(
        k <= 0, 0.0, np.where(k > n, 1.0, betainc(np.maximum(n - k + 1, 1e-300), np.maximum(k, 1e-300), 1 - p))
    )


def _moments(n1, n0, p1, p0, c, rule="strict"):
    """Vectorized exact moments of a row given that it is posted (any-arm rule)."""
    k1, k0 = kcross(n1, c, rule), kcross(n0, c, rule)
    S1, S0 = _sf(k1, n1, p1), _sf(k0, n0, p0)
    F1, F0 = _cdf_below(k1, n1, p1), _cdf_below(k0, n0, p0)
    T1 = p1 * (_sf(k1 - 1, n1 - 1, p1) - S1)
    T0 = p0 * (_sf(k0 - 1, n0 - 1, p0) - S0)
    pp = S1 + S0 - S1 * S0
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = p1 + F0 * T1 / pp
        e0 = p0 + F1 * T0 / pp
        kap = F1 * F0 / pp
    # P(posted) * E[x_j/n_j | posted], computed without dividing by P(posted)
    m1 = pp * p1 + F0 * T1
    m0 = pp * p0 + F1 * T0
    return pp, e1, e0, kap, m1, m0


def exact_conditional(n1, n0, p1, p0, c, rule="strict"):
    """Exact moments of one row given that it is posted under the any-arm rule.

    Uses E[x_j/n_j | posted] = p_j + kappa * D_j (Supplementary Material S1),
    evaluated with tail probabilities only so that it stays accurate when
    posting is rare.

    Parameters
    ----------
    n1, n0 : int
        Participants at risk in the drug and placebo arms.
    p1, p0 : float
        True risks in the drug and placebo arms.
    c : float
        Threshold as a proportion.
    rule : {"strict", "weak"}
        Crossing rule, as in :func:`kcross`.

    Returns
    -------
    dict
        ``p_post`` (probability the row is posted), ``e1`` and ``e0``
        (expected proportions among posted rows), ``rr_cond`` (their ratio,
        the posted-row risk ratio), ``rr_true`` (p1 / p0) and ``kappa``.
    """
    pp, e1, e0, kap, _, _ = _moments(n1, n0, p1, p0, c, rule)
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(
            p_post=float(pp),
            e1=float(e1),
            e0=float(e0),
            rr_cond=float(e1 / e0),
            rr_true=float(p1 / p0) if p0 > 0 else np.nan,
            kappa=float(kap),
        )


def _gauss_hermite(m):
    """Nodes and weights (summing to 1) for expectations over a standard normal variable.

    Golub-Welsch algorithm for the probabilists' Hermite polynomials; m = 1 gives
    the single node 0.
    """
    if m < 1:
        raise ValueError("nodes must be at least 1")
    if m == 1:
        return np.zeros(1), np.ones(1)
    off = np.sqrt(np.arange(1, m))
    val, vec = np.linalg.eigh(np.diag(off, 1) + np.diag(off, -1))
    w = vec[0] ** 2
    return val, w / w.sum()


def bias_check(n1, n0, p0, rr=1.0, c=0.05, rule="strict", sd_logit=0.0, tau=0.0, nodes=21):
    """Expected distortion of a pooled risk ratio when only posted rows are pooled.

    For a set of trials with arm sizes ``n1`` (drug) and ``n0`` (placebo), returns
    the value that the Mantel-Haenszel risk ratio pooled over the rows posted
    under the any-arm rule converges to when the same event is reported by many
    trials of this mix, and compares it with the value for complete tables. The
    limit is the ratio of weighted conditional means of Remark S1.6 of the
    Supplementary Material, with weights P(posted) n1 n0 / (n1 + n0). Between-trial
    variation in the placebo risk (normal on the logit scale, SD ``sd_logit``)
    and in the log risk ratio (normal, SD ``tau``) is integrated out by
    Gauss-Hermite quadrature.

    Parameters
    ----------
    n1, n0 : array_like of int
        Participants at risk in the drug and placebo arms, one value per trial.
    p0 : float or array_like
        Median placebo risk; several values give one result per value.
    rr : float
        Median risk ratio (drug / placebo).
    c : float
        Threshold as a proportion.
    rule : {"strict", "weak"}
        Crossing rule, as in :func:`kcross`.
    sd_logit : float
        SD of the placebo risk between trials, on the logit scale.
    tau : float
        SD of the log risk ratio between trials.
    nodes : int
        Quadrature nodes for each source of variation (used only when the SD is above 0).

    Returns
    -------
    dict of numpy arrays, one entry per value of ``p0``
        ``p0``; ``rr_complete`` (limit of the MH risk ratio over complete tables);
        ``rr_posted`` (limit over posted rows); ``distortion``
        (rr_posted / rr_complete); ``p_post`` (mean probability that a trial
        posts the event).
    """
    n1 = np.atleast_1d(np.asarray(n1)).astype(int)
    n0 = np.atleast_1d(np.asarray(n0)).astype(int)
    if n1.shape != n0.shape or n1.ndim != 1:
        raise ValueError("n1 and n0 must be vectors of the same length")
    if np.any(n1 < 1) or np.any(n0 < 1):
        raise ValueError("every arm needs at least one participant")
    p0v = np.atleast_1d(np.asarray(p0, float))
    if np.any(p0v <= 0) or np.any(p0v >= 1) or rr <= 0:
        raise ValueError("p0 must lie in (0, 1) and rr must be positive")
    u, wu = _gauss_hermite(nodes) if sd_logit > 0 else (np.zeros(1), np.ones(1))
    v, wv = _gauss_hermite(nodes) if tau > 0 else (np.zeros(1), np.ones(1))
    lp = np.log(p0v / (1 - p0v))[:, None, None, None] + sd_logit * u[None, None, :, None]
    q0 = 1 / (1 + np.exp(-lp))
    q1 = np.minimum(q0 * np.exp(np.log(rr) + tau * v[None, None, None, :]), 1.0)
    m1, m0 = n1[None, :, None, None], n0[None, :, None, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pp, _, _, _, pe1, pe0 = _moments(m1, m0, q1, q0, c, rule)
    w = (m1 * m0 / (m1 + m0)) * wu[None, None, :, None] * wv[None, None, None, :]
    ax = (1, 2, 3)
    rr_complete = (w * q1).sum(ax) / (w * q0).sum(ax)
    rr_posted = (w * pe1).sum(ax) / (w * pe0).sum(ax)
    p_post = (pp * wu[None, None, :, None] * wv[None, None, None, :]).sum(ax) / len(n1)
    return dict(
        p0=p0v,
        rr_complete=rr_complete,
        rr_posted=rr_posted,
        distortion=rr_posted / rr_complete,
        p_post=p_post,
    )


def mh_rr(x1, n1, x0, n0):
    """Mantel-Haenszel pooled log risk ratio with the Greenland-Robins standard error.

    Parameters
    ----------
    x1, n1 : array_like
        Events and participants at risk in the drug arm, one value per row.
    x0, n0 : array_like
        The same for the placebo arm.

    Returns
    -------
    tuple of float
        (log RR, SE); both NaN when either arm has no events in total.
    """
    x1, n1, x0, n0 = (np.asarray(a, float) for a in (x1, n1, x0, n0))
    N = n1 + n0
    R = (x1 * n0 / N).sum()
    S = (x0 * n1 / N).sum()
    if R <= 0 or S <= 0:
        return np.nan, np.nan
    P = ((n1 * n0 * (x1 + x0) - x1 * x0 * N) / N**2).sum()
    return np.log(R / S), np.sqrt(P / (R * S))


def iv_rr(x1, n1, x0, n0, cc=0.5, method="fixed"):
    """Inverse-variance pooled log risk ratio.

    Parameters
    ----------
    x1, n1, x0, n0 : array_like
        Events and participants at risk in the drug and placebo arms, one value per row.
    cc : float
        Continuity correction added to every cell of tables with no events in an arm.
    method : {"fixed", "dl"}
        Fixed-effect pooling or DerSimonian-Laird random effects.

    Returns
    -------
    tuple of float
        (log RR, SE).
    """
    if method not in ("fixed", "dl"):
        raise ValueError("method must be 'fixed' or 'dl'")
    x1, n1, x0, n0 = (np.asarray(a, float) for a in (x1, n1, x0, n0))
    z = (x1 == 0) | (x0 == 0)
    a = x1 + cc * z
    b = x0 + cc * z
    m1 = n1 + 2 * cc * z
    m0 = n0 + 2 * cc * z
    y = np.log(a / m1) - np.log(b / m0)
    v = 1 / a - 1 / m1 + 1 / b - 1 / m0
    w = 1 / v
    est = (w * y).sum() / w.sum()
    if method == "fixed" or len(y) < 2:
        return est, 1 / np.sqrt(w.sum())
    Q = (w * (y - est) ** 2).sum()
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - (len(y) - 1)) / C)
    ws = 1 / (v + tau2)
    return (ws * y).sum() / ws.sum(), 1 / np.sqrt(ws.sum())
