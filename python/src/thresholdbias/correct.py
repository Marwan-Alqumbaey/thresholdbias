"""Selection-corrected estimation of a common risk ratio from posted rows.

Model. Trial i has placebo risk p0_i with log p0_i ~ N(mu, sigma^2) and drug
risk p1_i = theta * p0_i. A row is observed only when it is posted under the
any-arm rule with threshold c_i. Each posted row contributes

    L_i = E[Bin(x1_i; n1_i, theta p0) Bin(x0_i; n0_i, p0)] / E[P(posted_i | theta p0, p0)],

the expectation being over the distribution of p0, with both risks capped at
0.999. When the trials that monitored the event without posting it are known,
the full likelihood is used instead: each posted row contributes the numerator
of L_i and each unposted trial contributes E[P(not posted_i)]. The parameters
are (log theta, mu, log sigma).

Numerical integration: adaptive Gauss-Hermite quadrature (20 nodes) for the
numerator, and an equally spaced grid of 361 points on the standard-normal
scale for the posting probabilities.
"""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import chi2, norm

from .core import _cdf_below, _sf, kcross

__all__ = ["fit_corrected"]

PMAX = 0.999
_AGH_X, _AGH_W = np.polynomial.hermite.hermgauss(20)
_ZG = np.linspace(-9.0, 9.0, 361)
_LZW = (-0.5 * _ZG**2 - 0.5 * np.log(2 * np.pi) + np.log(_ZG[1] - _ZG[0]))[None, :]
_LZW = _LZW - np.logaddexp.reduce(_LZW, axis=1, keepdims=True)
_LTAIL = float(norm.logsf(9.0))


def _logbin(x, n, p):
    p = np.clip(p, 1e-300, 1 - 1e-15)
    return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + x * np.log(p) + (n - x) * np.log1p(-p)


def _prep(x1, n1, x0, n0, c, rule):
    """Posted rows as column vectors, their cut-offs, and the distinct designs."""
    x1, n1, x0, n0 = [
        a[:, None] for a in np.broadcast_arrays(*[np.atleast_1d(np.asarray(a, float)) for a in (x1, n1, x0, n0)])
    ]
    c = np.broadcast_to(np.asarray(c, float), (x1.shape[0],))
    k1 = np.array([kcross(int(a), float(cc), rule) for a, cc in zip(n1[:, 0], c)], float)[:, None]
    k0 = np.array([kcross(int(a), float(cc), rule) for a, cc in zip(n0[:, 0], c)], float)[:, None]
    _, first, inv = np.unique(np.hstack([n1, n0, k1, k0]), axis=0, return_index=True, return_inverse=True)
    return x1, n1, x0, n0, k1, k0, first, np.ravel(inv)


def _prep_unposted(unposted, rule):
    """Distinct unposted designs (n1, n0, k1, k0) and the number of trials with each."""
    un1, un0, uc = [np.atleast_1d(np.asarray(a, float)) for a in unposted]
    uc = np.broadcast_to(uc, un1.shape) if uc.size == 1 else uc
    if uc.shape != un1.shape or un0.shape != un1.shape:
        raise ValueError("unposted arrays must have the same length")
    if un1.size == 0:
        e = np.zeros((0, 1))
        return e, e, e, e, np.zeros(0)
    uk1 = np.array([kcross(int(a), float(cc), rule) for a, cc in zip(un1, uc)], float)[:, None]
    uk0 = np.array([kcross(int(a), float(cc), rule) for a, cc in zip(un0, uc)], float)[:, None]
    key, uw = np.unique(np.hstack([un1[:, None], un0[:, None], uk1, uk0]), axis=0, return_counts=True)
    return key[:, [0]], key[:, [1]], key[:, [2]], key[:, [3]], uw.astype(float)


def _lognum(beta, mu, s, x1, n1, x0, n0):
    """log E[Bin(x1; n1, theta p0) Bin(x0; n0, p0)] by adaptive Gauss-Hermite quadrature."""
    th = np.exp(beta)
    zmax = (np.log(PMAX) - max(beta, 0.0) - mu) / s - 1e-9
    z = np.clip((np.log((x1 / th + x0 + 0.5) / (n1 + n0 + 1.0)) - mu) / s, -40, zmax)
    for _ in range(50):
        q0 = np.exp(mu + s * z)
        q1 = th * q0
        g = s * (x1 + x0 - (n1 - x1) * q1 / (1 - q1) - (n0 - x0) * q0 / (1 - q0)) - z
        H = -s * s * ((n1 - x1) * q1 / (1 - q1) ** 2 + (n0 - x0) * q0 / (1 - q0) ** 2) - 1
        zn = np.clip(z - g / H, -40, zmax)
        done = np.max(np.abs(zn - z)) < 1e-10
        z = zn
        if done:
            break
    q0 = np.exp(mu + s * z)
    q1 = th * q0
    H = -s * s * ((n1 - x1) * q1 / (1 - q1) ** 2 + (n0 - x0) * q0 / (1 - q0) ** 2) - 1
    sd = 1 / np.sqrt(-H)
    zk = z + np.sqrt(2) * sd * _AGH_X[None, :]
    p0 = np.minimum(np.exp(mu + s * zk), PMAX)
    p1 = np.minimum(th * p0, PMAX)
    lf = _logbin(x1, n1, p1) + _logbin(x0, n0, p0) - 0.5 * zk**2 - 0.5 * np.log(2 * np.pi)
    lw = np.log(_AGH_W)[None, :] + _AGH_X[None, :] ** 2 + np.log(np.sqrt(2) * sd)
    return np.logaddexp.reduce(lf + lw, axis=1)


def _grid_p(beta, mu, s):
    p0 = np.minimum(np.exp(mu + s * _ZG[None, :]), PMAX)
    return np.minimum(np.exp(beta) * p0, PMAX), p0


def negll(par, D, U=None):
    """Negative log-likelihood at par = (log theta, mu, log sigma).

    ``U=None`` gives the corrected likelihood of posted rows; otherwise ``U``
    holds the unposted designs and their counts and the full likelihood is used.
    """
    beta, mu, ls = par
    s = np.exp(ls)
    x1, n1, x0, n0, k1, k0, first, inv = D
    num = _lognum(beta, mu, s, x1, n1, x0, n0)
    p1, p0 = _grid_p(beta, mu, s)
    if U is None:
        S1 = _sf(k1[first], n1[first], p1)
        S0 = _sf(k0[first], n0[first], p0)
        lpost = np.log(np.maximum(S1 + S0 - S1 * S0, 1e-300))
        den = np.logaddexp.reduce(lpost + _LZW, axis=1)
        den = np.logaddexp(den, _LTAIL)
        return -(num - den[inv]).sum()
    ll = num.sum()
    un1, un0, uk1, uk0, uw = U
    if un1.shape[0] > 0:
        lnp = np.log(np.maximum(_cdf_below(uk1, un1, p1), 1e-300)) + np.log(
            np.maximum(_cdf_below(uk0, un0, p0), 1e-300)
        )
        ll += (uw * np.logaddexp(np.logaddexp.reduce(lnp + _LZW, axis=1), _LTAIL + lnp[:, 0])).sum()
    return -ll


def fit_corrected(x1, n1, x0, n0, c, rule="strict", unposted=None, ci="profile"):
    """Fit the selection-corrected model to posted rows.

    Parameters
    ----------
    x1, n1, x0, n0 : array_like
        Counts and numbers at risk of the posted rows (drug, placebo).
    c : float or array_like
        Threshold(s) as proportions, one per row or a single value.
    rule : {"strict", "weak"}
        Crossing rule, as in :func:`thresholdbias.kcross`.
    unposted : tuple of array_like, optional
        ``(n1, n0, c)`` for trials that monitored the event but did not post it.
        When given (even with no trials), the full likelihood is used.
    ci : {"profile", "wald", "none"}
        Type of 95% confidence interval for the log risk ratio (``None`` is the same as ``"none"``).

    Returns
    -------
    dict
        ``logRR``, ``mu``, ``sigma``, ``converged``, ``sigma_at_bound`` (true
        when sigma is at most 0.001 or at least 10, the limits used in the last
        optimization step), ``n_posted`` (number of posted rows) and, when
        requested, the CI limits ``lo`` and ``hi`` (with ``se`` for Wald intervals).

    Notes
    -----
    With few posted rows the model is weakly identified: sigma often goes to a
    bound and the estimate for a single event can be far from the complete-data
    value. In the registry experiment of the accompanying article, the median
    absolute error was 0.4 to 0.7 on the log scale with fewer than 5 posted rows,
    0.2 with 5 to 9 and below 0.1 with 10 or more, while the profile intervals
    widened accordingly. Report the profile interval, and treat estimates from
    fewer than 5 posted rows as uninformative.
    """
    if ci not in ("profile", "wald", "none", None):
        raise ValueError("ci must be 'profile', 'wald' or 'none'")
    D = _prep(x1, n1, x0, n0, c, rule)
    U = None if unposted is None else _prep_unposted(unposted, rule)
    x0tot = D[2].sum()
    n0tot = D[3].sum()

    def f(p):
        return negll(p, D, U)

    best = None
    for s0 in (0.3, 0.7, 1.5):
        start = [0.0, np.log(max(x0tot / n0tot, 1e-3)) - 0.5 * s0, np.log(s0)]
        r = minimize(f, start, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    bounds = [(-10, 10), (np.log(1e-8), np.log(PMAX)), (np.log(1e-3), np.log(10.0))]
    xs = np.clip(best.x, [b[0] for b in bounds], [b[1] for b in bounds])
    r = minimize(f, xs, method="L-BFGS-B", bounds=bounds)
    if r.fun > best.fun:
        r = best
    b, mu, ls = r.x
    lmin = r.fun
    b, mu, ls = float(b), float(mu), float(ls)
    out = dict(
        logRR=b,
        mu=mu,
        sigma=float(np.exp(ls)),
        converged=bool(r.success),
        sigma_at_bound=bool(ls <= np.log(1e-3) + 1e-6 or ls >= np.log(10.0) - 1e-6),
        n_posted=int(D[0].shape[0]),
    )
    if ci == "profile":
        cut = chi2.ppf(0.95, 1) / 2

        starts = [[mu, max(ls, np.log(1e-3))], [mu, np.log(0.5)]]

        def prof(bb):
            def g(q):
                return f([bb, q[0], q[1]])

            fits = [
                minimize(g, st, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
                for st in starts
            ]
            rr = min(fits, key=lambda z: z.fun)
            starts[2:] = [list(rr.x)]
            return rr.fun - lmin - cut

        try:
            lo = brentq(prof, b - 4, b) if prof(b - 4) > 0 else -np.inf
            hi = brentq(prof, b, b + 4) if prof(b + 4) > 0 else np.inf
        except (ValueError, RuntimeError):
            lo, hi = np.nan, np.nan
        out.update(lo=float(lo), hi=float(hi))
    elif ci == "wald":
        h = 1e-3
        x = np.asarray(r.x, float)
        H = np.zeros((3, 3))
        E = np.eye(3) * h
        for i in range(3):
            for j in range(i, 3):
                H[i, j] = H[j, i] = (
                    f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j]) + f(x - E[i] - E[j])
                ) / (4 * h * h)
        try:
            se = float(np.sqrt(np.linalg.inv(H)[0, 0]))
        except np.linalg.LinAlgError:
            se = np.nan
        out.update(se=se, lo=b - 1.959964 * se, hi=b + 1.959964 * se)
    return out
