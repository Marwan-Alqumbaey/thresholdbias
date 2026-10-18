"""Tests for thresholdbias.correct: selection-corrected random-effects likelihood."""

import numpy as np
import pytest
from scipy import integrate
from scipy.optimize import minimize
from scipy.stats import binom, norm
from thresholdbias import fit_corrected, kcross, mh_rr, posted_anyarm
from thresholdbias.correct import _prep, negll

_X, _W = np.polynomial.hermite.hermgauss(80)
_Z, _WZ = np.sqrt(2) * _X, _W / np.sqrt(np.pi)


def _row_lik(par, x1, n1, x0, n0):
    b, mu, ls = par
    p0 = np.minimum(np.exp(mu + np.exp(ls) * _Z), 0.999)
    p1 = np.minimum(np.exp(b) * p0, 0.999)
    return (binom.pmf(x1[:, None], n1[:, None], p1) * binom.pmf(x0[:, None], n0[:, None], p0)) @ _WZ, p1, p0


def ordinary_re_negll(par, x1, n1, x0, n0):
    L, _, _ = _row_lik(par, x1, n1, x0, n0)
    return -np.log(L).sum()


def zero_truncated_negll(par, x1, n1, x0, n0):
    L, p1, p0 = _row_lik(par, x1, n1, x0, n0)
    P0 = ((1 - p1) ** n1[:, None] * (1 - p0) ** n0[:, None]) @ _WZ
    return -(np.log(L) - np.log(1 - P0)).sum()


def _mle(f, start):
    r = minimize(f, start, method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=8000))
    return minimize(f, r.x, method="BFGS").x


def _sim(rng, K, theta, n1, n0, mu, sigma):
    p0 = np.exp(mu + sigma * rng.standard_normal(K))
    p1 = np.minimum(theta * p0, 0.999)
    return rng.binomial(n1, p1), np.full(K, n1), rng.binomial(n0, p0), np.full(K, n0)


@pytest.fixture(scope="module")
def data_nosel():
    rng = np.random.default_rng(42)
    return _sim(rng, 60, 1.3, 150, 100, np.log(0.04), 0.5)


ROWS = {
    "sparse": (
        np.array([7, 1, 12, 0]),
        np.array([200, 100, 150, 300]),
        np.array([6, 5, 3, 4]),
        np.array([100, 60, 150, 100]),
    ),
    "common": (np.array([60, 150, 40]), np.array([400, 600, 250]), np.array([25, 70, 30]), np.array([200, 300, 250])),
}


@pytest.mark.parametrize("rows", ["sparse", "common"])
@pytest.mark.parametrize("sigma", [0.3, 0.6, 1.5])
def test_negll_matches_numerical_integration(rows, sigma):
    """Conditional likelihood agrees with adaptive quadrature of its defining integrals."""
    x1, n1, x0, n0 = ROWS[rows]
    c = 0.05
    par = (0.2, np.log(0.03 if rows == "sparse" else 0.1), np.log(sigma))
    D = _prep(x1, n1, x0, n0, c, "strict")
    b, mu, s = par[0], par[1], np.exp(par[2])
    pts = np.linspace(-12, 12, 241)
    tot = 0.0
    for i in range(len(x1)):
        k1, k0 = kcross(n1[i], c), kcross(n0[i], c)

        def dens(z, what):
            p0 = min(np.exp(mu + s * z), 0.999)
            p1 = min(np.exp(b) * p0, 0.999)
            if what == "num":
                v = binom.pmf(x1[i], n1[i], p1) * binom.pmf(x0[i], n0[i], p0)
            else:
                S1, S0 = binom.sf(k1 - 1, n1[i], p1), binom.sf(k0 - 1, n0[i], p0)
                v = S1 + S0 - S1 * S0
            return v * norm.pdf(z)

        num = sum(
            integrate.quad(dens, a, c_, args=("num",), epsabs=0, epsrel=1e-11)[0] for a, c_ in zip(pts[:-1], pts[1:])
        )
        den = sum(
            integrate.quad(dens, a, c_, args=("den",), epsabs=0, epsrel=1e-11)[0] for a, c_ in zip(pts[:-1], pts[1:])
        )
        tot += np.log(num) - np.log(den)
    assert negll(par, D) == pytest.approx(-tot, abs=1e-6)


def test_weak_c0_equals_ordinary_re_mle(data_nosel):
    """Weak rule with c = 0 posts everything: P(posted) = 1 and the corrected MLE is the ordinary RE binomial MLE."""
    x1, n1, x0, n0 = data_nosel
    r = fit_corrected(x1, n1, x0, n0, 0.0, rule="weak", ci=None)
    ref = _mle(lambda p: ordinary_re_negll(p, x1, n1, x0, n0), [0.0, np.log(0.04), np.log(0.5)])
    assert r["logRR"] == pytest.approx(ref[0], abs=2e-4)
    assert r["mu"] == pytest.approx(ref[1], abs=2e-4)


def test_strict_c0_is_zero_truncated_mle(data_nosel):
    """Strict rule with c = 0 ('any event') conditions on X1 + X0 >= 1: a zero-truncated RE likelihood."""
    x1, n1, x0, n0 = [a[:30] for a in data_nosel]
    x1 = x1.copy()
    x0 = x0.copy()
    x1[:6] = 0
    x0[:6] = 0
    keep = (x1 + x0) > 0
    r = fit_corrected(x1[keep], n1[keep], x0[keep], n0[keep], 0.0, ci=None)
    ref = _mle(
        lambda p: zero_truncated_negll(p, x1[keep], n1[keep], x0[keep], n0[keep]), [0.0, np.log(0.04), np.log(0.5)]
    )
    assert r["logRR"] == pytest.approx(ref[0], abs=5e-4)


def test_full_likelihood_c0_equals_ordinary_mle_on_all_trials(data_nosel):
    """Full likelihood, strict c = 0: unposted = double-zero trials, P(not posted) = (1-p1)^n1 (1-p0)^n0, so the
    full likelihood is exactly the ordinary RE likelihood over all trials."""
    x1, n1, x0, n0 = [a[:40].copy() for a in data_nosel]
    x1[:8] = 0
    x0[:8] = 0
    post = (x1 + x0) > 0
    r = fit_corrected(
        x1[post], n1[post], x0[post], n0[post], 0.0, unposted=(n1[~post], n0[~post], np.zeros((~post).sum())), ci=None
    )
    ref = _mle(lambda p: ordinary_re_negll(p, x1, n1, x0, n0), [0.0, np.log(0.04), np.log(0.5)])
    assert r["logRR"] == pytest.approx(ref[0], abs=5e-4)


def test_empty_unposted_gives_full_likelihood(data_nosel):
    """An empty set of unposted trials gives the full likelihood, not the corrected likelihood."""
    x1, n1, x0, n0 = data_nosel
    post = posted_anyarm(x1, n1, x0, n0, 0.0, "weak")
    assert post.all()
    r_full = fit_corrected(x1, n1, x0, n0, 0.05, unposted=([], [], []), ci=None)
    ref = _mle(lambda p: ordinary_re_negll(p, x1, n1, x0, n0), [0.0, np.log(0.04), np.log(0.5)])
    assert r_full["logRR"] == pytest.approx(ref[0], abs=2e-4)
    r_cond = fit_corrected(x1, n1, x0, n0, 0.05, ci=None)
    assert abs(r_cond["logRR"] - r_full["logRR"]) > 1e-4


@pytest.mark.parametrize("theta", [1.0, 1.5])
def test_recovery_under_selection(theta):
    """2:1 allocation, strict 5% rule: posted-row MH is biased; the corrected estimate recovers log theta."""
    rng = np.random.default_rng(2026 + int(10 * theta))
    x1, n1, x0, n0 = _sim(rng, 4000, theta, 200, 100, np.log(0.03), 0.4)
    post = posted_anyarm(x1, n1, x0, n0, 0.05)
    xs = [a[post][:600] for a in (x1, n1, x0, n0)]
    r = fit_corrected(*xs, 0.05)
    assert r["lo"] < np.log(theta) < r["hi"]
    assert r["logRR"] == pytest.approx(np.log(theta), abs=0.08)
    if theta == 1.0:
        assert mh_rr(*xs)[0] < -0.08
    rf = fit_corrected(
        *xs,
        0.05,
        unposted=(
            n1[~post][: (~post[: post.nonzero()[0][599] + 1]).sum()],
            n0[~post][: (~post[: post.nonzero()[0][599] + 1]).sum()],
            0.05,
        ),
        ci="wald",
    )
    assert rf["logRR"] == pytest.approx(np.log(theta), abs=0.08)
    assert rf["lo"] < np.log(theta) < rf["hi"]


def test_profile_ci_brackets_estimate_and_agrees_with_wald():
    rng = np.random.default_rng(5)
    x1, n1, x0, n0 = _sim(rng, 3000, 1.2, 200, 100, np.log(0.03), 0.4)
    post = posted_anyarm(x1, n1, x0, n0, 0.05)
    xs = [a[post][:300] for a in (x1, n1, x0, n0)]
    rp = fit_corrected(*xs, 0.05, ci="profile")
    rw = fit_corrected(*xs, 0.05, ci="wald")
    assert rp["lo"] < rp["logRR"] < rp["hi"]
    assert rp["hi"] - rp["lo"] == pytest.approx(rw["hi"] - rw["lo"], rel=0.25)
    assert rp["n_posted"] == rw["n_posted"] == 300


def test_percentage_threshold_rejected():
    with pytest.raises(ValueError):
        fit_corrected([6], [100], [3], [50], 5.0)


def test_common_events_large_sigma():
    """Common events with large between-trial spread: sigma and log theta are recovered."""
    rng = np.random.default_rng(0)
    K = 150
    p0 = np.minimum(np.exp(np.log(0.08) + rng.standard_normal(3 * K)), 0.6)
    p1 = np.minimum(1.5 * p0, 0.999)
    n1 = rng.integers(150, 400, 3 * K)
    n0 = n1 // 2
    x1 = rng.binomial(n1, p1)
    x0 = rng.binomial(n0, p0)
    idx = np.nonzero(posted_anyarm(x1, n1, x0, n0, 0.05))[0][:K]
    r = fit_corrected(x1[idx], n1[idx], x0[idx], n0[idx], 0.05, ci=None)
    assert 0.7 < r["sigma"] < 1.3
    assert r["logRR"] == pytest.approx(np.log(1.5), abs=0.05)
