"""Tests for thresholdbias.core: posting rules, exact conditional moments and pooled estimators."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import binom
from thresholdbias import bias_check, exact_conditional, iv_rr, kcross, mh_rr, posted_anyarm, posted_pooled


def brute(n1, n0, p1, p0, c, rule="strict", pooled=False):
    """Full double sum over the product binomial pmf; posting decided with exact rational arithmetic."""
    cf = Fraction(c).limit_denominator(10**9)
    x1 = np.arange(n1 + 1)[:, None]
    x0 = np.arange(n0 + 1)[None, :]
    f = binom.pmf(x1, n1, p1) * binom.pmf(x0, n0, p0)
    cross = (
        (lambda x, n: Fraction(int(x), int(n)) > cf)
        if rule == "strict"
        else (lambda x, n: Fraction(int(x), int(n)) >= cf)
    )
    if pooled:
        post = np.array([[cross(a + b, n1 + n0) for b in range(n0 + 1)] for a in range(n1 + 1)])
    else:
        c1 = np.array([cross(a, n1) for a in range(n1 + 1)])[:, None]
        c0 = np.array([cross(b, n0) for b in range(n0 + 1)])[None, :]
        post = c1 | c0
    P = f[post].sum()
    e1 = (f * post * x1 / n1).sum() / P
    e0 = (f * post * x0 / n0).sum() / P
    mh_num = (f * post * x1 * n0 / (n1 + n0)).sum()
    mh_den = (f * post * x0 * n1 / (n1 + n0)).sum()
    return P, e1, e0, mh_num / mh_den


@pytest.mark.parametrize("c", [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.3, 1 / 3, 1.0])
@pytest.mark.parametrize("rule", ["strict", "weak"])
def test_kcross_matches_rational(c, rule):
    cf = Fraction(c).limit_denominator(10**9) if c != 1 / 3 else Fraction(1, 3)
    for n in list(range(1, 1001)) + [1999, 2000, 2001, 10000]:
        if rule == "strict":
            ref = next((x for x in range(n + 1) if Fraction(x, n) > cf), n + 1)
        else:
            ref = next((x for x in range(n + 1) if Fraction(x, n) >= cf), n + 1)
        assert kcross(n, c, rule) == ref, (n, c, rule)


def test_kcross_edge_cases():
    assert kcross(100, 0.05) == 6
    assert kcross(100, 0.05, "weak") == 5
    assert kcross(99, 0.05) == 5
    assert kcross(100, 0.0) == 1
    assert kcross(100, 0.0, "weak") == 0
    assert kcross(10, 1.0) == 11
    assert kcross(10, 1.0, "weak") == 10
    np.testing.assert_array_equal(kcross(np.array([20, 99, 100]), 0.05), [2, 5, 6])
    with pytest.raises(ValueError):
        kcross(100, 5)
    with pytest.raises(ValueError):
        kcross(100, 0.05, "Strict")


def test_posting_rules():
    assert posted_anyarm(6, 100, 0, 50, 0.05) and not posted_anyarm(5, 100, 2, 50, 0.05)
    assert posted_anyarm(0, 100, 3, 50, 0.05)
    assert posted_pooled(5, 100, 3, 50, 0.05) and not posted_pooled(5, 100, 2, 50, 0.05)
    np.testing.assert_array_equal(posted_anyarm([6, 5], [100, 100], [0, 2], [50, 50], 0.05), [True, False])
    np.testing.assert_array_equal(posted_pooled([5, 5], [100, 100], [3, 2], [50, 50], 0.05), [True, False])


def _designs(m=20, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(m):
        n1, n0 = int(rng.integers(5, 400)), int(rng.integers(5, 400))
        p0 = float(np.exp(rng.uniform(np.log(0.002), np.log(0.2))))
        p1 = float(min(p0 * np.exp(rng.normal(0, 0.5)), 0.6))
        c = float(rng.choice([0.0, 0.02, 0.05, 0.1, round(rng.uniform(0.005, 0.15), 3)]))
        rule = str(rng.choice(["strict", "weak"]))
        out.append((n1, n0, p1, p0, c, rule))
    return out


@pytest.mark.parametrize("design", _designs())
def test_exact_conditional_vs_double_sum(design):
    n1, n0, p1, p0, c, rule = design
    P, e1, e0, mh_lim = brute(n1, n0, p1, p0, c, rule)
    r = exact_conditional(n1, n0, p1, p0, c, rule)
    assert r["p_post"] == pytest.approx(P, rel=1e-9, abs=1e-15)
    assert r["e1"] == pytest.approx(e1, rel=1e-8)
    assert r["e0"] == pytest.approx(e0, rel=1e-8)
    assert r["rr_cond"] == pytest.approx(mh_lim, rel=1e-8)
    kap = r["kappa"]
    D1 = (r["e1"] - p1) / kap if kap > 0 else 0.0
    D0 = (r["e0"] - p0) / kap if kap > 0 else 0.0
    assert D1 >= -1e-15 and D0 >= -1e-15
    assert r["rr_cond"] / r["rr_true"] == pytest.approx((1 + kap * D1 / p1) / (1 + kap * D0 / p0), rel=1e-10)


def test_exact_conditional_vs_monte_carlo():
    rng = np.random.default_rng(7)
    for n1, n0, p1, p0, c in [(200, 100, 0.03, 0.03, 0.05), (300, 100, 0.045, 0.03, 0.05), (99, 100, 0.02, 0.02, 0.05)]:
        M = 2_000_000
        x1 = rng.binomial(n1, p1, M)
        x0 = rng.binomial(n0, p0, M)
        post = posted_anyarm(x1, n1, x0, n0, c)
        r = exact_conditional(n1, n0, p1, p0, c)
        assert post.mean() == pytest.approx(r["p_post"], abs=4 * np.sqrt(r["p_post"] / M))
        for x, n, e in [(x1, n1, r["e1"]), (x0, n0, r["e0"])]:
            ph = x[post] / n
            assert abs(ph.mean() - e) < 4.5 * ph.std() / np.sqrt(post.sum())


def test_exact_conditional_rare_posting_stable():
    """Very rare posting: the closed form stays accurate."""
    r = exact_conditional(80000, 40000, 0.03, 0.03, 0.05)
    assert 0 < r["p_post"] < 1e-50
    assert r["rr_cond"] == pytest.approx(0.6, abs=0.005)  # limit p1/c
    r = exact_conditional(90000, 30000, 0.045, 0.03, 0.05)
    assert r["rr_cond"] == pytest.approx(0.05 / 0.03, abs=0.01)  # limit c/p0


def test_symmetric_arms_unbiased_and_one_patient_breaks_it():
    assert exact_conditional(100, 100, 0.02, 0.02, 0.05)["rr_cond"] == pytest.approx(1.0, abs=1e-12)
    assert exact_conditional(100, 99, 0.02, 0.02, 0.05)["rr_cond"] == pytest.approx(0.65, abs=0.01)


def test_threshold_zero_strict_is_unbiased_for_any_rr():
    """With c = 0 and the strict rule, posting means at least one event; the ratio is unchanged."""
    for n1, n0, p1, p0 in [(200, 100, 0.01, 0.02), (300, 100, 0.03, 0.01), (57, 13, 0.1, 0.05)]:
        r = exact_conditional(n1, n0, p1, p0, 0.0)
        assert r["rr_cond"] == pytest.approx(p1 / p0, rel=1e-10)


@pytest.mark.parametrize("design", _designs(12, seed=3))
def test_pooled_rule_exactly_unbiased_at_rr1(design):
    n1, n0, _, p0, c, rule = design
    n1, n0 = min(n1, 150), min(n0, 150)
    P, e1, e0, mh_lim = brute(n1, n0, p0, p0, c, rule, pooled=True)
    if P > 1e-12:
        assert e1 / e0 == pytest.approx(1.0, abs=1e-9)
        assert mh_lim == pytest.approx(1.0, abs=1e-9)


def test_mh_single_table_reduces_to_katz():
    a, n1, b, n0 = 12, 200, 5, 100
    est, se = mh_rr([a], [n1], [b], [n0])
    assert est == pytest.approx(np.log((a / n1) / (b / n0)))
    assert se**2 == pytest.approx(1 / a - 1 / n1 + 1 / b - 1 / n0)


def test_mh_greenland_robins_formula_by_hand():
    x1, n1, x0, n0 = np.array([3, 10, 0]), np.array([40, 120, 60]), np.array([1, 8, 2]), np.array([20, 60, 30])
    N = n1 + n0
    R = sum(x1 * n0 / N)
    S = sum(x0 * n1 / N)
    P = sum(n1 * n0 * (x1 + x0) / N**2 - x1 * x0 / N)
    est, se = mh_rr(x1, n1, x0, n0)
    assert est == pytest.approx(np.log(R / S)) and se == pytest.approx(np.sqrt(P / (R * S)))


def test_mh_variance_matches_simulation():
    rng = np.random.default_rng(11)
    K, M = 15, 20000
    n1 = rng.integers(30, 200, K)
    n0 = rng.integers(20, 120, K)
    p0 = rng.uniform(0.03, 0.15, K)
    p1 = np.minimum(1.4 * p0, 0.9)
    ests, ses = [], []
    for _ in range(M):
        e, s = mh_rr(rng.binomial(n1, p1), n1, rng.binomial(n0, p0), n0)
        ests.append(e)
        ses.append(s)
    ests = np.array(ests)
    assert np.mean(ses) == pytest.approx(ests.std(), rel=0.04)
    assert ests.mean() == pytest.approx(np.log(1.4), abs=0.01)


def test_mh_returns_nan_without_events():
    assert np.isnan(mh_rr([0, 0], [10, 10], [1, 2], [10, 10])[0])


def test_iv_known_value():
    est, se = iv_rr([0], [100], [3], [50])
    a, b, m1, m0 = 0.5, 3.5, 101, 51
    assert est == pytest.approx(np.log(a / m1) - np.log(b / m0))
    assert se == pytest.approx(np.sqrt(1 / a - 1 / m1 + 1 / b - 1 / m0))


def test_iv_dersimonian_laird_by_hand():
    x1, n1, x0, n0 = (
        np.array([3, 10, 0, 7]),
        np.array([40, 120, 60, 90]),
        np.array([1, 8, 2, 1]),
        np.array([20, 60, 30, 45]),
    )
    z = (x1 == 0) | (x0 == 0)
    a, b = x1 + 0.5 * z, x0 + 0.5 * z
    m1, m0 = n1 + z, n0 + z
    y = np.log(a / m1) - np.log(b / m0)
    v = 1 / a - 1 / m1 + 1 / b - 1 / m0
    w = 1 / v
    fe = (w * y).sum() / w.sum()
    tau2 = max(0.0, ((w * (y - fe) ** 2).sum() - 3) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    est, se = iv_rr(x1, n1, x0, n0, method="dl")
    assert est == pytest.approx((ws * y).sum() / ws.sum())
    assert se == pytest.approx(1 / np.sqrt(ws.sum()))
    with pytest.raises(ValueError):
        iv_rr(x1, n1, x0, n0, method="reml")


def test_bias_check_single_design_equals_exact_conditional():
    for n1, n0, p0, rr in [(200, 100, 0.03, 1.0), (150, 150, 0.03, 1.5), (40, 120, 0.01, 0.67)]:
        r = bias_check([n1], [n0], p0, rr=rr)
        e = exact_conditional(n1, n0, rr * p0, p0, 0.05)
        assert r["rr_posted"][0] == pytest.approx(e["rr_cond"] / 1.0, rel=1e-12)
        assert r["rr_complete"][0] == pytest.approx(rr, rel=1e-12)
        assert r["p_post"][0] == pytest.approx(e["p_post"], rel=1e-12)


@pytest.mark.parametrize("rule", ["strict", "weak"])
def test_bias_check_design_mix_vs_double_sum(rule):
    designs = [(200, 100), (50, 50), (120, 40), (33, 61)]
    for p0, rr in [(0.02, 1.0), (0.03, 1.5), (0.05, 0.67)]:
        num = den = pp = 0.0
        for n1, n0 in designs:
            P, e1, e0, _ = brute(n1, n0, rr * p0, p0, 0.05, rule)
            w = n1 * n0 / (n1 + n0)
            num += w * P * e1
            den += w * P * e0
            pp += P / len(designs)
        r = bias_check([d[0] for d in designs], [d[1] for d in designs], p0, rr=rr, rule=rule)
        assert r["rr_posted"][0] == pytest.approx(num / den, rel=1e-9)
        assert r["p_post"][0] == pytest.approx(pp, rel=1e-9)


def test_bias_check_between_trial_variation_vs_monte_carlo():
    rng = np.random.default_rng(11)
    n1 = np.tile([200, 50, 120], 100000)
    n0 = np.tile([100, 50, 40], 100000)
    q0 = 1 / (1 + np.exp(-(np.log(0.03 / 0.97) + 0.5 * rng.standard_normal(n1.size))))
    q1 = np.minimum(q0 * np.exp(np.log(1.5) + 0.3 * rng.standard_normal(n1.size)), 1)
    x1, x0 = rng.binomial(n1, q1), rng.binomial(n0, q0)
    post = posted_anyarm(x1, n1, x0, n0, 0.05)
    N = n1 + n0
    mc = (x1 * n0 / N * post).sum() / (x0 * n1 / N * post).sum()
    mc_all = (x1 * n0 / N).sum() / (x0 * n1 / N).sum()
    r = bias_check([200, 50, 120], [100, 50, 40], 0.03, rr=1.5, sd_logit=0.5, tau=0.3)
    assert r["rr_posted"][0] == pytest.approx(mc, rel=0.01)
    assert r["rr_complete"][0] == pytest.approx(mc_all, rel=0.01)
    assert r["p_post"][0] == pytest.approx(post.mean(), abs=0.005)
    assert r["rr_complete"][0] == pytest.approx(1.5 * np.exp(0.3**2 / 2), rel=1e-6)


def test_bias_check_equal_arms_unbiased_and_rare_posting_stable():
    r = bias_check([100, 250, 400], [100, 250, 400], [0.005, 0.02, 0.04])
    assert np.allclose(r["distortion"], 1, atol=1e-12)
    big = bias_check([20000, 300], [10000, 150], [0.001, 0.01])
    assert np.all(np.isfinite(big["rr_posted"])) and np.all(big["rr_posted"] > 0)


def test_bias_check_input_checks():
    with pytest.raises(ValueError):
        bias_check([10, 20], [10], 0.02)
    with pytest.raises(ValueError):
        bias_check([10], [0], 0.02)
    with pytest.raises(ValueError):
        bias_check([10], [10], 0.0)
    with pytest.raises(ValueError):
        bias_check([10], [10], 0.02, nodes=0, sd_logit=0.5)
