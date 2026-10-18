"""Numerical checks of the results in Supplementary Material S1 (exact sums, Monte Carlo, limits)."""

from fractions import Fraction
from math import ceil, comb, floor

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln, logsumexp

rng = np.random.default_rng(20261001)
np.set_printoptions(precision=6, suppress=False)


# ---------------------------------------------------------------- cut-offs
def kcut(n, c, rule):
    """Smallest count that crosses. c given as str/Fraction for exact arithmetic."""
    c = Fraction(c)
    if rule == "strict":
        k = floor(c * n) + 1
    else:
        k = ceil(c * n)
    return min(k, n + 1)


# ---------------------------------------------------------------- closed form (Proposition S1.2)
def closed_form(n1, n0, p1, p0, k1, k0):
    """Returns E1, E0 (conditional mean proportions given A), P(A), kappa, D1, D0."""
    S = {}
    F = {}
    f = {}
    for j, (n, p, k) in {1: (n1, p1, k1), 0: (n0, p0, k0)}.items():
        S[j] = stats.binom.sf(k - 1, n, p) if k >= 1 else 1.0
        F[j] = stats.binom.cdf(k - 1, n, p) if k >= 1 else 0.0
        f[j] = stats.binom.pmf(k, n, p) if 0 <= k <= n else 0.0
    PA = S[1] + F[1] * S[0]  # = 1 - F1 F0 without cancellation
    # D_j = (k_j/n_j)(1-p_j) P(X_j=k_j)/F_j ; kappa D_j = F_{j'} (k_j/n_j)(1-p_j) f_j / P(A)
    kD1 = F[0] * (k1 / n1) * (1 - p1) * f[1] / PA
    kD0 = F[1] * (k0 / n0) * (1 - p0) * f[0] / PA
    E1, E0 = p1 + kD1, p0 + kD0
    D1 = (k1 / n1) * (1 - p1) * f[1] / F[1] if F[1] > 0 else np.nan
    D0 = (k0 / n0) * (1 - p0) * f[0] / F[0] if F[0] > 0 else np.nan
    kappa = F[1] * F[0] / PA
    return dict(E1=E1, E0=E0, PA=PA, RA=E1 / E0, kappa=kappa, D1=D1, D0=D0, S1=S[1], S0=S[0])


def closed_form_literal(n1, n0, p1, p0, k1, k0):
    """Literal form p_j + kappa (p_j - ell_j), ell_j = E[X_j/n_j | X_j < k_j] (for cross-check)."""
    F = {}
    L = {}
    for j, (n, p, k) in {1: (n1, p1, k1), 0: (n0, p0, k0)}.items():
        x = np.arange(0, k)
        pm = stats.binom.pmf(x, n, p)
        F[j] = pm.sum()
        L[j] = (x * pm).sum() / n / F[j] if F[j] > 0 else 0.0
    kappa = F[1] * F[0] / (1 - F[1] * F[0])
    E1 = p1 + kappa * (p1 - L[1])
    E0 = p0 + kappa * (p0 - L[0])
    return E1, E0


def brute(n1, n0, p1, p0, k1, k0):
    x1 = np.arange(n1 + 1)
    x0 = np.arange(n0 + 1)
    b1 = stats.binom.pmf(x1, n1, p1)
    b0 = stats.binom.pmf(x0, n0, p0)
    J = np.outer(b1, b0)
    A = (x1[:, None] >= k1) | (x0[None, :] >= k0)
    PA = J[A].sum()
    E1 = (J * x1[:, None] / n1)[A].sum() / PA
    E0 = (J * x0[None, :] / n0)[A].sum() / PA
    return dict(E1=E1, E0=E0, PA=PA, RA=E1 / E0)


print("=" * 78)
print("PART 1: closed form vs brute-force double sum, random designs")
print("=" * 78)
cs = ["0.01", "0.02", "0.03", "0.05"]
err = dict(E1=0.0, E0=0.0, RA_rel=0.0, PA_rel=0.0, literal=0.0)
nD = 0
nneg = 0
sign_ok = 0
sign_tot = 0
mediant_ok = 0
minPA = 1.0
for it in range(2500):
    n1, n0 = rng.integers(5, 401, size=2)
    p0 = float(np.exp(rng.uniform(np.log(0.001), np.log(0.2))))
    RR = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    p1 = min(RR * p0, 0.999)
    RR = p1 / p0
    c = cs[rng.integers(4)]
    for rule in ("strict", "weak"):
        k1, k0 = kcut(n1, c, rule), kcut(n0, c, rule)
        cf = closed_form(n1, n0, p1, p0, k1, k0)
        if cf["PA"] <= 0:
            continue
        bf = brute(n1, n0, p1, p0, k1, k0)
        nD += 1
        minPA = min(minPA, cf["PA"])
        err["E1"] = max(err["E1"], abs(cf["E1"] - bf["E1"]))
        err["E0"] = max(err["E0"], abs(cf["E0"] - bf["E0"]))
        err["RA_rel"] = max(err["RA_rel"], abs(cf["RA"] / bf["RA"] - 1))
        err["PA_rel"] = max(err["PA_rel"], abs(cf["PA"] / bf["PA"] - 1))
        if cf["PA"] > 1e-8:  # literal form, checked where P(A) is not negligible
            e1, e0 = closed_form_literal(n1, n0, p1, p0, k1, k0)
            err["literal"] = max(err["literal"], abs(e1 - bf["E1"]), abs(e0 - bf["E0"]))
        for D in (cf["D1"], cf["D0"]):
            if np.isfinite(D) and D < 0:
                nneg += 1
        if 0 < cf["kappa"] and np.isfinite(cf["D1"]) and np.isfinite(cf["D0"]):
            r1, r0 = cf["D1"] / p1, cf["D0"] / p0
            if abs(r1 - r0) > 1e-9 * max(r1, r0) and abs(bf["RA"] / RR - 1) > 1e-10:
                sign_tot += 1
                sign_ok += np.sign(bf["RA"] - RR) == np.sign(r1 - r0)
            lo, hi = sorted([RR, cf["D1"] / cf["D0"]]) if cf["D0"] > 0 else (RR, np.inf)
            mediant_ok += lo * (1 - 1e-10) <= bf["RA"] <= hi * (1 + 1e-10)
print(f"designs evaluated (2 rules each): {nD}; smallest P(A) = {minPA:.3e}")
print(f"max |E1 closed - brute|            = {err['E1']:.3e}")
print(f"max |E0 closed - brute|            = {err['E0']:.3e}")
print(f"max |R_A closed / brute - 1|       = {err['RA_rel']:.3e}")
print(f"max |P(A) closed / brute - 1|      = {err['PA_rel']:.3e}")
print(f"max abs err, literal p+kappa(p-ell) form (P(A)>1e-8) = {err['literal']:.3e}")
print(f"negative D_j encountered           = {nneg}")
print(f"sign rule sign(R_A-RR)=sign(r1-r0) : {sign_ok}/{sign_tot}")
print(f"R_A between RR and D1/D0 (mediant) : {mediant_ok}/{nD}")
assert max(err["E1"], err["E0"]) < 1e-12 and nneg == 0 and sign_ok == sign_tot and mediant_ok == nD

# ---------------------------------------------------------------- specific numbers
print()
print("=" * 78)
print("SPECIFIC DESIGNS (c = 0.05)")
print("=" * 78)


def report(n1, n0, p1, p0, c="0.05"):
    for rule in ("strict", "weak"):
        k1, k0 = kcut(n1, c, rule), kcut(n0, c, rule)
        cf = closed_form(n1, n0, p1, p0, k1, k0)
        bf = brute(n1, n0, p1, p0, k1, k0)
        print(
            f"n1={n1:4d} n0={n0:4d} p1={p1:.4f} p0={p0:.4f} RR={p1 / p0:.3f} {rule:6s} "
            f"k=({k1},{k0}) P(A)={cf['PA']:.4f} E1={cf['E1']:.5f} E0={cf['E0']:.5f} "
            f"kappa={cf['kappa']:.4f} R_A={cf['RA']:.4f} (brute {bf['RA']:.4f})"
        )


report(200, 100, 0.03, 0.03)
report(200, 100, 0.04, 0.02)
report(225, 75, 0.03, 0.03)
report(150, 150, 0.045, 0.03)
report(200, 100, 0.045, 0.03)

# ---------------------------------------------------------------- Part 2: Monte Carlo MH
print()
print("=" * 78)
print("PART 2: Monte Carlo, Mantel-Haenszel RR over posted rows -> R_A")
print("=" * 78)
for n1, n0, p1, p0 in [(200, 100, 0.03, 0.03), (200, 100, 0.045, 0.03), (150, 150, 0.045, 0.03)]:
    k1, k0 = kcut(n1, "0.05", "strict"), kcut(n0, "0.05", "strict")
    RA = closed_form(n1, n0, p1, p0, k1, k0)["RA"]
    M = 2_000_000
    X1 = rng.binomial(n1, p1, M)
    X0 = rng.binomial(n0, p0, M)
    A = (X1 >= k1) | (X0 >= k0)
    N = n1 + n0
    mh = (X1[A] * n0 / N).sum() / (X0[A] * n1 / N).sum()
    # batch standard error (20 batches)
    idx = np.array_split(np.flatnonzero(A), 20)
    b = [(X1[i] * n0).sum() / (X0[i] * n1).sum() for i in idx]
    print(
        f"n1={n1} n0={n0} p1={p1} p0={p0}: posted={A.sum()}  MH={mh:.4f} "
        f"(batch SE {np.std(b, ddof=1) / np.sqrt(20):.4f})  R_A={RA:.4f}"
    )

# ---------------------------------------------------------------- Part 3: equal arms / counterexample
print()
print("=" * 78)
print("PART 3: equal arms, RR=1 -> R_A=1 exactly; unequal-by-one counterexample")
print("=" * 78)


def exact_RA(n1, n0, p, k1, k0):
    p = Fraction(p)

    def b(n, x):
        return comb(n, x) * p**x * (1 - p) ** (n - x)

    num1 = den = num0 = Fraction(0)
    for x1 in range(n1 + 1):
        for x0 in range(n0 + 1):
            if x1 >= k1 or x0 >= k0:
                w = b(n1, x1) * b(n0, x0)
                den += w
                num1 += w * x1
                num0 += w * x0
    return (num1 / n1) / (num0 / n0)


for n, p, c, rule in [(12, "1/10", "1/5", "strict"), (15, "3/50", "1/10", "weak"), (9, "1/7", "1/3", "strict")]:
    k = kcut(n, c, rule)
    print(f"exact rational: n1=n0={n}, p={p}, c={c}, {rule}: R_A = {exact_RA(n, n, p, k, k)}")
maxdev = 0.0
for it in range(500):
    n = int(rng.integers(5, 401))
    p = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2))))
    c = cs[rng.integers(4)]
    rule = ["strict", "weak"][rng.integers(2)]
    k = kcut(n, c, rule)
    bf = brute(n, n, p, p, k, k)
    if bf["PA"] > 0:
        maxdev = max(maxdev, abs(bf["RA"] - 1))
print(f"500 random equal-arm designs at RR=1: max |R_A - 1| = {maxdev:.3e}")
assert maxdev < 1e-12
for rule in ("strict", "weak"):
    k1, k0 = kcut(100, "0.05", rule), kcut(99, "0.05", rule)
    cf = closed_form(100, 99, 0.02, 0.02, k1, k0)
    bf = brute(100, 99, 0.02, 0.02, k1, k0)
    print(
        f"n1=100 n0=99 p=0.02 c=0.05 {rule}: k1={k1} ({k1}/100={k1 / 100:.4f}) k0={k0} ({k0}/99={k0 / 99:.4f}) "
        f"S1={cf['S1']:.5f} S0={cf['S0']:.5f} D1={cf['D1']:.5f} D0={cf['D0']:.5f} "
        f"P(A)={cf['PA']:.5f} R_A={cf['RA']:.5f} (brute {bf['RA']:.5f})"
    )
print(
    "exact rational R_A (n1=100, n0=99, p=1/50, c=1/20, strict) = %.10f"
    % float(exact_RA(100, 99, "1/50", kcut(100, "0.05", "strict"), kcut(99, "0.05", "strict")))
)

# ---------------------------------------------------------------- Part 4: pooled-arm rule
print()
print("=" * 78)
print("PART 4: pooled-arm rule B = {X1+X0 >= k(N)}")
print("=" * 78)


def pooled(n1, n0, p1, p0, c, rule, exact=False):
    N = n1 + n0
    kT = kcut(N, c, rule)
    if exact:
        p1, p0 = Fraction(p1), Fraction(p0)

        def b(n, x, p):
            return comb(n, x) * p**x * (1 - p) ** (n - x)

        ee = Fraction(0)
        s1 = s0 = Fraction(0)
        for x1 in range(n1 + 1):
            for x0 in range(n0 + 1):
                if x1 + x0 >= kT:
                    w = b(n1, x1, p1) * b(n0, x0, p0)
                    ee += w * (x1 * n0 - x0 * n1)
                    s1 += w * x1
                    s0 += w * x0
        return ee, (s1 * n0) / (s0 * n1)
    x1 = np.arange(n1 + 1)
    x0 = np.arange(n0 + 1)
    J = np.outer(stats.binom.pmf(x1, n1, p1), stats.binom.pmf(x0, n0, p0))
    B = (x1[:, None] + x0[None, :]) >= kT
    ee = (J * (x1[:, None] * n0 - x0[None, :] * n1))[B].sum()
    scale = (J * (x1[:, None] * n0 + x0[None, :] * n1))[B].sum()
    RB = (J * x1[:, None])[B].sum() * n0 / ((J * x0[None, :])[B].sum() * n1)
    return ee / scale, RB


for n1, n0, p, c in [(7, 4, "1/10", "1/5"), (13, 5, "1/20", "1/10"), (10, 3, "1/4", "1/3")]:
    ee, RB = pooled(n1, n0, p, p, c, "strict", exact=True)
    print(f"exact rational n1={n1} n0={n0} p={p} c={c}: E[(X1 n0 - X0 n1)1_B] = {ee}, R_B = {RB}")
mx = 0.0
for it in range(500):
    n1, n0 = (int(v) for v in rng.integers(5, 401, size=2))
    p = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2))))
    c = cs[rng.integers(4)]
    rule = ["strict", "weak"][rng.integers(2)]
    ee, RB = pooled(n1, n0, p, p, c, rule)
    mx = max(mx, abs(RB - 1))
print(f"500 random unequal designs at RR=1: max |R_B - 1| = {mx:.3e}")
assert mx < 1e-12
for n1, n0, p0, RR in [(200, 100, 0.03, 1.5), (100, 100, 0.02, 2.0), (200, 100, 0.02, 3.0), (100, 300, 0.01, 0.5)]:
    ee, RB = pooled(n1, n0, RR * p0, p0, "0.05", "strict")
    k1, k0 = kcut(n1, "0.05", "strict"), kcut(n0, "0.05", "strict")
    RA = closed_form(n1, n0, RR * p0, p0, k1, k0)["RA"]
    print(f"n1={n1} n0={n0} p0={p0} RR={RR}: pooled R_B={RB:.4f}  (any-arm R_A={RA:.4f})")

# ---------------------------------------------------------------- Part 5: large n
print()
print("=" * 78)
print("PART 5: large-n behaviour, n_j = alpha_j n")
print("=" * 78)


def log_tail(n, p, k):
    """log P(X>=k), log P(X=k), log P(X<k) for X~Bin(n,p), stable for any n."""
    if k > n:
        return -np.inf, -np.inf, 0.0
    x = np.arange(k, min(n, k + 20000) + 1)
    lp = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + x * np.log(p) + (n - x) * np.log1p(-p)
    lS = logsumexp(lp)
    lF = np.log(-np.expm1(lS)) if lS < -1e-300 else -np.inf
    return lS, lp[0], lF


def RA_large(n1, n0, p1, p0, c="0.05", rule="strict"):
    k1, k0 = kcut(n1, c, rule), kcut(n0, c, rule)
    lS1, lf1, lF1 = log_tail(n1, p1, k1)
    lS0, lf0, lF0 = log_tail(n0, p0, k0)
    lPA = np.logaddexp(lS1, lF1 + lS0)
    E1 = p1 + (k1 / n1) * (1 - p1) * np.exp(lF0 + lf1 - lPA)
    E0 = p0 + (k0 / n0) * (1 - p0) * np.exp(lF1 + lf0 - lPA)
    kappa = (lF1 + lF0 - lPA) / np.log(10)  # log10 kappa
    return E1 / E0, kappa, lPA, E1, E0


def KL(a, b):
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


c = 0.05
cases = [
    ("2:1, p1=p0=0.03", 2 / 3, 1 / 3, 0.03, 0.03),
    ("1:2, p1=p0=0.03", 1 / 3, 2 / 3, 0.03, 0.03),
    ("1:1, p1=0.04,p0=0.02", 1 / 2, 1 / 2, 0.04, 0.02),
    ("1:1, p1=0.02,p0=0.04", 1 / 2, 1 / 2, 0.02, 0.04),
    ("1:1, p1=0.06,p0=0.04 (one arm above c)", 1 / 2, 1 / 2, 0.06, 0.04),
    ("1:1, p1=0.08,p0=0.06 (both above c)", 1 / 2, 1 / 2, 0.08, 0.06),
]
for name, al1, al0, p1, p0 in cases:
    r1, r0 = al1 * KL(c, p1), al0 * KL(c, p0)
    if p1 < c and p0 < c:
        lim = p1 / c if r0 < r1 else c / p0
        tag = "placebo dominates -> p1/c" if r0 < r1 else "drug dominates -> c/p0"
    else:
        lim = p1 / p0
        tag = "an arm above c -> RR"
    print(f"{name}: rates alpha1*KL1={r1:.5f}, alpha0*KL0={r0:.5f}; {tag} = {lim:.4f}; RR={p1 / p0:.4f}")
    for n in [300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]:
        n1, n0 = int(round(al1 * n)), int(round(al0 * n))
        RA, kap, lPA, E1, E0 = RA_large(n1, n0, p1, p0)
        print(
            f"   n={n:8d} (n1={n1},n0={n0}) log P(A)={lPA:10.2f} log10 kappa={kap:8.2f} "
            f"E1={E1:.5f} E0={E0:.5f} R_A={RA:.5f}"
        )
print("p1 = p0 = c (RR=1), 2:1: sqrt(n)(R_A - 1) vs predicted (1/3) lambda(0) sqrt((1-c)/c)(alpha1^-1/2 - alpha0^-1/2)")
al1, al0 = 2 / 3, 1 / 3
pred = (1 / 3) * np.sqrt(2 / np.pi) * np.sqrt((1 - c) / c) * (al1**-0.5 - al0**-0.5)
for n in [300, 3000, 30000, 300000, 3000000]:
    n1, n0 = int(round(al1 * n)), int(round(al0 * n))
    RA, kap, lPA, E1, E0 = RA_large(n1, n0, c, c)
    print(
        f"   n={n:8d}: kappa={10**kap:.4f} R_A-1={RA - 1:+.5f} "
        f"sqrt(n)(R_A-1)={np.sqrt(n) * (RA - 1):+.4f}  predicted {pred:+.4f}"
    )

print("strict rule at c = 0 (k_j = 1, row posted iff T >= 1): R_A should equal RR exactly")
mx = 0.0
for it in range(300):
    n1, n0 = (int(v) for v in rng.integers(5, 401, size=2))
    p0 = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2))))
    RR = float(np.exp(rng.uniform(np.log(0.3), np.log(3))))
    p1 = min(RR * p0, 0.999)
    bf = brute(n1, n0, p1, p0, kcut(n1, "0", "strict"), kcut(n0, "0", "strict"))
    mx = max(mx, abs(bf["RA"] / (p1 / p0) - 1))
print(f"   300 random designs: max |R_A/RR - 1| = {mx:.3e}")
assert mx < 1e-12
print("near-equal arms at large n (n1 = n0 + 1, p = 0.03, c = 0.05 strict):")
for n0 in [1999, 2000, 19999, 20000]:
    n1 = n0 + 1
    k1, k0 = kcut(n1, "0.05", "strict"), kcut(n0, "0.05", "strict")
    RA = RA_large(n1, n0, 0.03, 0.03)[0]
    print(f"   n1={n1} n0={n0} k=({k1},{k0}) R_A={RA:.4f}")

# ---------------------------------------------------------------- Part 6: normal approximation
print()
print("=" * 78)
print("PART 6: normal approximation D_j ~ s_j lambda(a_j)")
print("=" * 78)


def lam(a):
    return np.exp(stats.norm.logpdf(a) - stats.norm.logcdf(a))


for p in [0.02, 0.03, 0.04]:
    for n in [50, 100, 200, 400, 1600, 6400]:
        k = kcut(n, "0.05", "strict")
        cf = closed_form(n, n, p, p, k, k)
        s = np.sqrt(p * (1 - p) / n)
        a_cc = ((k - 0.5) / n - p) / s
        a_raw = (0.05 - p) / s
        print(
            f"p={p} n={n:5d} k={k:4d}: D exact={cf['D1']:.6f}  s*lam(a_cc)={s * lam(a_cc):.6f} "
            f"(rel {s * lam(a_cc) / cf['D1'] - 1:+.3f})  s*lam(a_raw)={s * lam(a_raw):.6f}"
        )
a = np.linspace(-30, 30, 600001)
g = 1 + a**2 + a * lam(a)
print(f"min over a in [-30,30] of 1 + a^2 + a*lambda(a) = {g.min():.3e} (must be > 0 => dD/ds > 0)")
print("local regime (a fixed, p = c - a s, n growing): relative error of s*lambda(a_cc)")
for a0 in [-1.0, 0.0, 1.0, 2.0]:
    row = []
    for n in [100, 400, 1600, 6400, 25600, 102400]:
        p = optimize.brentq(lambda q: 0.05 - q - a0 * np.sqrt(q * (1 - q) / n), 1e-6, 0.5)
        k = kcut(n, "0.05", "strict")
        D = closed_form(n, n, p, p, k, k)["D1"]
        s = np.sqrt(p * (1 - p) / n)
        acc = ((k - 0.5) / n - p) / s
        row.append(f"n={n}: {s * lam(acc) / D - 1:+.4f}")
    print(f"   a={a0:+.0f}: " + "  ".join(row))
print("exact D_j along n multiples of 20 (c n integer, same lattice position), strict, c=0.05:")
for p in [0.01, 0.02, 0.03, 0.04]:
    ns = np.arange(20, 2001, 20)
    Ds = np.array(
        [
            closed_form(int(n), int(n), p, p, kcut(int(n), "0.05", "strict"), kcut(int(n), "0.05", "strict"))["D1"]
            for n in ns
        ]
    )
    print(
        f"   p={p}: D strictly decreasing in n: {bool(np.all(np.diff(Ds) < 0))}; "
        f"D(20)={Ds[0]:.5f}, D(100)={Ds[4]:.5f}, D(400)={Ds[19]:.6f}"
    )
print("exact D_j for consecutive n (lattice sawtooth), p=0.03, strict, c=0.05:")
print(
    "   "
    + "  ".join(
        f"n={n}:{closed_form(n, n, 0.03, 0.03, kcut(n, '0.05', 'strict'), kcut(n, '0.05', 'strict'))['D1']:.5f}"
        for n in [95, 99, 100, 101, 105, 119, 120]
    )
)
