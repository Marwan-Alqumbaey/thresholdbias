"""Simulation, part A: common pooling practices under the posting rules.

Meta-analyses of K two-arm trials; placebo risk logit-normal around a median
p_med with SD 0.5; drug risk = RR x placebo risk. Methods:
  M0 Mantel-Haenszel (MH) on all trials (reference)
  M1 MH on rows posted under the any-arm rule
  M2 DerSimonian-Laird on posted rows, 0.5 continuity correction
  M3 as M2 with unposted trials entered as zero events
  M4 MH under a one-sided rule (drug arm >= c and drug rate > placebo rate)
  M7 MH under the pooled-arm rule
1,000 replicates per scenario. Output: results/simulation/naive_methods.csv
"""

import itertools
import time

import numpy as np
import pandas as pd
from common_sim import OUT, SD_LOGIT
from thresholdbias import kcross

REPS = 1000
KS = [10, 30, 100]
RATIOS = [(1, 1), (3, 2), (2, 1), (3, 1)]
NS = [150, 300, 600]
PMED = [0.01, 0.02, 0.03, 0.04, 0.06]
RRS = [0.67, 1.0, 1.5, 2.0]
CS = [0.0, 0.02, 0.05]


def mh_masked(x1, n1, x0, n0, m):
    """Row-wise (replicate-wise) MH log RR and Greenland-Robins SE over the rows selected by m."""
    N = n1 + n0
    R = np.where(m, x1 * n0 / N, 0).sum(1)
    S = np.where(m, x0 * n1 / N, 0).sum(1)
    P = np.where(m, (n1 * n0 * (x1 + x0) - x1 * x0 * N) / N**2, 0).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = np.log(R / S)
        se = np.sqrt(P / (R * S))
    ok = (R > 0) & (S > 0)
    return np.where(ok, est, np.nan), np.where(ok, se, np.nan)


def dl_masked(x1, n1, x0, n0, m, cc=0.5):
    """Replicate-wise DerSimonian-Laird log RR over the rows selected by m."""
    z = (x1 == 0) | (x0 == 0)
    a, b = x1 + cc * z, x0 + cc * z
    m1, m0 = n1 + 2 * cc * z, n0 + 2 * cc * z
    y = np.log(a / m1) - np.log(b / m0)
    v = 1 / a - 1 / m1 + 1 / b - 1 / m0
    w = np.where(m, 1 / v, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = (w * y).sum(1) / w.sum(1)
        k = m.sum(1)
        Q = (w * (y - est[:, None]) ** 2).sum(1)
        C = w.sum(1) - (w**2).sum(1) / w.sum(1)
        tau2 = np.where(k > 1, np.maximum(0, (Q - (k - 1)) / C), 0)
        ws = np.where(m, 1 / (v + tau2[:, None]), 0)
        return (ws * y).sum(1) / ws.sum(1), 1 / np.sqrt(ws.sum(1))


def perf(est, se, truth):
    """Bias, empirical SE and coverage over replicates in which the estimate exists."""
    ok = np.isfinite(est)
    e, s = est[ok], se[ok]
    if ok.sum() < 10:
        return dict(n=int(ok.sum()))
    cov = np.mean((e - 1.96 * s <= truth) & (truth <= e + 1.96 * s))
    return dict(
        n=int(ok.sum()),
        bias=float(e.mean() - truth),
        mcse_bias=float(e.std(ddof=1) / np.sqrt(ok.sum())),
        empse=float(e.std(ddof=1)),
        cover=float(cov),
        mcse_cover=float(np.sqrt(cov * (1 - cov) / ok.sum())),
    )


def scenario(K, ratio, N, pmed, rr, c, rng):
    a, b = ratio
    n1 = int(round(N * a / (a + b)))
    n0 = N - n1
    lp = np.log(pmed / (1 - pmed)) + SD_LOGIT * rng.standard_normal((REPS, K))
    p0 = 1 / (1 + np.exp(-lp))
    p1 = np.minimum(rr * p0, 0.99)
    x1 = rng.binomial(n1, p1).astype(float)
    x0 = rng.binomial(n0, p0).astype(float)
    N1, N0 = np.full_like(x1, n1), np.full_like(x0, n0)
    truth = np.log(rr)
    allm = np.ones_like(x1, bool)
    out = {"M0": perf(*mh_masked(x1, N1, x0, N0, allm), truth)}
    if c > 0:
        post = (x1 >= kcross(n1, c)) | (x0 >= kcross(n0, c))
        out["M1"] = perf(*mh_masked(x1, N1, x0, N0, post), truth)
        out["M2"] = perf(*dl_masked(x1, N1, x0, N0, post), truth)
        out["M3"] = perf(*dl_masked(np.where(post, x1, 0), N1, np.where(post, x0, 0), N0, allm), truth)
        asym = (x1 / n1 >= c) & (x1 / n1 > x0 / n0)
        out["M4"] = perf(*mh_masked(x1, N1, x0, N0, asym), truth)
        pooled = (x1 + x0) >= kcross(N, c)
        out["M7"] = perf(*mh_masked(x1, N1, x0, N0, pooled), truth)
        out["_posted"] = dict(
            mean_rows_posted_anyarm=float(post.sum(1).mean()), mean_rows_posted_pooled=float(pooled.sum(1).mean())
        )
    return out


def main():
    rng = np.random.default_rng(20261001)
    rows = []
    t0 = time.time()
    for K, ratio, N, pmed, rr, c in itertools.product(KS, RATIOS, NS, PMED, RRS, CS):
        res = scenario(K, ratio, N, pmed, rr, c, rng)
        extra = res.pop("_posted", {})
        for m, v in res.items():
            rows.append(
                dict(K=K, ratio=f"{ratio[0]}:{ratio[1]}", N=N, pmed=pmed, rr=rr, c=c, method=m, reps=REPS, **v, **extra)
            )
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["estimable_share"] = df.n / df.reps
    df.to_csv(OUT / "naive_methods.csv", index=False)
    print("scenarios:", len(rows), "time:", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
