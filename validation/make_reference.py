"""Reference values from the Python package, used by the R package's equivalence tests."""

import json
import warnings
from pathlib import Path

import numpy as np
from thresholdbias import bias_check, exact_conditional, fit_corrected, iv_rr, kcross, mh_rr, posted_anyarm
from thresholdbias.correct import _prep, _prep_unposted, negll

warnings.filterwarnings("ignore")
OUT = (
    Path(__file__).resolve().parents[1]
    / "r"
    / "thresholdbias"
    / "tests"
    / "testthat"
    / "fixtures"
    / "reference_python.json"
)
rng = np.random.default_rng(2026)
ref = {"kcross": [], "exact": [], "pool": [], "fit": [], "bias_check": []}
PARS = [(0.0, np.log(0.03), np.log(0.5)), (0.3, np.log(0.02), np.log(1.0)), (-0.4, np.log(0.05), np.log(0.2))]
for n in [1, 5, 19, 20, 21, 99, 100, 101, 150, 1000, 2001]:
    for c in [0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 1 / 3, 1.0]:
        for rule in ["strict", "weak"]:
            ref["kcross"].append(dict(n=n, c=c, rule=rule, k=int(kcross(n, c, rule))))
for _ in range(400):
    n1, n0 = int(rng.integers(5, 500)), int(rng.integers(5, 500))
    p0 = float(np.exp(rng.uniform(np.log(0.002), np.log(0.25))))
    p1 = float(min(p0 * np.exp(rng.normal(0, 0.6)), 0.7))
    c = float(rng.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.1]))
    rule = str(rng.choice(["strict", "weak"]))
    r = exact_conditional(n1, n0, p1, p0, c, rule)
    if np.isfinite(r["rr_cond"]):
        ref["exact"].append(
            dict(n1=n1, n0=n0, p1=p1, p0=p0, c=c, rule=rule, **{k: r[k] for k in ["p_post", "e1", "e0", "rr_cond"]})
        )
for _ in range(100):
    K = int(rng.integers(2, 40))
    n1, n0 = rng.integers(10, 300, K), rng.integers(10, 300, K)
    p0 = rng.uniform(0.005, 0.2, K)
    x1, x0 = rng.binomial(n1, np.minimum(p0 * 1.3, 0.9)), rng.binomial(n0, p0)
    a = mh_rr(x1, n1, x0, n0)
    b = iv_rr(x1, n1, x0, n0, method="fixed")
    d = iv_rr(x1, n1, x0, n0, method="dl")
    ref["pool"].append(
        dict(
            x1=x1.tolist(),
            n1=n1.tolist(),
            x0=x0.tolist(),
            n0=n0.tolist(),
            mh=[float(v) for v in a],
            fixed=[float(v) for v in b],
            dl=[float(v) for v in d],
        )
    )
for i in range(6):
    K, n1, n0, th = 120, int(rng.integers(60, 300)), int(rng.integers(30, 150)), [1.0, 1.5, 0.7][i % 3]
    p0 = np.exp(np.log(rng.uniform(0.015, 0.05)) + 0.5 * rng.standard_normal(K))
    x1, x0 = rng.binomial(n1, np.minimum(th * p0, 0.99)), rng.binomial(n0, p0)
    post = posted_anyarm(x1, n1, x0, n0, 0.05)
    xs = dict(
        x1=x1[post].tolist(),
        n1=[n1] * int(post.sum()),
        x0=x0[post].tolist(),
        n0=[n0] * int(post.sum()),
        un1=[n1] * int((~post).sum()),
        un0=[n0] * int((~post).sum()),
    )
    f = fit_corrected(xs["x1"], xs["n1"], xs["x0"], xs["n0"], 0.05)
    g = fit_corrected(xs["x1"], xs["n1"], xs["x0"], xs["n0"], 0.05, unposted=(xs["un1"], xs["un0"], 0.05), ci=None)
    D = _prep(xs["x1"], xs["n1"], xs["x0"], xs["n0"], 0.05, "strict")
    U = _prep_unposted((xs["un1"], xs["un0"], 0.05), "strict")
    lik = [dict(par=list(par), conditional=float(negll(par, D)), full=float(negll(par, D, U))) for par in PARS]
    ref["fit"].append(dict(**xs, conditional=[f["logRR"], f["lo"], f["hi"], f["mu"]], full=g["logRR"], negll=lik))
rb = np.random.default_rng(2027)
for i in range(30):
    K = int(rb.integers(1, 25))
    n1, n0 = rb.integers(5, 800, K), rb.integers(5, 800, K)
    p0 = [0.004, 0.01, 0.02, 0.035, 0.06]
    rr = float(rb.choice([0.67, 1.0, 1.5, 2.0]))
    c = float(rb.choice([0.0, 0.02, 0.05, 0.1]))
    rule = str(rb.choice(["strict", "weak"]))
    sd, tau = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.3), (0.8, 0.5)][i % 4]
    r = bias_check(n1, n0, p0, rr=rr, c=c, rule=rule, sd_logit=sd, tau=tau, nodes=15)
    ref["bias_check"].append(
        dict(
            n1=n1.tolist(),
            n0=n0.tolist(),
            p0=p0,
            rr=rr,
            c=c,
            rule=rule,
            sd_logit=sd,
            tau=tau,
            **{k: r[k].tolist() for k in ["rr_complete", "rr_posted", "distortion", "p_post"]},
        )
    )
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(ref, fh)
print({k: len(v) for k, v in ref.items()})
