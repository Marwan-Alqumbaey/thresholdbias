"""Model-predicted shift for each term of the registry experiment.

For each allocation group and term, a beta-binomial model is fitted to the event
counts (both arms combined) of all trials in the group (trials that did not post
the term had no events). With theta set to the complete-data Mantel-Haenszel
risk ratio, counts are simulated 200 times from the posterior of each trial's
risk, the 5% any-arm rule is applied, and the shift
log RR_MH(posted) - log RR_MH(all) is averaged.

Usage: python 03_predicted_shift.py [--tag primary] [--min-arm 0]
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from common import RESULTS, C, load, write_json
from scipy.optimize import minimize
from scipy.stats import betabinom
from thresholdbias import kcross

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="primary")
ap.add_argument("--min-arm", type=int, default=0)
a = ap.parse_args()
REPS = 200
OUT = RESULTS / "registry"

t, r = load(0)
if a.min_arm:
    t = t[(t.drug_o_risk >= a.min_arm) & (t.pl_o_risk >= a.min_arm)]
    r = r[r.nct.isin(t.nct)]
D = pd.read_csv(OUT / f"rethreshold_{a.tag}_terms.csv")
D = D[np.isfinite(D.shift_post)]
o = r[r.type == "O"]
rng = np.random.default_rng(99)
_kc = {}


def K(n):
    n = np.asarray(n, int)
    for v in np.unique(n):
        if v not in _kc:
            _kc[v] = kcross(int(v), C)
    return np.vectorize(_kc.get)(n)


out = []
for s in D.stratum.unique():
    ts = t[t.stratum == s][["nct", "drug_o_risk", "pl_o_risk"]].reset_index(drop=True)
    for term in D[D.stratum == s].term:
        x = ts.merge(o[(o.stratum == s) & (o.term == term)][["nct", "xd", "nd", "xp", "np"]], on="nct", how="left")
        nd = x.nd.fillna(x.drug_o_risk).values.astype(int)
        np_ = x.np.fillna(x.pl_o_risk).values.astype(int)
        ok = (nd > 0) & (np_ > 0)
        nd, np_ = nd[ok], np_[ok]
        xd, xp = x.xd.fillna(0).values[ok], x.xp.fillna(0).values[ok]
        X, N = xd + xp, nd + np_
        th = (xd * np_ / N).sum() / (xp * nd / N).sum()

        def nll(p):
            return -betabinom.logpmf(X, N, np.exp(p[0]), np.exp(p[1])).sum()

        m = X.sum() / N.sum()
        f = minimize(nll, [np.log(m * 10), np.log((1 - m) * 10)], method="Nelder-Mead")
        al, be = np.exp(f.x)
        q = rng.beta(al + X, be + N - X, size=(REPS, len(X)))
        p0 = np.clip(q * N / (th * nd + np_), 0, 1)
        p1 = np.clip(th * p0, 0, 1)
        yd, yp = rng.binomial(nd, p1), rng.binomial(np_, p0)
        post = (yd >= K(nd)) | (yp >= K(np_))
        w1, w0 = yd * np_ / N, yp * nd / N
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.log((w1 * post).sum(1) / (w0 * post).sum(1)) - np.log(w1.sum(1) / w0.sum(1))
        sh = sh[np.isfinite(sh)]
        out.append(
            dict(stratum=s, term=term, predicted_shift=float(sh.mean()) if len(sh) > 20 else np.nan, draws_used=len(sh))
        )
P = pd.DataFrame(out).merge(
    D[["stratum", "term", "shift_post", "lr_full", "nposted", "ntrials"]], on=["stratum", "term"]
)
P.to_csv(OUT / f"predicted_shift_{a.tag}_terms.csv", index=False)
res = {}
for s, g in P.groupby("stratum"):
    g = g.dropna(subset=["predicted_shift"])
    b = np.polyfit(g.predicted_shift, g.shift_post, 1)
    res[s] = dict(
        terms=len(g),
        median_predicted=float(g.predicted_shift.median()),
        median_observed=float(g.shift_post.median()),
        slope=float(b[0]),
        intercept=float(b[1]),
        spearman=float(g[["predicted_shift", "shift_post"]].corr("spearman").iloc[0, 1]),
    )
write_json(res, OUT / f"predicted_shift_{a.tag}.json")
print(res)
