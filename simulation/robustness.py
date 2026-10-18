"""Simulation, part C: departures from the assumptions of the corrected likelihood.

Five scenarios with a 5% threshold and 30 trials per meta-analysis, 100 replicates each:
  0-1  2:1 allocation, 300 participants, log risk ratio varying between trials (SD 0.3)
  2    2:1 allocation, 300 participants, placebo risk from a two-component mixture
       (logit risk centred 0.7 below or above the logit of the median, SD 0.3)
  3-4  arm sizes drawn from the registry trials that declared a 5% threshold and had
       more participants in the drug arm than in the placebo arm (ratio above 1.1)
The placebo risk is logit-normal around 3% with SD 0.5 (except scenario 2).
Methods: M0, M1, M5, M6 and M7 as in parts A and B. The estimand is the mean log
risk ratio. Per-replicate results are saved, so a run can be resumed and split over processes.

Usage: python robustness.py --chunk 0 --nchunks 2 [--reps 100]
       python robustness.py --summarize
"""

import argparse
import sys
import time
import warnings

import numpy as np
from common_sim import OUT, ROOT, SD_LOGIT, write_json
from thresholdbias import fit_corrected, kcross, mh_rr

REPDIR = OUT / "robustness_replicates"
K, C, PMED = 30, 0.05, 0.03


def scenarios():
    """The five scenarios of part C."""
    base = dict(design="2:1", N=300, pmed=PMED, tau=0.0, baseline="logit-normal")
    return [
        dict(base, rr=1.0, tau=0.3),
        dict(base, rr=1.5, tau=0.3),
        dict(base, rr=1.0, baseline="mixture"),
        dict(base, design="registry", N=None, rr=1.0),
        dict(base, design="registry", N=None, rr=1.5),
    ]


def registry_designs():
    """Participants at risk (drug, placebo) in 5%-threshold trials with ratio above 1.1."""
    sys.path.insert(0, str(ROOT / "analysis"))
    from common import load

    t, _ = load(5)
    t = t[(t.drug_o_risk > 0) & (t.pl_o_risk > 0) & (t.ratio > 1.1)]
    return t.drug_o_risk.values.astype(int), t.pl_o_risk.values.astype(int)


def draw(s, rng, pool):
    """Arm sizes and event counts of one simulated meta-analysis."""
    if s["design"] == "registry":
        i = rng.integers(0, len(pool[0]), K)
        n1, n0 = pool[0][i], pool[1][i]
    else:
        n1, n0 = np.full(K, 200), np.full(K, 100)
    lm = np.log(s["pmed"] / (1 - s["pmed"]))
    if s["baseline"] == "mixture":
        lp = lm + 0.7 * rng.choice([-1.0, 1.0], K) + 0.3 * rng.standard_normal(K)
    else:
        lp = lm + SD_LOGIT * rng.standard_normal(K)
    p0 = 1 / (1 + np.exp(-lp))
    lrr = np.log(s["rr"]) + s["tau"] * rng.standard_normal(K)
    p1 = np.minimum(np.exp(lrr) * p0, 0.99)
    return n1, n0, rng.binomial(n1, p1), rng.binomial(n0, p0)


def run(chunk, nchunks, reps):
    warnings.filterwarnings("ignore")
    REPDIR.mkdir(parents=True, exist_ok=True)
    pool = registry_designs()
    for si, s in enumerate(scenarios()):
        if si % nchunks != chunk or (REPDIR / f"scenario_{si:02d}.npz").exists():
            continue
        rng = np.random.default_rng(7000 + si)
        t0 = time.time()
        rec = {m: [] for m in ["M0", "M1", "M5", "M6", "M7"]}
        for _ in range(reps):
            n1, n0, x1, x0 = draw(s, rng, pool)
            post = (x1 >= kcross(n1, C)) | (x0 >= kcross(n0, C))
            pooled = (x1 + x0) >= kcross(n1 + n0, C)
            e, se = mh_rr(x1, n1, x0, n0)
            rec["M0"].append((e, e - 1.96 * se, e + 1.96 * se))
            e, se = mh_rr(x1[pooled], n1[pooled], x0[pooled], n0[pooled]) if pooled.any() else (np.nan, np.nan)
            rec["M7"].append((e, e - 1.96 * se, e + 1.96 * se))
            if post.sum() >= 2:
                e, se = mh_rr(x1[post], n1[post], x0[post], n0[post])
                rec["M1"].append((e, e - 1.96 * se, e + 1.96 * se))
                f5 = fit_corrected(x1[post], n1[post], x0[post], n0[post], C)
                rec["M5"].append((f5["logRR"], f5["lo"], f5["hi"]))
                up = ~post
                f6 = fit_corrected(
                    x1[post], n1[post], x0[post], n0[post], C, unposted=(n1[up], n0[up], np.full(up.sum(), C))
                )
                rec["M6"].append((f6["logRR"], f6["lo"], f6["hi"]))
            else:
                for m in ["M1", "M5", "M6"]:
                    rec[m].append((np.nan,) * 3)
        np.savez(REPDIR / f"scenario_{si:02d}.npz", **{m: np.array(v, float) for m, v in rec.items()})
        print(si, s, round(time.time() - t0), "s", flush=True)


def summarize():
    out = []
    for si, s in enumerate(scenarios()):
        z = np.load(REPDIR / f"scenario_{si:02d}.npz")
        truth = np.log(s["rr"])
        common = np.isfinite(z["M1"][:, 0]) & np.isfinite(z["M5"][:, 0]) & np.isfinite(z["M6"][:, 0])
        res = {}
        for m in ["M0", "M1", "M5", "M6", "M7"]:
            v = z[m][common & np.isfinite(z[m][:, 0])]
            e = v[:, 0]
            cov = np.mean((v[:, 1] <= truth) & (truth <= v[:, 2]))
            res[m] = dict(
                n=int(len(e)),
                bias=float(e.mean() - truth),
                mcse_bias=float(e.std(ddof=1) / np.sqrt(len(e))),
                empse=float(e.std(ddof=1)),
                cover=float(cov),
                mcse_cover=float(np.sqrt(cov * (1 - cov) / len(e))),
            )
        out.append(
            dict(scenario=si, **s, K=K, c=C, reps=len(z["M0"]), common_replicates=int(common.sum()), results=res)
        )
    write_json(out, OUT / "robustness.json")
    for o in out:
        print(
            o["scenario"],
            o["design"],
            o["baseline"],
            o["tau"],
            o["rr"],
            o["common_replicates"],
            {m: (round(v["bias"], 3), round(v["cover"], 3)) for m, v in o["results"].items()},
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.chunk, a.nchunks, a.reps)
