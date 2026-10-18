"""Simulation, part B: selection-corrected estimators.

Twenty scenarios (see common_sim.corrected_scenarios), 200 replicates each.
  M0 MH on all trials; M1 MH on posted rows;
  M5 corrected likelihood on posted rows; M6 full likelihood (posted rows and unposted trials).
Per-replicate results are saved, so a run can be resumed and split over processes.

Usage: python corrected_methods.py --chunk 0 --nchunks 2 [--reps 200]
       python corrected_methods.py --summarize
"""

import argparse
import time
import warnings

import numpy as np
from common_sim import OUT, SD_LOGIT, corrected_scenarios, write_json
from thresholdbias import fit_corrected, kcross, mh_rr

REPDIR = OUT / "corrected_replicates"


def run(chunk, nchunks, reps):
    warnings.filterwarnings("ignore")
    REPDIR.mkdir(parents=True, exist_ok=True)
    for si, s in enumerate(corrected_scenarios()):
        if si % nchunks != chunk or (REPDIR / f"scenario_{si:02d}.npz").exists():
            continue
        rng = np.random.default_rng(5000 + si)
        t0 = time.time()
        a, b = s["ratio"]
        n1 = int(round(s["N"] * a / (a + b)))
        n0 = s["N"] - n1
        k1, k0 = kcross(n1, s["c"]), kcross(n0, s["c"])
        rec = {m: [] for m in ["M0", "M1", "M5", "M6"]}
        flags = {"M5": [], "M6": []}
        for _ in range(reps):
            lp = np.log(s["pmed"] / (1 - s["pmed"])) + SD_LOGIT * rng.standard_normal(s["K"])
            p0 = 1 / (1 + np.exp(-lp))
            p1 = np.minimum(s["rr"] * p0, 0.99)
            x1, x0 = rng.binomial(n1, p1), rng.binomial(n0, p0)
            N1, N0 = np.full(s["K"], n1), np.full(s["K"], n0)
            post = (x1 >= k1) | (x0 >= k0)
            e, se = mh_rr(x1, N1, x0, N0)
            rec["M0"].append((e, e - 1.96 * se, e + 1.96 * se))
            if post.sum() >= 2:
                e, se = mh_rr(x1[post], N1[post], x0[post], N0[post])
                rec["M1"].append((e, e - 1.96 * se, e + 1.96 * se))
                f5 = fit_corrected(x1[post], N1[post], x0[post], N0[post], s["c"])
                rec["M5"].append((f5["logRR"], f5["lo"], f5["hi"]))
                flags["M5"].append((f5["converged"], f5["sigma_at_bound"]))
                up = ~post
                f6 = fit_corrected(
                    x1[post], N1[post], x0[post], N0[post], s["c"], unposted=(N1[up], N0[up], np.full(up.sum(), s["c"]))
                )
                rec["M6"].append((f6["logRR"], f6["lo"], f6["hi"]))
                flags["M6"].append((f6["converged"], f6["sigma_at_bound"]))
            else:
                for m in ["M1", "M5", "M6"]:
                    rec[m].append((np.nan,) * 3)
                for m in ["M5", "M6"]:
                    flags[m].append((False, False))
        np.savez(
            REPDIR / f"scenario_{si:02d}.npz",
            **{m: np.array(v, float) for m, v in rec.items()},
            **{"flags_" + m: np.array(v, bool) for m, v in flags.items()},
        )
        print(si, s, round(time.time() - t0), "s", flush=True)


def summarize():
    out = []
    for si, s in enumerate(corrected_scenarios()):
        z = np.load(REPDIR / f"scenario_{si:02d}.npz")
        truth = np.log(s["rr"])
        common = np.isfinite(z["M1"][:, 0]) & np.isfinite(z["M5"][:, 0]) & np.isfinite(z["M6"][:, 0])
        res = {}
        for m in ["M0", "M1", "M5", "M6"]:
            v = z[m][common]
            e = v[:, 0]
            cov = np.mean((v[:, 1] <= truth) & (truth <= v[:, 2]))
            res[m] = dict(
                bias=float(e.mean() - truth),
                mcse_bias=float(e.std(ddof=1) / np.sqrt(len(e))),
                empse=float(e.std(ddof=1)),
                cover=float(cov),
                mcse_cover=float(np.sqrt(cov * (1 - cov) / len(e))),
                ci_missing=int(np.isnan(v[:, 1]).sum() + np.isnan(v[:, 2]).sum()),
            )
        fl = {
            m: dict(
                not_converged=int((~z["flags_" + m][common, 0]).sum()),
                sigma_at_bound=int(z["flags_" + m][common, 1].sum()),
            )
            for m in ["M5", "M6"]
        }
        out.append(
            dict(
                scenario=si,
                K=s["K"],
                ratio=f"{s['ratio'][0]}:{s['ratio'][1]}",
                N=s["N"],
                pmed=s["pmed"],
                rr=s["rr"],
                c=s["c"],
                reps=len(z["M0"]),
                common_replicates=int(common.sum()),
                results=res,
                flags=fl,
            )
        )
    write_json(out, OUT / "corrected_methods.json")
    for o in out:
        print(
            o["scenario"],
            o["ratio"],
            o["pmed"],
            o["rr"],
            o["K"],
            o["common_replicates"],
            {m: (round(v["bias"], 3), round(v["cover"], 3)) for m, v in o["results"].items()},
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else run(a.chunk, a.nchunks, a.reps)
