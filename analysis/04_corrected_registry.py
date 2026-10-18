"""Selection-corrected likelihood applied to the registry experiment.

For each term analysed in the registry experiment, the corrected likelihood is fitted to
the posted rows (5% any-arm rule) and compared with (i) the Mantel-Haenszel risk
ratio from all rows and (ii) the same model fitted to all rows with at least one
event (threshold 0), which separates the selection correction from model misfit.

Usage: python 04_corrected_registry.py [--workers 1]
"""

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from common import RESULTS, STRATA, C, load, post_flags, write_json
from thresholdbias import fit_corrected

OUT = RESULTS / "registry"


def fit_term(args):
    warnings.filterwarnings("ignore")
    stratum, term, post_rows, all_rows = args
    xd, nd, xp, np_ = post_rows
    f = fit_corrected(xd, nd, xp, np_, C, ci="profile")
    g = fit_corrected(*all_rows, 0.0, ci=None)
    return dict(
        stratum=stratum,
        term=term,
        n_posted=len(xd),
        lr_corrected=f["logRR"],
        lo=f["lo"],
        hi=f["hi"],
        sigma=f["sigma"],
        converged=f["converged"],
        sigma_at_bound=f["sigma_at_bound"],
        lr_same_model_all_rows=g["logRR"],
        converged_all_rows=g["converged"],
    )


def build_jobs():
    t, r = load(0)
    o = r[(r.type == "O") & ((r.xd + r.xp) > 0)].copy()
    o["post"], o["pool"] = post_flags(o)
    D = pd.read_csv(OUT / "rethreshold_primary_terms.csv")
    D = D[np.isfinite(D.shift_post)]
    jobs = []
    for row in D.itertuples():
        a_ = o[(o.stratum == row.stratum) & (o.term == row.term)]
        p = a_[a_.post]
        jobs.append(
            (
                row.stratum,
                row.term,
                (p.xd.values, p.nd.values, p.xp.values, p.np.values),
                (a_.xd.values, a_.nd.values, a_.xp.values, a_.np.values),
            )
        )
    return D, jobs


def summarize(res, D):
    M = pd.DataFrame(res).merge(D[["stratum", "term", "lr_full", "lr_post", "nposted"]], on=["stratum", "term"])
    M.to_csv(OUT / "corrected_registry_terms.csv", index=False)

    rng = np.random.default_rng(3)
    summ = {}
    for s in STRATA:
        g = M[(M.stratum == s) & np.isfinite(M.lr_corrected)]
        if len(g) == 0:
            continue
        block = {}
        for nm, sub in [("all", g), ("lt5", g[g.nposted < 5]), ("ge5", g[g.nposted >= 5])]:
            if len(sub) == 0:
                continue
            naive = (sub.lr_post - sub.lr_full).values
            corr = (sub.lr_corrected - sub.lr_full).values
            same = (sub.lr_corrected - sub.lr_same_model_all_rows).values
            bs = np.array(
                [
                    [np.median(v[idx]) for v in (naive, corr, same)]
                    for idx in (rng.integers(0, len(naive), len(naive)) for _ in range(2000))
                ]
            )
            ci = np.percentile(bs, [2.5, 97.5], axis=0)
            block[nm] = dict(
                terms=int(len(sub)),
                naive=[float(np.median(naive))] + ci[:, 0].tolist(),
                corrected_vs_mh_all_rows=[float(np.median(corr))] + ci[:, 1].tolist(),
                corrected_vs_same_model_all_rows=[float(np.median(same))] + ci[:, 2].tolist(),
            )
        block["median_abs_error_naive"] = float(np.median(np.abs(g.lr_post - g.lr_full)))
        block["median_abs_error_corrected"] = float(np.median(np.abs(g.lr_corrected - g.lr_full)))
        block["ci_contains_mh_all_rows"] = float(((g.lo <= g.lr_full) & (g.lr_full <= g.hi)).mean())
        block["not_converged"] = int((~g.converged.astype(bool)).sum())
        block["sigma_at_bound"] = int(g.sigma_at_bound.astype(bool).sum())
        summ[s] = block
    write_json(summ, OUT / "corrected_registry.json")
    print(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    D, jobs = build_jobs()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(fit_term, jobs, chunksize=4))
    summarize(res, D)


if __name__ == "__main__":
    main()
