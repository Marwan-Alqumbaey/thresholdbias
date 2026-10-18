"""Registry experiment (primary analysis and sensitivity analyses).

Threshold-0 trials posted every non-serious adverse event. Applying the 5% rule to
them shows what the rule does, with the complete data as reference. For each
allocation group and each term with events in at least ``--min-trials`` trials,
the Mantel-Haenszel risk ratio is computed from all rows, from rows posted under
the any-arm rule and from rows posted under the pooled-arm rule. Confidence
intervals come from a bootstrap that resamples whole trials.

Usage: python 02_rethreshold.py [--tag primary] [--parallel] [--verified]
       [--min-trials 10] [--c 0.05] [--min-arm 0] [--small-arm 0] [--boot 2000]
"""

import argparse

import numpy as np
import pandas as pd
from common import RESULTS, STRATA, C, load, mh_parts, post_flags, write_json
from scipy import sparse

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="primary")
ap.add_argument("--parallel", action="store_true", help="parallel-group designs only")
ap.add_argument(
    "--verified", action="store_true", help="drop trials (both arms >= 20) with no posted row below 5%% in both arms"
)
ap.add_argument("--min-trials", type=int, default=10)
ap.add_argument("--c", type=float, default=C)
ap.add_argument("--min-arm", type=int, default=0, help="keep trials with both arms of at least this size")
ap.add_argument("--small-arm", type=int, default=0, help="keep trials with an arm below this size")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()


def ci95(v):
    """Percentile 95% interval of bootstrap values, ignoring undefined ones."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [float("nan"), float("nan")]


B = a.boot
OUT = RESULTS / "registry"

t, r = load(0, exclude_crossover=True, parallel_only=a.parallel)
if a.verified:
    o = r[r.type == "O"]
    below = ((o.xd / o.nd <= C) & (o.xp / o.np <= C)).groupby(o.nct).any()
    cand = t[(t.drug_o_risk >= 20) & (t.pl_o_risk >= 20)].nct
    drop = set(cand) - set(below[below].index)
    t, r = t[~t.nct.isin(drop)], r[~r.nct.isin(drop)]
if a.min_arm:
    keep = t[(t.drug_o_risk >= a.min_arm) & (t.pl_o_risk >= a.min_arm)].nct
    t, r = t[t.nct.isin(keep)], r[r.nct.isin(keep)]
if a.small_arm:
    keep = t[(t.drug_o_risk < a.small_arm) | (t.pl_o_risk < a.small_arm)].nct
    t, r = t[t.nct.isin(keep)], r[r.nct.isin(keep)]
o = r[(r.type == "O") & ((r.xd + r.xp) > 0)].copy()
o["post"], o["pool"] = post_flags(o, a.c)
o["R"], o["S"], o["P"] = mh_parts(o)

EXN = [
    "median",
    "ivw_mean",
    "median_ge5rows",
    "median_lt5rows",
    "median_inf_incl",
    "median_nearnull",
    "median_post_common",
    "median_pool_common",
]
rng = np.random.default_rng(20261001)
out, boot, extra, bxs = [], {}, {}, {}
for sname in STRATA:
    os_ = o[o.stratum == sname]
    cnt = os_.groupby("term").nct.nunique()
    terms = cnt[cnt >= a.min_trials].index
    os_ = os_[os_.term.isin(terms)].copy()
    trials = np.array(sorted(t[t.stratum == sname].nct))
    tix = {n: i for i, n in enumerate(trials)}
    terms = np.array(sorted(terms))
    mix = {n: i for i, n in enumerate(terms)}
    ri = os_.nct.map(tix).values
    ti = os_.term.map(mix).values
    A = sparse.csr_matrix((np.ones(len(os_)), (ti, np.arange(len(os_)))), shape=(len(terms), len(os_)))

    def est(w_trial):
        w = w_trial[ri]
        res = {}
        for nm, m in [("full", np.ones(len(os_), bool)), ("post", os_.post.values), ("pool", os_.pool.values)]:
            Rs = A @ (w * os_.R.values * m)
            Ss = A @ (w * os_.S.values * m)
            Ps = A @ (w * os_.P.values * m)
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.where((Rs > 0) & (Ss > 0), np.log(Rs / Ss), np.nan)
                se = np.sqrt(Ps / (Rs * Ss))
            res[nm] = (lr, se)
        return res

    e = est(np.ones(len(trials)))
    df = pd.DataFrame(
        {
            "stratum": sname,
            "term": terms,
            "ntrials": cnt[terms].values,
            "lr_full": e["full"][0],
            "se_full": e["full"][1],
            "lr_post": e["post"][0],
            "se_post": e["post"][1],
            "lr_pool": e["pool"][0],
            "se_pool": e["pool"][1],
            "nposted": A @ os_.post.values.astype(float),
            "npool": A @ os_.pool.values.astype(float),
        }
    )
    df["shift_post"] = df.lr_post - df.lr_full
    df["shift_pool"] = df.lr_pool - df.lr_full
    out.append(df)

    @np.errstate(divide="ignore", invalid="ignore")
    def extras(eb, w):
        s = eb["post"][0] - eb["full"][0]
        ok = np.isfinite(s)
        npw = A @ (w[ri] * os_.post.values)
        v = eb["post"][1] ** 2
        lf, lp = eb["full"][0], eb["post"][0]
        Rs = A @ (w[ri] * os_.R.values * os_.post.values)
        Ss = A @ (w[ri] * os_.S.values * os_.post.values)
        sinf = (
            np.where(
                np.isfinite(lp),
                lp,
                np.where((Rs == 0) & (Ss > 0), -np.inf, np.where((Rs > 0) & (Ss == 0), np.inf, np.nan)),
            )
            - lf
        )
        sinf = sinf[np.isfinite(lf) & ~np.isnan(sinf)]
        near = ok & (np.abs(lf) < 0.2)
        sp = eb["pool"][0] - lf
        common = ok & np.isfinite(sp)
        return [
            np.nanmedian(s[ok]) if ok.sum() >= 5 else np.nan,
            np.sum(s[ok] / v[ok]) / np.sum(1 / v[ok]) if ok.sum() >= 5 else np.nan,
            np.median(s[ok & (npw >= 5)]) if (ok & (npw >= 5)).sum() >= 5 else np.nan,
            np.median(s[ok & (npw < 5)]) if (ok & (npw < 5)).sum() >= 5 else np.nan,
            np.median(sinf) if len(sinf) >= 5 else np.nan,
            np.median(s[near]) if near.sum() >= 5 else np.nan,
            np.median(s[common]) if common.sum() >= 5 else np.nan,
            np.median(sp[common]) if common.sum() >= 5 else np.nan,
        ]

    ext0 = extras(e, np.ones(len(trials)))
    bm = np.full((B, 2), np.nan)
    bx = np.full((B, len(EXN)), np.nan)
    for b in range(B):
        w = rng.multinomial(len(trials), np.ones(len(trials)) / len(trials)).astype(float)
        eb = est(w)
        for j, nm in enumerate(["post", "pool"]):
            s = eb[nm][0] - eb["full"][0]
            if np.isfinite(s).sum() >= 5:
                bm[b, j] = np.nanmedian(s)
        bx[b] = extras(eb, w)
    boot[sname] = bm
    bxs[sname] = bx
    xs = df.dropna(subset=["shift_post"])
    sf = np.abs(xs.lr_full / xs.se_full) > 1.96
    sp = np.abs(xs.lr_post / xs.se_post) > 1.96
    extra[sname] = {n: dict(est=float(ext0[i]), ci=ci95(bx[:, i])) for i, n in enumerate(EXN)}
    extra[sname]["n_terms_ge5rows"] = int(((df.nposted.values >= 5) & np.isfinite(df.shift_post.values)).sum())
    extra[sname]["n_nonestimable_post"] = int((~np.isfinite(df.shift_post.values)).sum())
    extra[sname]["n_no_posted_row"] = int((df.nposted.values == 0).sum())
    extra[sname]["n_one_arm_without_events"] = int(((df.nposted.values > 0) & ~np.isfinite(df.lr_post.values)).sum())
    extra[sname]["n_terms_common"] = int((np.isfinite(df.shift_post.values) & np.isfinite(df.shift_pool.values)).sum())
    extra[sname]["significance"] = dict(
        lost=int((sf & ~sp).sum()),
        gained=int((~sf & sp).sum()),
        both=int((sf & sp).sum()),
        direction_changed=int((np.sign(xs.lr_full) != np.sign(xs.lr_post)).sum()),
        n=int(len(xs)),
    )
    print(sname, "trials", len(trials), "terms", len(terms), "median shift %.3f" % df.shift_post.median(), flush=True)

D = pd.concat(out)
OUT.mkdir(parents=True, exist_ok=True)
D.to_csv(OUT / f"rethreshold_{a.tag}_terms.csv", index=False)
summ = {
    "trials": {s: int((t.stratum == s).sum()) for s in STRATA},
    "trials_with_events": {s: int(o[o.stratum == s].nct.nunique()) for s in STRATA},
}
for sname in STRATA:
    d = D[D.stratum == sname]
    bm = boot[sname]
    summ[sname] = dict(
        terms=int(len(d)),
        estimable_post=int(np.isfinite(d.shift_post).sum()),
        estimable_pool=int(np.isfinite(d.shift_pool).sum()),
        median_shift_post=float(d.shift_post.median()),
        ci_post=ci95(bm[:, 0]),
        p_one_sided_post=float(np.mean(bm[:, 0] >= 0)),
        frac_negative_post=float((d.shift_post.dropna() < 0).mean()),
        median_shift_pool=float(d.shift_pool.median()),
        ci_pool=ci95(bm[:, 1]),
        rows_posted_any=int(d.nposted.sum()),
        rows_posted_pooled=int(d.npool.sum()),
        rows=int(d.ntrials.sum()),
        **extra[sname],
    )
dd = boot[">=1.7"][:, 0] - boot["0.9-1.1"][:, 0]
summ["H2_difference"] = dict(
    est=summ[">=1.7"]["median_shift_post"] - summ["0.9-1.1"]["median_shift_post"],
    ci=ci95(dd),
    p_one_sided=float(np.nanmean(dd >= 0)),
)
for i, n in [(5, "median_nearnull"), (1, "ivw_mean"), (2, "median_ge5rows")]:
    dd = bxs[">=1.7"][:, i] - bxs["0.9-1.1"][:, i]
    summ[f"H2_difference_{n}"] = dict(
        est=summ[">=1.7"][n]["est"] - summ["0.9-1.1"][n]["est"],
        ci=ci95(dd),
    )
summ["bootstrap_resamples"] = B
write_json(summ, OUT / f"rethreshold_{a.tag}.json")
