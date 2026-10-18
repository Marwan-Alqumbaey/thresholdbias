"""Observational comparison: trials that used a 5% threshold against threshold-0 trials.

For each allocation group and each term with events in at least 10 trials at each
threshold (5% and 0), d = log RR_MH(5% trials, posted rows) - log RR_MH(threshold-0 trials, all rows).
Serious adverse events, which have no threshold, are the negative control.
"""

import numpy as np
import pandas as pd
from common import RESULTS, STRATA, load, mh_parts, write_json
from scipy import sparse

B, MIN = 1000, 10
OUT = RESULTS / "registry"
t0, r0 = load(0)
t5, r5 = load(5)
r0["grp"] = 0
r5["grp"] = 5
R = pd.concat([r0, r5])
R = R[(R.xd + R.xp) > 0].copy()
R["R"], R["S"], _ = mh_parts(R)
TT = pd.concat([t0.assign(grp=0), t5.assign(grp=5)])
rng = np.random.default_rng(7)
summ, boots, allterms = {}, {}, []
for typ in ["O", "S"]:
    for sname in STRATA:
        key = f"{'nonserious' if typ == 'O' else 'serious'} {sname}"
        x = R[(R.type == typ) & (R.stratum == sname)]
        c0 = x[x.grp == 0].groupby("term").nct.nunique()
        c5 = x[x.grp == 5].groupby("term").nct.nunique()
        terms = np.array(sorted(set(c0[c0 >= MIN].index) & set(c5[c5 >= MIN].index)))
        if len(terms) < 5:
            summ[key] = dict(terms=int(len(terms)))
            continue
        x = x[x.term.isin(terms)]
        mix = {n: i for i, n in enumerate(terms)}
        tr = TT[TT.stratum == sname]
        trials = tr.nct.values
        tix = {n: i for i, n in enumerate(trials)}
        g0 = tr.grp.values == 0
        ri, ti, gg = x.nct.map(tix).values, x.term.map(mix).values, x.grp.values
        A = sparse.csr_matrix((np.ones(len(x)), (ti, np.arange(len(x)))), shape=(len(terms), len(x)))

        def est(w):
            wr = w[ri]
            res = []
            for g in (0, 5):
                m = gg == g
                Rs = A @ (wr * x.R.values * m)
                Ss = A @ (wr * x.S.values * m)
                with np.errstate(divide="ignore", invalid="ignore"):
                    res.append(np.where((Rs > 0) & (Ss > 0), np.log(Rs / Ss), np.nan))
            return res[1] - res[0], res

        d, (l0, l5) = est(np.ones(len(trials)))
        allterms.append(pd.DataFrame(dict(type=typ, stratum=sname, term=terms, lr_thr0=l0, lr_thr5=l5, d=d)))
        bm = []
        for b in range(B):
            w = np.zeros(len(trials))
            for m in (g0, ~g0):
                idx = np.where(m)[0]
                w[idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx))
            db, _ = est(w)
            bm.append(np.nanmedian(db))
        boots[key] = np.array(bm)
        summ[key] = dict(
            terms=int(len(terms)),
            median_difference=float(np.nanmedian(d)),
            ci=[float(v) for v in np.nanpercentile(bm, [2.5, 97.5])],
        )
        print(key, summ[key], flush=True)
for sname in STRATA:
    a_, b_ = f"nonserious {sname}", f"serious {sname}"
    if a_ in boots and b_ in boots:
        dd = boots[a_] - boots[b_]
        summ[f"nonserious minus serious {sname}"] = dict(
            est=summ[a_]["median_difference"] - summ[b_]["median_difference"],
            ci=[float(v) for v in np.nanpercentile(dd, [2.5, 97.5])],
        )
summ["median_arm_sizes"] = {
    f"{sname} threshold {g}": dict(
        drug=float(TT[(TT.stratum == sname) & (TT.grp == g)].drug_o_risk.median()),
        placebo=float(TT[(TT.stratum == sname) & (TT.grp == g)].pl_o_risk.median()),
        trials=int(((TT.stratum == sname) & (TT.grp == g)).sum()),
    )
    for sname in STRATA
    for g in (0, 5)
}
OUT.mkdir(parents=True, exist_ok=True)
pd.concat(allterms).to_csv(OUT / "observational_terms.csv", index=False)
write_json(summ, OUT / "observational.json")
