"""Rows posted and median shifts under pooled-arm cut-offs of 3% to 5% (exploratory)."""

import numpy as np
import pandas as pd
from common import RESULTS, STRATA, load, mh_parts, post_flags, write_json

t, r = load(0)
o = r[(r.type == "O") & ((r.xd + r.xp) > 0)].copy()
D = pd.read_csv(RESULTS / "registry" / "rethreshold_primary_terms.csv")
o = o.merge(D[["stratum", "term"]], on=["stratum", "term"])
anyarm, _ = post_flags(o, 0.05)
o["R"], o["S"], o["P"] = mh_parts(o)
res = {"rows_with_events": int(len(o)), "rows_posted_anyarm_5": int(anyarm.sum())}
for c in [0.05, 0.04, 0.035, 0.03]:
    _, pooled = post_flags(o, c)
    o["pl"] = pooled
    med = {}
    for s in STRATA:
        g = o[o.stratum == s]
        full = g.groupby("term").apply(lambda d: np.log(d.R.sum() / d.S.sum()), include_groups=False)
        pst = (
            g[g.pl]
            .groupby("term")
            .apply(
                lambda d: np.log(d.R.sum() / d.S.sum()) if d.R.sum() > 0 and d.S.sum() > 0 else np.nan,
                include_groups=False,
            )
        )
        sh = (pst - full.reindex(pst.index)).replace([np.inf, -np.inf], np.nan).dropna()
        med[s] = float(sh.median())
    res[f"pooled_{c}"] = dict(rows_posted=int(pooled.sum()), median_shift=med)
write_json(res, RESULTS / "registry" / "pooled_cutoffs.json")
print(res)
