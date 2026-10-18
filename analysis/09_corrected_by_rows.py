"""Corrected likelihood in registry data: accuracy for single terms by the number of posted rows.

Reads the per-term fits of 04_corrected_registry.py and summarizes, for the
1.7-or-more group and for all groups together, the error of each term's estimate
against the complete-data Mantel-Haenszel log risk ratio, by the number of rows
posted under the 5% any-arm rule.

Output: results/registry/corrected_by_posted_rows.json
"""

import numpy as np
import pandas as pd
from common import RESULTS, write_json

OUT = RESULTS / "registry"
BINS = [(1, 2), (3, 4), (5, 9), (10, 19), (20, 10**9)]

M = pd.read_csv(OUT / "corrected_registry_terms.csv")
M = M[np.isfinite(M.lr_corrected)]
res = {}
for name, g in [("1.7 or more", M[M.stratum == ">=1.7"]), ("all groups", M)]:
    block = []
    for a, b in BINS:
        x = g[(g.n_posted >= a) & (g.n_posted <= b)]
        ec, en = x.lr_corrected - x.lr_full, x.lr_post - x.lr_full
        block.append(
            dict(
                posted_rows=f"{a}-{b}" if b < 10**9 else f"{a}+",
                terms=int(len(x)),
                median_abs_error_corrected=float(ec.abs().median()),
                median_abs_error_naive=float(en.abs().median()),
                median_error_corrected=float(ec.median()),
                median_error_naive=float(en.median()),
                median_ci_width=float((x.hi - x.lo).median()),
                ci_contains_complete_data=float(((x.lo <= x.lr_full) & (x.lr_full <= x.hi)).mean()),
                sigma_at_bound=float(x.sigma_at_bound.astype(bool).mean()),
            )
        )
    res[name] = block
write_json(res, OUT / "corrected_by_posted_rows.json")
for k, v in res.items():
    print(k)
    print(pd.DataFrame(v).round(3).to_string(index=False))
