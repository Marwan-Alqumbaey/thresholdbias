"""Bias map: expected distortion for the arm sizes of real registry trials.

For the trials that declared a 5% threshold, and for the threshold-0 trials of the
registry experiment, the exact theory (bias_check, Remark S1.6 of the Supplementary
Material) gives the value that a Mantel-Haenszel risk ratio pooled over the rows
posted under the 5% any-arm rule converges to, divided by its value for complete
tables. It is computed for each allocation group and for all trials together, for
placebo risks of 0.5% to 4% and risk ratios of 1 and 1.5: with the same risks in
every trial; with the placebo risk varying between trials (SD 0.5 on the logit
scale); and with, in addition, the log risk ratio varying between trials (SD 0.3).

Output: results/registry/bias_map.csv and bias_map.json
"""

import numpy as np
import pandas as pd
from common import RESULTS, STRATA, C, load, write_json
from thresholdbias import bias_check

OUT = RESULTS / "registry"
P0 = [0.005, 0.01, 0.02, 0.03, 0.04]
RRS = [1.0, 1.5]
HET = {"none": (0.0, 0.0), "placebo risk": (0.5, 0.0), "placebo risk and effect": (0.5, 0.3)}

rows = []
sizes = {}
for thr, label in [(5, "5% threshold"), (0, "threshold 0")]:
    t, _ = load(thr)
    t = t[(t.drug_o_risk > 0) & (t.pl_o_risk > 0)]
    sizes[label] = {
        g: dict(
            trials=int(len(x)),
            median_participants=float((x.drug_o_risk + x.pl_o_risk).median()),
            median_smaller_arm=float(np.minimum(x.drug_o_risk, x.pl_o_risk).median()),
        )
        for g, x in [("all", t)] + [(s, t[t.stratum == s]) for s in STRATA]
    }
    for g, x in [("all", t)] + [(s, t[t.stratum == s]) for s in STRATA]:
        for rr in RRS:
            for het, (sd, tau) in HET.items():
                r = bias_check(x.drug_o_risk.values, x.pl_o_risk.values, P0, rr=rr, c=C, sd_logit=sd, tau=tau)
                for i, p in enumerate(P0):
                    rows.append(
                        dict(
                            trials=label,
                            group=g,
                            n_trials=int(len(x)),
                            rr=rr,
                            variation=het,
                            p0=p,
                            rr_complete=float(r["rr_complete"][i]),
                            rr_posted=float(r["rr_posted"][i]),
                            distortion=float(r["distortion"][i]),
                            p_post=float(r["p_post"][i]),
                        )
                    )
D = pd.DataFrame(rows)
D.to_csv(OUT / "bias_map.csv", index=False)


def pick(trials, group, rr, variation, p0):
    x = D[(D.trials == trials) & (D.group == group) & (D.rr == rr) & (D.variation == variation) & (D.p0 == p0)]
    return float(x.distortion.iloc[0])


summary = {
    "sizes": sizes,
    "distortion_rr1_none": {
        lab: {g: {f"{p:g}": pick(lab, g, 1.0, "none", p) for p in P0} for g in ["all"] + STRATA}
        for lab in ["5% threshold", "threshold 0"]
    },
    "distortion_rr1.5_none": {
        lab: {g: {f"{p:g}": pick(lab, g, 1.5, "none", p) for p in P0} for g in ["all"] + STRATA}
        for lab in ["5% threshold", "threshold 0"]
    },
    "distortion_rr1_placebo_risk": {
        lab: {g: {f"{p:g}": pick(lab, g, 1.0, "placebo risk", p) for p in P0} for g in ["all"] + STRATA}
        for lab in ["5% threshold", "threshold 0"]
    },
    "distortion_rr1_placebo_risk_and_effect": {
        lab: {g: {f"{p:g}": pick(lab, g, 1.0, "placebo risk and effect", p) for p in P0} for g in ["all"] + STRATA}
        for lab in ["5% threshold", "threshold 0"]
    },
}
write_json(summary, OUT / "bias_map.json")
with pd.option_context("display.width", 200, "display.max_rows", 500):
    print(D.pivot_table(index=["variation", "trials", "rr", "group"], columns="p0", values="distortion").round(3))
