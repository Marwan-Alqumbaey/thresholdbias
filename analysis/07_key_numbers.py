"""Collect the key numbers reported in the paper into results/key_numbers.json."""

import json
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from common import DERIVED, RESULTS, load, write_json
from thresholdbias import exact_conditional

reg = RESULTS / "registry"
sim = RESULTS / "simulation"
k = {}


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def rnd(x, d):
    """Round half up to d decimals (values such as 0.385 print as 0.39)."""
    return str(Decimal(repr(round(float(x), 12))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def f3(x):
    return rnd(x, 3).replace("-", "\u2212")


def f2(x):
    return rnd(x, 2).replace("-", "\u2212")


def ci3(c):
    return f"{f3(c[0])} to {f3(c[1])}"


def pct_lower(x):
    return f"{round(100 * (1 - np.exp(x)))}"


# Theory
for name, args in {
    "t_21_4": (200, 100, 0.04, 0.04),
    "t_21_3": (200, 100, 0.03, 0.03),
    "t_21_2": (200, 100, 0.02, 0.02),
    "t_31_3": (225, 75, 0.03, 0.03),
    "t_11_rr15": (150, 150, 0.045, 0.03),
    "t_21_rr15": (200, 100, 0.045, 0.03),
    "t_100_99": (100, 99, 0.02, 0.02),
}.items():
    r = exact_conditional(*args, 0.05)
    k[name] = rnd(r["rr_cond"], 2)
    k[name + "_ppost"] = f"{round(100 * r['p_post'])}"
k["t_21_3_weak"] = rnd(exact_conditional(200, 100, 0.03, 0.03, 0.05, "weak")["rr_cond"], 2)
k["t_100_99_weak"] = rnd(exact_conditional(100, 99, 0.02, 0.02, 0.05, "weak")["rr_cond"], 2)

# Registry: data
b = read_json(reg / "build_log.json")
T = pd.read_pickle(DERIVED / "trials.pkl")
k["n_trials"] = f"{b['trials']:,}"
k["n_rows"] = f"{b['rows_after_cleaning']:,}"
k["n_thr0"] = f"{int((T.thr == 0).sum()):,}"
t0 = T[T.thr == 0]
k["n_cross"] = f"{int((t0.model == 'CROSSOVER').sum()):,}"
k["n_noratio"] = f"{int((~(np.isfinite(t0.ratio) & (t0.ratio > 0)) & (t0.model != 'CROSSOVER')).sum()):,}"
k["n_thr5"] = f"{len(load(5)[0]):,}"
P = read_json(reg / "rethreshold_primary.json")
k["n_sample"] = f"{sum(P['trials'].values()):,}"
k["n_sample_events"] = f"{sum(P['trials_with_events'].values()):,}"
for s, tag in [("<0.9", "lt09"), ("0.9-1.1", "eq"), ("1.1-1.7", "mid"), (">=1.7", "ge17")]:
    d = P[s]
    k[f"trials_{tag}"] = f"{P['trials'][s]:,}"
    k[f"trials_ev_{tag}"] = f"{P['trials_with_events'][s]:,}"
    k[f"terms_{tag}"] = f"{d['terms']}"
    k[f"est_{tag}"] = f"{d['estimable_post']}"
    k[f"shift_{tag}"] = f3(d["median_shift_post"])
    k[f"ci_{tag}"] = ci3(d["ci_post"])
    k[f"pool_{tag}"] = f3(d["median_shift_pool"])
    k[f"poolci_{tag}"] = ci3(d["ci_pool"])
    k[f"neg_{tag}"] = f"{round(100 * d['frac_negative_post'])}"
    k[f"posted_pct_{tag}"] = f"{round(100 * d['rows_posted_any'] / d['rows'])}"
    for m in ["ivw_mean", "median_ge5rows", "median_lt5rows", "median_nearnull", "median_inf_incl"]:
        if d[m]["est"] is not None:
            k[f"{m}_{tag}"] = f3(d[m]["est"])
            k[f"{m}_ci_{tag}"] = ci3(d[m]["ci"])
    k[f"sig_lost_{tag}"] = str(d["significance"]["lost"])
    k[f"sig_gained_{tag}"] = str(d["significance"]["gained"])
    k[f"dir_changed_{tag}"] = str(d["significance"]["direction_changed"])
    k[f"nonest_{tag}"] = str(d["n_nonestimable_post"])
    k[f"nonest_onearm_{tag}"] = str(d["n_one_arm_without_events"])
    k[f"nonest_norow_{tag}"] = str(d["n_no_posted_row"])
    k[f"n_inf_incl_{tag}"] = str(d["estimable_post"] + d["n_one_arm_without_events"])
    k[f"pool_terms_{tag}"] = str(d["estimable_pool"])
    k[f"n_common_{tag}"] = str(d["n_terms_common"])
    for m in ["median_post_common", "median_pool_common"]:
        if d[m]["est"] is not None:
            k[f"{m[7:]}_{tag}"] = f3(d[m]["est"])
            k[f"{m[7:]}_ci_{tag}"] = ci3(d[m]["ci"])
ge = P[">=1.7"]
k["n_drug_larger"] = f"{P['trials']['1.1-1.7'] + P['trials']['>=1.7']:,}"
tt = pd.read_csv(reg / "rethreshold_primary_terms.csv")
g17 = tt[(tt.stratum == ">=1.7") & np.isfinite(tt.shift_post)]
k["n_lt5_ge17"] = str(int((g17.nposted < 5).sum()))
k["n_ge5_ge17"] = str(int((g17.nposted >= 5).sum()))
k["n_near_ge17"] = str(int((g17.lr_full.abs() < 0.2).sum()))
t_, _ = load(0)
g_ = t_[t_.stratum == ">=1.7"]
k["med_pl_ge17"] = f"{g_.pl_o_risk.median():.0f}"
k["n_small_ge17"] = str(int(((g_.drug_o_risk < 20) | (g_.pl_o_risk < 20)).sum()))
k["pct_sham"] = f"{round(100 * t_.pl_title.str.contains('sham', case=False).mean())}"
k["pct_ge17"] = pct_lower(ge["median_shift_post"])
k["pct_lt5_ge17"] = pct_lower(ge["median_lt5rows"]["est"])
k["pct_ge5_ge17"] = rnd(100 * (1 - np.exp(ge["median_ge5rows"]["est"])), 1)
k["boot_p_ge17"] = "< 0.001" if ge["p_one_sided_post"] < 0.001 else rnd(ge["p_one_sided_post"], 3)
k["h2"] = f3(P["H2_difference"]["est"])
k["h2_ci"] = ci3(P["H2_difference"]["ci"])
rows_any = sum(P[s]["rows_posted_any"] for s in ["<0.9", "0.9-1.1", "1.1-1.7", ">=1.7"])
rows_pool = sum(P[s]["rows_posted_pooled"] for s in ["<0.9", "0.9-1.1", "1.1-1.7", ">=1.7"])
k["pool_fewer_pct"] = f"{round(100 * (1 - rows_pool / rows_any))}"
for tag in ["parallel", "verified", "min20", "c2", "minarm20", "minarm40", "smallarm20"]:
    d = read_json(reg / f"rethreshold_{tag}.json")[">=1.7"]
    k[f"sens_{tag}"] = f3(d["median_shift_post"])
    k[f"sens_{tag}_ci"] = ci3(d["ci_post"])
    k[f"sens_{tag}_terms"] = str(d["estimable_post"])
    k[f"sens_{tag}_pool"] = f3(d["median_shift_pool"])
    k[f"sens_{tag}_poolci"] = ci3(d["ci_pool"])
k["pct_smallarm20"] = pct_lower(read_json(reg / "rethreshold_smallarm20.json")[">=1.7"]["median_shift_post"])
pr = read_json(reg / "predicted_shift_primary.json")[">=1.7"]
k["pred_rho"] = rnd(pr["spearman"], 2)
k["pred_slope"] = rnd(pr["slope"], 2)
k["pred_median"] = f3(pr["median_predicted"])
k["pred_median_obs"] = f3(pr["median_observed"])
k["pred_median_minarm20"] = f3(read_json(reg / "predicted_shift_minarm20.json")[">=1.7"]["median_predicted"])
ob = read_json(reg / "observational.json")
k["obs_ns"] = f3(ob["nonserious >=1.7"]["median_difference"])
k["obs_ns_ci"] = ci3(ob["nonserious >=1.7"]["ci"])
k["obs_s"] = f3(ob["serious >=1.7"]["median_difference"])
k["obs_s_ci"] = ci3(ob["serious >=1.7"]["ci"])
a5 = ob["median_arm_sizes"][">=1.7 threshold 5"]
a0 = ob["median_arm_sizes"][">=1.7 threshold 0"]
k["obs_arms5"] = f"{a5['drug']:.0f} and {a5['placebo']:.0f}"
k["obs_arms0"] = f"{a0['drug']:.0f} and {a0['placebo']:.0f}"
pc = read_json(reg / "pooled_cutoffs.json")
k["pc_any"] = f"{pc['rows_posted_anyarm_5']:,}"
for c in ["0.035", "0.03"]:
    k[f"pc_{c}"] = f"{pc['pooled_' + c]['rows_posted']:,}"
k["pc_maxabs"] = rnd(max(abs(v) for c in ["0.035", "0.03"] for v in pc["pooled_" + c]["median_shift"].values()), 2)

# Simulation, naive methods (K = 30, N = 300, c = 5%)
S = pd.read_csv(sim / "naive_methods.csv")
x = S[(S.K == 30) & (S.N == 300) & (S.c == 0.05)]


def g(m, ratio, p, rr, col):
    return float(x[(x.method == m) & (x.ratio == ratio) & (x.pmed == p) & (x.rr == rr)][col].iloc[0])


for ratio, tag in [("1:1", "11"), ("2:1", "21"), ("3:1", "31")]:
    k[f"sim_bias_{tag}_2"] = f2(g("M1", ratio, 0.02, 1.0, "bias"))
    k[f"sim_cov_{tag}_2"] = rnd(g("M1", ratio, 0.02, 1.0, "cover"), 2)
    k[f"sim_cov_{tag}_3"] = rnd(g("M1", ratio, 0.03, 1.0, "cover"), 2)
k["sim_bias_11_rr15_2"] = f2(g("M1", "1:1", 0.02, 1.5, "bias"))
k["sim_bias_11_rr15_3"] = f2(g("M1", "1:1", 0.03, 1.5, "bias"))
k["sim_bias_31_rr15_2"] = f2(g("M1", "3:1", 0.02, 1.5, "bias"))
k["sim_m2_31_3"] = f2(g("M2", "3:1", 0.03, 1.0, "bias"))
k["sim_m1_31_3"] = f2(g("M1", "3:1", 0.03, 1.0, "bias"))
m7 = x[(x.method == "M7") & (x.n >= 500)]
k["sim_m7_max"] = rnd(m7.bias.abs().max(), 2)
k["n_scen_naive"] = f"{len(S.groupby(['K', 'ratio', 'N', 'pmed', 'rr', 'c'])):,}"
for K_ in [10, 30, 100]:
    row = S[(S.K == K_) & (S.N == 300) & (S.c == 0.05) & (S.ratio == "3:1") & (S.pmed == 0.02) & (S.rr == 1.0)]
    row = row[row.method == "M1"].iloc[0]
    k[f"sim_K{K_}_bias"] = f2(row.bias)
    k[f"sim_K{K_}_cov"] = rnd(row.cover, 2)
k["n_scen_posted"] = f"{len(S[S.c > 0].groupby(['K', 'ratio', 'N', 'pmed', 'rr', 'c'])):,}"

# Simulation, corrected methods
C = read_json(sim / "corrected_methods.json")
k["sb_min_common"] = str(min(c["common_replicates"] for c in C))
k["sb_m1_minbias"] = f2(min(c["results"]["M1"]["bias"] for c in C))
k["sb_m1_maxbias"] = f2(max(c["results"]["M1"]["bias"] for c in C))
k["sb_m1_mincov"] = rnd(min(c["results"]["M1"]["cover"] for c in C), 2)
for m in ["M5", "M6"]:
    k[f"sb_{m}_maxabs"] = rnd(max(abs(c["results"][m]["bias"]) for c in C), 3)
    k[f"sb_{m}_cov_lo"] = rnd(min(c["results"][m]["cover"] for c in C), 3)
    k[f"sb_{m}_cov_hi"] = rnd(max(c["results"][m]["cover"] for c in C), 3)
k["sb_fits"] = f"{sum(c['common_replicates'] for c in C):,}"
k["sb_sigma_bound"] = f"{sum(c['flags']['M5']['sigma_at_bound'] for c in C):,}"

# Registry: corrected likelihood
CR = read_json(reg / "corrected_registry.json")
R = CR[">=1.7"]
for tag in ["all", "lt5", "ge5"]:
    k[f"cr_{tag}"] = f3(R[tag]["corrected_vs_mh_all_rows"][0])
    k[f"cr_{tag}_ci"] = ci3(R[tag]["corrected_vs_mh_all_rows"][1:])
    k[f"cr_naive_{tag}"] = f3(R[tag]["naive"][0])
k["cr_same"] = f3(R["all"]["corrected_vs_same_model_all_rows"][0])
k["cr_cover"] = f"{round(100 * R['ci_contains_mh_all_rows'])}"
k["cr_mae_corr"] = rnd(R["median_abs_error_corrected"], 2)
k["cr_mae_naive"] = rnd(R["median_abs_error_naive"], 2)
k["cr_mid"] = f3(CR["1.1-1.7"]["all"]["corrected_vs_mh_all_rows"][0])
k["cr_mid_naive"] = f3(CR["1.1-1.7"]["all"]["naive"][0])
k["cr_eq"] = f3(CR["0.9-1.1"]["all"]["corrected_vs_mh_all_rows"][0])
k["cr_eq_naive"] = f3(CR["0.9-1.1"]["all"]["naive"][0])
k["cr_terms"] = str(R["all"]["terms"])
k["cr_terms_total"] = f"{sum(v['all']['terms'] for v in CR.values()):,}"
k["cr_sigma_bound"] = str(R["sigma_at_bound"])
k["cr_nonconv"] = str(R["not_converged"])

# Expected distortion for registry arm sizes (bias map)
BM = read_json(reg / "bias_map.json")
for lab, tl in [("5% threshold", "t5"), ("threshold 0", "t0")]:
    k[f"bm_{tl}_trials"] = f"{BM['sizes'][lab]['all']['trials']:,}"
    k[f"bm_{tl}_median_n"] = f"{BM['sizes'][lab]['all']['median_participants']:.0f}"
    for g, gl in [("all", "all"), (">=1.7", "17"), ("0.9-1.1", "eq")]:
        for p in ["0.01", "0.02", "0.03"]:
            k[f"bm_{tl}_{gl}_{p}"] = f2(BM["distortion_rr1_none"][lab][g][p])
k["bm_eq_max_pct"] = str(
    round(
        100
        * max(
            abs(v - 1) for lab in BM["distortion_rr1_none"] for v in BM["distortion_rr1_none"][lab]["0.9-1.1"].values()
        )
    )
)
k["bm_t5_17_0.02_placebo_var"] = f2(BM["distortion_rr1_placebo_risk"]["5% threshold"][">=1.7"]["0.02"])
k["bm_t5_17_0.01_placebo_var"] = f2(BM["distortion_rr1_placebo_risk"]["5% threshold"][">=1.7"]["0.01"])
k["bm_t5_eq_0.01_effect_var"] = f2(BM["distortion_rr1_placebo_risk_and_effect"]["5% threshold"]["0.9-1.1"]["0.01"])

# Corrected likelihood for single terms, by number of posted rows (1.7-or-more group)
CB = read_json(reg / "corrected_by_posted_rows.json")["1.7 or more"]
for b in CB:
    tag = b["posted_rows"].replace("-", "_").replace("+", "plus")
    k[f"cbr_{tag}_terms"] = str(b["terms"])
    k[f"cbr_{tag}_mae_corr"] = f2(b["median_abs_error_corrected"])
    k[f"cbr_{tag}_mae_naive"] = f2(b["median_abs_error_naive"])
    k[f"cbr_{tag}_med_corr"] = f2(b["median_error_corrected"])
    k[f"cbr_{tag}_med_naive"] = f2(b["median_error_naive"])
    k[f"cbr_{tag}_ciw"] = f2(b["median_ci_width"])

# Simulation, departures from the assumptions of the corrected likelihood
RB = read_json(sim / "robustness.json")
k["rob_min_common"] = str(min(c["common_replicates"] for c in RB))
for c in RB:
    for m in ["M0", "M1", "M5", "M6", "M7"]:
        k[f"rob{c['scenario']}_{m}_bias"] = f3(c["results"][m]["bias"])
        k[f"rob{c['scenario']}_{m}_cov"] = rnd(c["results"][m]["cover"], 2)

write_json(k, RESULTS / "key_numbers.json")
print(len(k), "numbers written")
