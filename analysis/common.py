"""Shared paths, data loading and helpers for the registry analyses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from thresholdbias import kcross

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"

C = 0.05
STRATA = ["<0.9", "0.9-1.1", "1.1-1.7", ">=1.7"]


def stratum(ratio):
    """Allocation group from the drug:placebo ratio of participants at risk."""
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    if ratio < 0.9:
        return "<0.9"
    if ratio <= 1.1:
        return "0.9-1.1"
    if ratio < 1.7:
        return "1.1-1.7"
    return ">=1.7"


def load(threshold=0, exclude_crossover=True, parallel_only=False):
    """Trials with a given declared threshold (percent) and their adverse-event rows."""
    T = pd.read_pickle(DERIVED / "trials.pkl")
    R = pd.read_pickle(DERIVED / "rows.pkl")
    t = T[T.thr == threshold] if threshold is not None else T
    if exclude_crossover:
        t = t[t.model != "CROSSOVER"]
    if parallel_only:
        t = t[t.model == "PARALLEL"]
    t = t[np.isfinite(t.ratio) & (t.ratio > 0)].copy()
    t["stratum"] = t.ratio.map(stratum)
    return t, R[R.nct.isin(t.nct)].merge(t[["nct", "stratum", "ratio", "thr"]], on="nct")


def post_flags(r, c=C):
    """Posting indicators of each row under the any-arm and pooled-arm rules (strict)."""
    nd, np_ = r.nd.values.astype(int), r.np.values.astype(int)
    xd, xp = r.xd.values, r.xp.values
    kd = np.array([kcross(a, c) for a in nd])
    kp = np.array([kcross(a, c) for a in np_])
    kpool = np.array([kcross(a, c) for a in nd + np_])
    return (xd >= kd) | (xp >= kp), (xd + xp) >= kpool


def mh_parts(r):
    """Per-row Mantel-Haenszel numerator, denominator and Greenland-Robins variance terms."""
    N = r.nd + r.np
    R_ = r.xd * r.np / N
    S_ = r.xp * r.nd / N
    P_ = (r.nd * r.np * (r.xd + r.xp) - r.xd * r.xp * N) / N**2
    return R_.values, S_.values, P_.values


def _finite(obj):
    """Replace undefined numbers (NaN, infinity) by None so that the output is valid JSON."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(obj, path):
    """Write ``obj`` as indented JSON (undefined numbers as null), creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_finite(obj), fh, indent=1, allow_nan=False)
