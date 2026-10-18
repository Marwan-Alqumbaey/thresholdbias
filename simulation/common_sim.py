"""Shared settings for the simulation study."""

import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "simulation"
SD_LOGIT = 0.5


def write_json(obj, path):
    """Write ``obj`` as indented JSON, with undefined numbers as null."""

    def finite(o):
        if isinstance(o, dict):
            return {k: finite(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [finite(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o

    with open(path, "w") as fh:
        json.dump(finite(obj), fh, indent=1, allow_nan=False)


def corrected_scenarios():
    """The 20 scenarios used for the corrected estimators."""
    sc = [
        dict(K=30, ratio=r, N=300, pmed=p, rr=rr, c=0.05)
        for r, p, rr in itertools.product([(1, 1), (2, 1), (3, 1)], [0.02, 0.03, 0.04], [1.0, 1.5])
    ]
    sc += [
        dict(K=100, ratio=(2, 1), N=300, pmed=0.03, rr=1.0, c=0.05),
        dict(K=10, ratio=(2, 1), N=300, pmed=0.03, rr=1.0, c=0.05),
    ]
    return sc
