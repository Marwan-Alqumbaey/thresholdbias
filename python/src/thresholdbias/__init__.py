"""thresholdbias: bias from adverse-event reporting thresholds.

Exact theory, pooled estimators and a selection-corrected likelihood for
adverse-event tables posted only when an arm exceeds a frequency threshold.
"""

from .core import bias_check, exact_conditional, iv_rr, kcross, mh_rr, posted_anyarm, posted_pooled
from .correct import fit_corrected

__version__ = "1.0.0"
__all__ = [
    "kcross",
    "posted_anyarm",
    "posted_pooled",
    "exact_conditional",
    "bias_check",
    "mh_rr",
    "iv_rr",
    "fit_corrected",
]
