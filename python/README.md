# thresholdbias (Python)

Tools for adverse-event tables posted only when an arm exceeds a frequency
threshold, as on ClinicalTrials.gov.

```
pip install .
```

```python
from thresholdbias import bias_check, exact_conditional, fit_corrected, posted_anyarm

# Which rows would be posted under the 5% any-arm rule
posted_anyarm(x1=[6, 3], n1=[100, 100], x0=[0, 2], n0=[50, 50], c=0.05)   # [True, False]

# One trial with 200 drug and 100 placebo participants, both at 3% risk, 5% threshold
r = exact_conditional(200, 100, 0.03, 0.03, 0.05)
r["p_post"], r["rr_cond"]          # 0.118, 0.729

# Distortion expected for the arm sizes of the trials in a review, at three placebo risks
bias_check([200, 150, 400, 90], [100, 50, 200, 45], [0.01, 0.02, 0.03])["distortion"]   # 0.22, 0.49, 0.72

# Selection-corrected common log risk ratio from posted rows
import numpy as np
rng = np.random.default_rng(1)
p0 = np.exp(np.log(0.03) + 0.4 * rng.standard_normal(400))
x1, x0 = rng.binomial(200, p0), rng.binomial(100, p0)
keep = posted_anyarm(x1, 200, x0, 100, 0.05)
fit = fit_corrected(x1[keep], 200, x0[keep], 100, c=0.05)
fit["logRR"], fit["lo"], fit["hi"]   # 0.042, -0.058, 0.140
```

| Function | Purpose |
|---|---|
| `kcross(n, c, rule)` | Smallest event count that crosses the threshold |
| `posted_anyarm`, `posted_pooled` | Posting rules (any-arm rule; pooled-arm rule) |
| `exact_conditional` | Exact moments of a row given that it is posted |
| `bias_check` | Distortion expected for a set of trials, with optional between-trial variation |
| `mh_rr` | Mantel-Haenszel log risk ratio with Greenland-Robins standard error |
| `iv_rr` | Inverse-variance or DerSimonian-Laird log risk ratio |
| `fit_corrected` | Selection-corrected random-effects likelihood for a common risk ratio |

Thresholds are proportions (0.05 for 5%). The strict rule (`x/n > c`) is the
default; `rule="weak"` gives `x/n >= c`.

Install the test dependency with `pip install ".[test]"` and run `pytest tests`.
