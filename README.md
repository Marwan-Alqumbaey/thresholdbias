# thresholdbias: bias from adverse-event reporting thresholds

[![Python package](https://github.com/Marwan-Alqumbaey/thresholdbias/actions/workflows/python.yml/badge.svg)](https://github.com/Marwan-Alqumbaey/thresholdbias/actions/workflows/python.yml)
[![R package](https://github.com/Marwan-Alqumbaey/thresholdbias/actions/workflows/r.yml/badge.svg)](https://github.com/Marwan-Alqumbaey/thresholdbias/actions/workflows/r.yml)
[![Reproduce](https://github.com/Marwan-Alqumbaey/thresholdbias/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Marwan-Alqumbaey/thresholdbias/actions/workflows/reproduce.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Code, data and software for the article

> Alqumbaey M, Nasher YM. Adverse-event reporting thresholds bias drug-placebo
> comparisons: exact theory, simulation, and a registry experiment. Manuscript submitted
> for publication.

ClinicalTrials.gov, and many journal articles, list a non-serious adverse event only if
it exceeds a frequency threshold in at least one arm (at most 5% on ClinicalTrials.gov).
Among the rows that are listed, the arm that crosses the threshold more easily has its
event rate pushed up more, which biases drug-placebo risk ratios pooled from posted
tables. This repository contains the exact theory checks, the simulation study, the
registry experiment, and Python and R packages that compute the expected distortion and
fit a selection-corrected likelihood.

## Repository layout

| Folder | Contents |
|---|---|
| `python/` | Python package `thresholdbias` with tests |
| `r/thresholdbias/` | R package `thresholdbias` with tests, including equivalence with Python |
| `data/raw/` | Frozen ClinicalTrials.gov extract used in the paper ([data dictionary](data/README.md)) |
| `analysis/` | Registry analyses, numbered in the order they run |
| `simulation/` | Simulation study |
| `figures/` | Figures 1 to 4 and the script that draws them |
| `results/` | All numerical results reported in the paper |
| `validation/` | Theory checks, Python reference values for the R tests, and the registry experiment repeated in R |

## Installation

Python (3.10 or later):

```
pip install ./python
```

R (4.0 or later):

```r
remotes::install_github("Marwan-Alqumbaey/thresholdbias", subdir = "r/thresholdbias")
```

## Quick example

```python
from thresholdbias import exact_conditional
r = exact_conditional(n1=200, n0=100, p1=0.03, p0=0.03, c=0.05)
print(round(r["p_post"], 3), round(r["rr_cond"], 3))   # 0.118 0.729
```

With 200 drug and 100 placebo participants, equal 3% risks and a 5% threshold, only
12% of rows are posted, and among them the risk ratio settles at 0.73 instead of 1.

To check the trials of a review, give the arm sizes of each trial and a range of
plausible event rates. `bias_check` returns the value that a Mantel-Haenszel risk ratio
pooled over posted rows would approach, divided by its value for complete tables:

```python
from thresholdbias import bias_check
r = bias_check(n1=[200, 150, 400, 90], n0=[100, 50, 200, 45], p0=[0.01, 0.02, 0.03], rr=1.0, c=0.05)
print(r["distortion"].round(2))   # [0.22 0.49 0.72]
```

Between-trial variation in the placebo risk (`sd_logit`) and in the log risk ratio
(`tau`) can be added. The R package has the same function:

```r
library(thresholdbias)
bias_check(n1 = c(200, 150, 400, 90), n0 = c(100, 50, 200, 45), p0 = c(0.01, 0.02, 0.03))
```

## Reproducing the paper

```
pip install -r requirements-lock.txt "./python[analysis,test]"
./run_all.sh
```

`run_all.sh` rebuilds the analysis data set from `data/raw`, runs every registry
analysis and the simulation study, and redraws the figures. The registry analyses take
about ten minutes. The corrected-likelihood steps are slow (about 4 to 5 CPU-hours for the
registry fits and about 14 CPU-hours for the simulation) and are split over `WORKERS`
processes (default 2; for example `WORKERS=8 ./run_all.sh`). The per-replicate results of
the corrected-likelihood simulations are kept in `results/simulation/corrected_replicates/`
and `results/simulation/robustness_replicates/`; scenarios already present there are not
recomputed, so delete those folders to rerun them.
All random seeds are fixed. With the package versions in `requirements-lock.txt` the
files in `results/` are reproduced exactly. `results/key_numbers.json` collects every number
quoted in the paper.

Checks:

```
# Python package and data parser
pytest python/tests analysis/tests

# R package, including equivalence with Python (needs testthat and jsonlite)
R CMD build r/thresholdbias && NOT_CRAN=true R CMD check --no-manual thresholdbias_1.0.0.tar.gz

# Registry experiment and three simulation cells repeated in R
R CMD INSTALL r/thresholdbias
Rscript validation/reproduce_registry.R
Rscript validation/simulation_spot_check.R

# Results of Supplementary Material S1
python validation/theory_checks.py

# Extract against values read again from the registry for 20 random trials
python validation/registry_spot_check.py
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Code: MIT. Data: ClinicalTrials.gov records are in the public domain.
