#!/usr/bin/env bash
# Reproduce every result and figure from the frozen registry extract.
# Set WORKERS to the number of processes for the corrected-likelihood steps (default 2).
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-2}"

cd analysis
python 01_build.py
python 02_rethreshold.py --tag primary
python 02_rethreshold.py --tag parallel --parallel
python 02_rethreshold.py --tag verified --verified
python 02_rethreshold.py --tag min20 --min-trials 20
python 02_rethreshold.py --tag c2 --c 0.02
python 02_rethreshold.py --tag minarm20 --min-arm 20
python 02_rethreshold.py --tag minarm40 --min-arm 40
python 02_rethreshold.py --tag smallarm20 --small-arm 20
python 03_predicted_shift.py --tag primary
python 03_predicted_shift.py --tag minarm20 --min-arm 20
python 04_corrected_registry.py --workers "$WORKERS"
python 05_observational.py
python 06_pooled_cutoffs.py
python 08_bias_map.py
python 09_corrected_by_rows.py

cd ../simulation
python naive_methods.py
pids=()
for ((i = 0; i < WORKERS; i++)); do
  python corrected_methods.py --chunk "$i" --nchunks "$WORKERS" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
python corrected_methods.py --summarize
pids=()
for ((i = 0; i < WORKERS; i++)); do
  python robustness.py --chunk "$i" --nchunks "$WORKERS" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
python robustness.py --summarize

cd ..
python figures/make_figures.py
(cd analysis && python 07_key_numbers.py)
