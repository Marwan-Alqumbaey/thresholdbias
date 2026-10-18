# Validation

| File | Purpose |
|---|---|
| `theory_checks.py` | Checks the results of Supplementary Material S1 against exact double sums, exact rational arithmetic and Monte Carlo |
| `theory_checks_output.txt` | Output of `theory_checks.py` |
| `make_reference.py` | Writes the Python reference values used by the R package's equivalence tests |
| `reproduce_registry.R` | Repeats the registry experiment in R from the raw extract and compares every term with the Python results |
| `simulation_spot_check.R` | Repeats three cells of the simulation study in R and compares bias and coverage with the Python results |
| `registry_spot_check.py` | Compares the extract with values read again from the registry for 20 randomly drawn trials (files in `registry_spot_check/`) |
