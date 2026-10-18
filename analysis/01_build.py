"""Build the analysis data set from the ClinicalTrials.gov extract in data/raw."""

import csv
import glob

import pandas as pd
from common import DERIVED, RAW, RESULTS, write_json


def num(s):
    return pd.to_numeric(s, errors="coerce")


T = pd.read_csv(RAW / "trials.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
R = pd.concat(
    [
        pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
        for f in sorted(glob.glob(str(RAW / "rows_*.tsv.gz")))
    ],
    ignore_index=True,
)
# Some registry terms carry literal double quotes; they are removed so that equal terms match.
R["term"] = R.term.str.replace('"', "", regex=False).str.strip()
log = {"trials": len(T)}
T["thr"] = num(T.threshold)
for col in [
    "drug_o_aff",
    "drug_o_risk",
    "pl_o_aff",
    "pl_o_risk",
    "drug_s_aff",
    "drug_s_risk",
    "pl_s_aff",
    "pl_s_risk",
    "enrollment",
]:
    T[col] = num(T[col])
T["ratio"] = T.drug_o_risk / T.pl_o_risk
log["threshold_missing"] = int(T.thr.isna().sum())
for col in ["xd", "nd", "xp", "np"]:
    R[col] = num(R[col])
log["rows"] = len(R)
bad = (
    R[["xd", "nd", "xp", "np"]].isna().any(axis=1)
    | (R.xd > R.nd)
    | (R.xp > R.np)
    | (R.nd <= 0)
    | (R.np <= 0)
    | (R.xd < 0)
    | (R.xp < 0)
)
log["rows_invalid_counts"] = int(bad.sum())
R = R[~bad].copy()
R["tot"] = R.xd + R.xp
log["duplicate_term_groups"] = int(
    R[R.duplicated(["nct", "type", "term"], keep=False)].groupby(["nct", "type", "term"]).ngroups
)
# A term listed more than once in a trial: keep the row with the most events,
# and among equal totals the first row in the registry record.
R = R.sort_values("tot", ascending=False, kind="mergesort").drop_duplicates(["nct", "type", "term"]).drop(columns="tot")
log["rows_after_cleaning"] = len(R)
DERIVED.mkdir(parents=True, exist_ok=True)
T.to_pickle(DERIVED / "trials.pkl")
R.to_pickle(DERIVED / "rows.pkl")
write_json(log, RESULTS / "registry" / "build_log.json")
print(log)
