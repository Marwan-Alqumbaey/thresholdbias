"""Compare the frozen extract with values read again from ClinicalTrials.gov.

Twenty trials drawn at random from the registry experiment were retrieved again
from the registry (API version 2) after the extract was made. The folder registry_spot_check
holds the response for the trial-level fields of all 20 trials and the non-serious
event rows of four of them. This script checks every value against data/raw.

Usage: python validation/registry_spot_check.py
"""

import csv
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHECK = ROOT / "validation" / "registry_spot_check"
PLACEBO = re.compile(r"placebo|vehicle|sham|dummy|saline control", re.I)


def read(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


trials = read(RAW / "trials.tsv.gz").set_index("nct")
rows = pd.concat([read(f) for f in sorted(RAW.glob("rows_*.tsv.gz"))])

with open(CHECK / "trials.json") as fh:
    studies = json.load(fh)["studies"]
n_values = 0
for s in studies:
    nct = s["protocolSection"]["identificationModule"]["nctId"]
    ae = s["resultsSection"]["adverseEventsModule"]
    placebo = [g for g in ae["eventGroups"] if PLACEBO.search(g["title"])]
    drug = [g for g in ae["eventGroups"] if not PLACEBO.search(g["title"])]
    assert len(placebo) == 1 and len(drug) == 1, nct
    expected = {"threshold": ae["frequencyThreshold"], "drug_title": drug[0]["title"], "pl_title": placebo[0]["title"]}
    for arm, g in [("drug", drug[0]), ("pl", placebo[0])]:
        expected[f"{arm}_o_aff"] = g["otherNumAffected"]
        expected[f"{arm}_o_risk"] = g["otherNumAtRisk"]
        expected[f"{arm}_s_aff"] = g["seriousNumAffected"]
        expected[f"{arm}_s_risk"] = g["seriousNumAtRisk"]
    for col, value in expected.items():
        assert trials.loc[nct, col].strip() == str(value).strip(), (nct, col)
        n_values += 1
print(f"Trial-level values: {len(studies)} trials, {n_values} values, all equal to the extract.")

events = pd.read_csv(CHECK / "other_events.csv", dtype=str, keep_default_na=False)
n_values = 0
for nct, ev in events.groupby("nct"):
    drug_first = not PLACEBO.search(ev.group1_title.iloc[0])
    d, p = ("group1", "group2") if drug_first else ("group2", "group1")
    expected = sorted(
        zip(ev.term.str.lower(), ev[f"{d}_affected"], ev[f"{d}_at_risk"], ev[f"{p}_affected"], ev[f"{p}_at_risk"])
    )
    ext = rows[(rows.nct == nct) & (rows.type == "O")]
    assert sorted(zip(ext.term, ext.xd, ext.nd, ext.xp, ext.np)) == expected, nct
    n_values += 4 * len(expected)
print(f"Non-serious event rows: {events.nct.nunique()} trials, {len(events)} rows, {n_values} counts, all equal.")
