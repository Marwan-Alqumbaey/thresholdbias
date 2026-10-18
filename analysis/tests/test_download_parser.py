"""The API parser reproduces the frozen extract for a sample of studies."""

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("download", HERE.parent / "00_download.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)
RAW = HERE.parents[1] / "data" / "raw"


def test_parser_matches_frozen_extract():
    studies = json.load(gzip.open(HERE / "fixtures" / "api_studies.json.gz", "rt"))["studies"]
    T = pd.read_csv(RAW / "trials.tsv.gz", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    R = pd.concat(
        [
            pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
            for f in sorted(RAW.glob("rows_*.tsv.gz"))
        ]
    )
    checked = 0
    for s in studies:
        res = download.parse_study(s)
        assert res is not None
        trial, rows = res
        nct = trial[0]
        ref_t = T[T.nct == nct]
        assert len(ref_t) == 1
        cols = [
            "threshold",
            "drug_o_aff",
            "drug_o_risk",
            "pl_o_aff",
            "pl_o_risk",
            "drug_s_aff",
            "drug_s_risk",
            "pl_s_aff",
            "pl_s_risk",
            "drug_title",
            "pl_title",
        ]
        got = dict(zip(download.TRIAL_COLS, trial))
        for c in cols:
            assert got[c] == ref_t.iloc[0][c], (nct, c)
        ref_r = R[R.nct == nct][["type", "term", "xd", "nd", "xp", "np"]]
        got_r = pd.DataFrame(rows, columns=download.ROW_COLS)[["type", "term", "xd", "nd", "xp", "np"]]
        assert sorted(map(tuple, got_r.values)) == sorted(map(tuple, ref_r.values)), nct
        checked += 1
    assert checked == len(studies) >= 20
