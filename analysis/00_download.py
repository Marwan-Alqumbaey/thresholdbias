"""Download the registry extract from the ClinicalTrials.gov API (version 2).

The frozen extract used in the paper (data of 30 September 2026) is in data/raw;
this script documents how it was made and can rebuild it from the live registry.
Selection: interventional, randomized studies with posted results whose
adverse-event section has exactly two groups, exactly one of them a placebo,
vehicle, sham, dummy or saline-control group.

Usage: python 00_download.py [--out DIR]   (default: data/raw_new in the repository root)
"""

import argparse
import gzip
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://clinicaltrials.gov/api/v2/studies"
QUERY = "AREA[HasResults]true AND AREA[DesignAllocation]RANDOMIZED AND AREA[StudyType]INTERVENTIONAL"
FIELDS = "NCTId,DesignModule,LeadSponsor,Phase,Condition,StartDate,CompletionDate,AdverseEventsModule"
PLACEBO = re.compile(r"placebo|vehicle|sham|dummy|saline control", re.I)
TRIAL_COLS = [
    "nct",
    "threshold",
    "phases",
    "sponsor_class",
    "model",
    "masking",
    "purpose",
    "enrollment",
    "start",
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
    "conditions",
]
ROW_COLS = ["nct", "type", "term", "organ", "assess", "xd", "nd", "xp", "np"]


def clean(s):
    return re.sub(r"[\t\r\n]+", " ", "" if s is None else str(s)).strip()


def blank(v):
    return "" if v is None else str(v)


def parse_study(s):
    """Return (trial_record, rows) for an eligible study, or None."""
    ae = (s.get("resultsSection") or {}).get("adverseEventsModule")
    if not ae or not ae.get("eventGroups"):
        return None
    gs = ae["eventGroups"]
    if len(gs) != 2:
        return None
    is_pl = [bool(PLACEBO.search(g.get("title") or "")) for g in gs]
    if sum(is_pl) != 1:
        return None
    ip = is_pl.index(True)
    idr = 1 - ip
    gp, gd = gs[ip], gs[idr]
    ps = s["protocolSection"]
    dm = ps.get("designModule") or {}
    di = dm.get("designInfo") or {}
    sponsor = (ps.get("sponsorCollaboratorsModule") or {}).get("leadSponsor")
    nct = ps["identificationModule"]["nctId"]
    trial = [
        nct,
        blank(ae.get("frequencyThreshold")),
        "/".join(dm.get("phases") or []),
        sponsor.get("class", "") if sponsor else "",
        di.get("interventionModel", ""),
        (di.get("maskingInfo") or {}).get("masking", ""),
        di.get("primaryPurpose", ""),
        blank((dm.get("enrollmentInfo") or {}).get("count")),
        ((ps.get("statusModule") or {}).get("startDateStruct") or {}).get("date", ""),
        blank(gd.get("otherNumAffected")),
        blank(gd.get("otherNumAtRisk")),
        blank(gp.get("otherNumAffected")),
        blank(gp.get("otherNumAtRisk")),
        blank(gd.get("seriousNumAffected")),
        blank(gd.get("seriousNumAtRisk")),
        blank(gp.get("seriousNumAffected")),
        blank(gp.get("seriousNumAtRisk")),
        clean(gd.get("title"))[:80],
        clean(gp.get("title"))[:80],
        clean("; ".join((ps.get("conditionsModule") or {}).get("conditions") or []))[:120],
    ]
    rows = []
    for typ, events in (("O", ae.get("otherEvents") or []), ("S", ae.get("seriousEvents") or [])):
        for e in events:
            st = {q.get("groupId"): q for q in e.get("stats") or []}
            a, b = st.get(gd["id"], {}), st.get(gp["id"], {})
            rows.append(
                [
                    nct,
                    typ,
                    clean(e.get("term")).lower(),
                    clean(e.get("organSystem")).lower(),
                    e.get("assessmentType", ""),
                    blank(a.get("numAffected")),
                    blank(a.get("numAtRisk")),
                    blank(b.get("numAffected")),
                    blank(b.get("numAtRisk")),
                ]
            )
    return trial, rows


def download(out):
    out.mkdir(parents=True, exist_ok=True)
    trials, rows, token = [], [], None
    base = (
        API
        + "?format=json&pageSize=1000&filter.advanced="
        + urllib.parse.quote(QUERY)
        + "&fields="
        + urllib.parse.quote(FIELDS)
    )
    while True:
        with urllib.request.urlopen(base + (f"&pageToken={token}" if token else ""), timeout=120) as fh:
            page = json.load(fh)
        for s in page.get("studies", []):
            res = parse_study(s)
            if res:
                trials.append(res[0])
                rows.extend(res[1])
        token = page.get("nextPageToken")
        if not token:
            break
    with gzip.open(out / "trials.tsv.gz", "wt") as fh:
        fh.write("\t".join(TRIAL_COLS) + "\n" + "".join("\t".join(r) + "\n" for r in trials))
    k = -(-len(rows) // 5)
    for i in range(5):
        with gzip.open(out / f"rows_{i + 1}.tsv.gz", "wt") as fh:
            fh.write("\t".join(ROW_COLS) + "\n" + "".join("\t".join(r) + "\n" for r in rows[i * k : (i + 1) * k]))
    print(f"{len(trials)} trials, {len(rows)} rows written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw_new"))
    download(Path(ap.parse_args().out))
