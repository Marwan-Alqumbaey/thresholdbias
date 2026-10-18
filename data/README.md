# Data

`raw/` holds the registry extract analysed in the paper: every interventional,
randomized study with posted results on ClinicalTrials.gov (API version 2.0.5, data of
30 September 2026) whose adverse-event section has exactly two groups, exactly one of
them with a title containing placebo, vehicle, sham, dummy or saline control.
`analysis/00_download.py` builds the same extract from the registry as it stands on the
day the script is run.

| Step | Studies |
|---|---:|
| Interventional, randomized, results posted | 48,077 |
| Exactly two adverse-event groups | 29,228 |
| Exactly one placebo-type group (analysed) | 9,414 |

## `trials.tsv.gz` (one row per trial)

| Column | Meaning |
|---|---|
| `nct` | ClinicalTrials.gov identifier |
| `threshold` | Frequency threshold for posting non-serious adverse events, in percent |
| `phases`, `sponsor_class`, `model`, `masking`, `purpose`, `enrollment`, `start` | Design fields as registered |
| `drug_o_aff`, `drug_o_risk` | Drug arm: participants with any non-serious event; participants at risk |
| `pl_o_aff`, `pl_o_risk` | Placebo arm: the same |
| `drug_s_aff`, `drug_s_risk`, `pl_s_aff`, `pl_s_risk` | The same for serious adverse events |
| `drug_title`, `pl_title` | Group titles |
| `conditions` | Conditions studied |

## `rows_1.tsv.gz` to `rows_5.tsv.gz` (one row per adverse-event term per trial)

| Column | Meaning |
|---|---|
| `nct` | Trial identifier |
| `type` | `O`, non-serious ("other") event; `S`, serious event |
| `term` | Event term, lower case |
| `organ` | Organ system, lower case |
| `assess` | Assessment type (systematic or non-systematic) |
| `xd`, `nd` | Drug arm: participants with the event; participants at risk |
| `xp`, `np` | Placebo arm: the same |

ClinicalTrials.gov data are in the public domain.
