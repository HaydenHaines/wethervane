# Resource Index

Stage key: `S1` Data Assembly · `S2` Community Detection · `S3` Covariance Estimation · `S4` Poll Propagation · `S5` Prediction · `S6` Validation · `ALL` Cross-cutting

Add an entry here **at the same time** as creating a reference file. See `GOVERNANCE.md` for rules.

---

## Cross-cutting (`ALL`)

| File | What it covers | Tags |
|------|---------------|------|
| [skills/anthropic-skills-guidance.md](skills/anthropic-skills-guidance.md) | Skill type taxonomy (9 types), key design tips, gotchas pattern, reference→skill pipeline | `ALL` |

---

## S1 — Data Assembly

| File | What it covers | Tags |
|------|---------------|------|
| [data-sources/census-acs-tract.md](data-sources/census-acs-tract.md) | ACS 5-year API, key tables, MOE flagging, vintage convention | `S1` |
| [data-sources/vest-election-crosswalk.md](data-sources/vest-election-crosswalk.md) | VEST precinct→block→tract crosswalk, download, aggregation strategy | `S1`, `S3` |

**Known gaps to fill as work begins:**
- Religious congregation survey methodology and access (ARDA)

---

## S2 — Community Detection

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- NMF vs. LDA tradeoffs for mixed-membership community detection
- Soft assignment implementation patterns (scikit-learn NMF)
- Feature normalization conventions for county-level data

---

## S3 — Covariance Estimation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- Stan hierarchical model patterns for covariance estimation
- cmdstanpy interface: data passing, sampling, diagnostics
- Covariance matrix parameterization in Stan (LKJ prior)

---

## S4 — Poll Propagation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- MRP methodology: multilevel model + poststratification frame construction
- brms/rstanarm syntax for MRP models
- Poststratification frame construction from ACS microdata

---

## S5 — Prediction / Interpretation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- Joint vote share + turnout prediction from shared community structure
- Uncertainty propagation from Stan posteriors through to county-level estimates

---

## S6 — Validation

| File | What it covers | Tags |
|------|---------------|------|
| *(none yet)* | | |

**Known gaps:**
- Holdout backtesting design for election prediction
- Calibration diagnostics for probabilistic forecasts
- Proper scoring rules for dual-output (vote share + turnout) predictions

---

## _deprecated

| File | What it covered | Why deprecated |
|------|----------------|----------------|
| *(none yet)* | | |
