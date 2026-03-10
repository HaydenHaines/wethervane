# US Political Covariation Model

A Bayesian model that discovers latent community types from non-political data -- religious affiliation, class and occupation structure, neighborhood characteristics -- and estimates how those community types covary in their political behavior. By learning the covariance structure from historical elections, the model can propagate sparse polling information through communities that share social identity, producing county-level estimates of both vote share and turnout. The key premise is that communities sharing social structure will move together politically, even when they are geographically distant, and that this covariance is detectable without using political data to define the communities in the first place.

## Core Hypothesis

Political behavior at the community level is better predicted by shared social identity and behavioral patterns (religion, class, occupation, neighborhood type) than by geography or broad demographics alone. Communities that share these non-political characteristics will exhibit correlated political shifts -- a pattern that can be learned from historical data and exploited for prediction.

This hypothesis is **falsifiable by design**: the model discovers community types entirely from non-political data, then separately tests whether those types predict political covariance. If they do not, the hypothesis fails cleanly.

## Architecture

The system is a six-stage pipeline with a strict firewall between community detection and political modeling:

```
Data Assembly --> Community Detection --> Covariance Estimation --> Poll Propagation --> Prediction --> Validation
   (Python)          (Python)            (Python + Stan)           (R + Stan)          (Python)     (Python + R)
```

**Stage 1 -- Data Assembly.** Ingest and harmonize county-level data from the Census, American Community Survey, religious congregation surveys, and related public sources. Output: a clean county-by-feature matrix for FL, GA, and AL (226 counties).

**Stage 2 -- Community Detection.** Discover latent community types from the non-political feature matrix. Counties receive soft assignments (probability vectors across community types), not hard cluster labels, because real counties contain mixtures of community types.

**Stage 3 -- Covariance Estimation.** Using historical election returns, estimate how the discovered community types covary in their political behavior. This is where political data enters the model for the first time. Implemented in Stan for full Bayesian uncertainty quantification.

**Stage 4 -- Poll Propagation.** Propagate current polling data through the community covariance structure using multilevel regression and poststratification (MRP). When a poll captures opinion in one community type, the covariance structure informs estimates for related community types elsewhere. Implemented in R + Stan, leveraging R's mature MRP ecosystem.

**Stage 5 -- Prediction and Interpretation.** Combine propagated estimates with community-type assignments to produce county-level predictions. The model outputs two quantities jointly: **vote share** (partisan split) and **turnout** (participation rate).

**Stage 6 -- Validation.** Holdout backtesting against known election results, cross-validation, and calibration diagnostics. Designed to stress-test the model, not confirm it.

## Key Innovation: Dual Output

Most political prediction models estimate vote share only, treating turnout as exogenous or ignoring it. This model estimates vote share and turnout jointly through the same community covariance structure. This matters because turnout variation is one of the largest sources of prediction error in elections -- communities that shift in partisan preference often shift in participation simultaneously, and the two are driven by related social dynamics.

## Proof of Concept

The initial implementation covers **Florida, Georgia, and Alabama** (226 counties). This three-state region was chosen for its political diversity: major metro areas, rural counties, the Black Belt, retirement communities, college towns, military-adjacent communities, and Cuban-American enclaves. It is large enough to test the covariance structure meaningfully and small enough to iterate quickly.

## Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| Data assembly | Python (pandas, geopandas) | Ingestion, cleaning, harmonization |
| Community detection | Python (scikit-learn, custom) | Latent community type discovery |
| Covariance estimation | Python + Stan (cmdstanpy) | Bayesian covariance modeling |
| Poll propagation | R + Stan (cmdstanr, brms) | MRP and multilevel modeling |
| Prediction | Python | Combining estimates, generating outputs |
| Validation | Python + R | Backtesting, calibration, diagnostics |
| Visualization | Python (matplotlib, plotly) | Maps, diagnostics, results |

Stan serves as the bridge between ecosystems: the same `.stan` model files are called from both Python (via cmdstanpy) and R (via cmdstanr).

## Project Status

**Early-stage research.** Architecture is complete. Literature review is underway. Implementation has not started. The target is a functional prediction system by the **October 2026 midterm elections**.

## Repository Structure

```
docs/           Detailed documentation (architecture, assumptions, data sources, decisions)
research/       Literature review and methods research
src/            Source code organized by pipeline stage
data/           Data artifacts (gitignored)
notebooks/      Exploratory analysis
tests/          Test suite
scripts/        Utility scripts
```

See `docs/ARCHITECTURE.md` for the full technical specification. See `docs/ASSUMPTIONS_LOG.md` for explicit modeling assumptions and their status. See `research/` for literature review on voter stability, community detection methods, and cross-disciplinary approaches.

## License

To be decided.
