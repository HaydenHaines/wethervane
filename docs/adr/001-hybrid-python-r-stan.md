# ADR-001: Hybrid Python + R + Stan Technology Stack

## Status
Accepted

## Context
The US Political Covariation Model spans several computational domains, each with a different "best available" ecosystem:

- **Community detection and network analysis**: The leading libraries -- `leidenalg` (Leiden algorithm), `graph-tool` (stochastic block models), `igraph`, and `scikit-learn` (NMF) -- are Python-native or have Python as their primary interface. Python's scientific computing ecosystem (NumPy, pandas, SciPy, NetworkX) is the natural environment for data assembly and feature engineering.

- **Bayesian election modeling and MRP**: The closest existing election model to our design -- the Economist 2020 presidential model -- is written in R with Stan. The best existing MRP pipeline for CES data (`ccesMRPprep`, Kuriwaki) is R. The `brms` and `rstanarm` packages provide high-level interfaces for multilevel models. R's `tidyverse` ecosystem is strong for survey data manipulation.

- **Custom Bayesian models**: Stan is the standard language for specifying custom Bayesian models with efficient HMC sampling. Stan models are plain text `.stan` files that can be compiled and called from both Python (`CmdStanPy`) and R (`CmdStanR`).

We considered three alternatives:
1. **Python-only** (using `pystan` or `CmdStanPy` for Stan, reimplementing MRP tooling). Rejected because it would require reimplementing mature R packages and would diverge from the Economist model template.
2. **R-only** (using `reticulate` to call Python libraries from R). Rejected because `reticulate` adds friction and debugging complexity, and R is not the natural environment for network analysis at scale.
3. **Hybrid** (Python for data/detection, R for MRP/Bayesian, Stan shared). This incurs interface overhead but lets each component use its strongest tools.

## Decision
Adopt a hybrid stack:

- **Python** is the primary language for data assembly (`src/assembly/`), community detection (`src/detection/`), and visualization (`src/viz/`).
- **R** is used for MRP modeling, Bayesian covariance estimation (`src/covariance/`), and poll propagation (`src/propagation/`), leveraging existing packages and the Economist model as a template.
- **Stan** models are written as standalone `.stan` files in a shared directory, callable from both Python (via `CmdStanPy`) and R (via `CmdStanR`).
- Data exchange between Python and R stages uses Parquet files (readable by both `pyarrow` and `arrow` R package) and CSV for small datasets.
- The pipeline orchestration is Python-based, calling R scripts via subprocess or `rpy2` where needed.

## Consequences
**What becomes easier:**
- Each pipeline stage uses the best available tools and libraries without compromise.
- We can directly adapt the Economist Stan model and `ccesMRPprep` R pipeline rather than reimplementing them.
- Stan models are portable artifacts that can be used from either ecosystem.
- Contributors can work in whichever language they are strongest in, as long as they respect the data exchange interfaces.

**What becomes more difficult:**
- Developers need familiarity with both Python and R ecosystems.
- The data exchange interface (Parquet files, agreed-upon schemas) must be explicitly maintained and documented.
- Debugging cross-language issues (e.g., R subprocess failures, data type mismatches in Parquet) adds complexity.
- CI/CD must install and test both language environments.
- Dependency management spans two ecosystems (`requirements.txt` / `pyproject.toml` for Python, `renv.lock` for R).
