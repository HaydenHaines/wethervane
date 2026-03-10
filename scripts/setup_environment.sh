#!/usr/bin/env bash
# Setup script for the US Political Covariation Model
# Installs Python and R dependencies

set -euo pipefail

echo "=== Setting up Python environment ==="

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment and install Python dependencies
uv venv
uv pip install -e ".[dev]"

# Install CmdStan (required for cmdstanpy)
echo "=== Installing CmdStan ==="
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

echo "=== Setting up R environment ==="

# Check for R
if ! command -v Rscript &> /dev/null; then
    echo "R not found. Install R from https://cran.r-project.org/"
    echo "Skipping R setup."
else
    Rscript -e '
    # Install renv for dependency management
    if (!require("renv", quietly = TRUE)) install.packages("renv")

    # Install core R packages
    packages <- c(
        "cmdstanr",
        "brms",
        "tidycensus",
        "tidyverse",
        "sf",
        "tmap",
        "vegan",
        "surveillance",
        "arviz"
    )

    # Install from CRAN
    for (pkg in packages) {
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
            install.packages(pkg, repos = "https://cloud.r-project.org")
        }
    }

    # Install packages from GitHub
    if (!require("remotes", quietly = TRUE)) install.packages("remotes")
    remotes::install_github("kuriwaki/ccesMRPprep")
    remotes::install_github("kuriwaki/ccesMRPrun")

    # Install cmdstanr from Stan repo
    install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))

    cat("R packages installed successfully.\n")
    '
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Note: graph-tool must be installed separately via conda:"
echo "  conda install -c conda-forge graph-tool"
echo ""
echo "Activate the Python environment with: source .venv/bin/activate"
