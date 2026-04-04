#!/usr/bin/env bash
# Wethervane model retraining pipeline automation
# Runs the full end-to-end retrain in correct order with error handling
#
# Usage:
#   ./scripts/retrain_pipeline.sh              # Run full pipeline
#   ./scripts/retrain_pipeline.sh --help       # Show help
#   ./scripts/retrain_pipeline.sh --skip-features   # Skip feature build (step 1)
#   ./scripts/retrain_pipeline.sh --skip-discovery  # Skip discovery+description (steps 2-5)

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
SKIP_FEATURES=0
SKIP_DISCOVERY=0
HELP=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-features)
      SKIP_FEATURES=1
      echo -e "${YELLOW}Skipping step 1: feature building${NC}"
      shift
      ;;
    --skip-discovery)
      SKIP_DISCOVERY=1
      echo -e "${YELLOW}Skipping steps 2-5: type discovery and description${NC}"
      shift
      ;;
    --help)
      HELP=1
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

if [ $HELP -eq 1 ]; then
  cat << 'EOF'
Wethervane Model Retraining Pipeline

Usage:
  ./scripts/retrain_pipeline.sh [OPTIONS]

Options:
  --skip-features      Skip step 1 (feature building). Use when only model
                       parameters changed, not feature data.
  --skip-discovery     Skip steps 2-5 (type discovery and description).
                       Use when only Ridge/ensemble parameters changed.
  --help              Show this help message.

Pipeline Steps:
  1. Build county features (national)
  2. Type discovery with PCA whitening and KMeans
  3. Describe types with demographics
  4. Generate display names for types
  5. Construct observed Ledoit-Wolf covariance matrix
  6. Train Ridge model on pruned features
  7. Train Ridge+HGB ensemble model
  8. Generate 2026 type predictions
  9. Validate types and report metrics
  10. Rebuild DuckDB database

Each step must complete successfully before the next begins. If any step fails,
the pipeline stops and reports the error.

Examples:
  # Full retrain
  ./scripts/retrain_pipeline.sh

  # Retrain only Ridge model (types unchanged)
  ./scripts/retrain_pipeline.sh --skip-discovery

  # Rebuild features and retrain all models
  ./scripts/retrain_pipeline.sh

EOF
  exit 0
fi

# Timing tracking
PIPELINE_START=$(date +%s%N)

# Function to run a pipeline step
run_step() {
  local step_num=$1
  local step_name=$2
  local step_cmd=$3

  local step_start=$(date +%s%N)
  echo ""
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}Step ${step_num}: ${step_name}${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo "Command: ${step_cmd}"
  echo ""

  if ! eval "${step_cmd}"; then
    local step_end=$(date +%s%N)
    local step_elapsed=$(( (step_end - step_start) / 1000000 ))
    echo ""
    echo -e "${RED}FAILED at step ${step_num}: ${step_name}${NC}"
    echo "Elapsed time: ${step_elapsed}ms"
    exit 1
  fi

  local step_end=$(date +%s%N)
  local step_elapsed=$(( (step_end - step_start) / 1000000 ))
  local step_sec=$(( step_elapsed / 1000 ))
  local step_ms=$(( step_elapsed % 1000 ))

  echo ""
  echo -e "${GREEN}✓ Step ${step_num} complete${NC}"
  echo -e "${GREEN}Time: ${step_sec}s ${step_ms}ms${NC}"
}

# Verify we're in the project root
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
  echo -e "${RED}Error: Not in wethervane project root${NC}"
  echo "Run this script from /home/hayden/projects/wethervane"
  exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   WetherVane Model Retraining Pipeline                 ║${NC}"
echo -e "${BLUE}║   Starting at $(date '+%Y-%m-%d %H:%M:%S')                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

# Step 1: Build county features
if [ $SKIP_FEATURES -eq 0 ]; then
  run_step 1 "Build County Features (National)" \
    "uv run python -m src.assembly.build_county_features_national"
else
  echo -e "${YELLOW}Skipping Step 1: Build County Features${NC}"
fi

# Steps 2-5: Type discovery and description
if [ $SKIP_DISCOVERY -eq 0 ]; then
  run_step 2 "Type Discovery (KMeans + PCA Whitening)" \
    "uv run python -m src.discovery.run_type_discovery"

  run_step 3 "Describe Types (Demographics Overlay)" \
    "uv run python -m src.description.describe_types"

  run_step 4 "Name Types (Generate Display Names)" \
    "uv run python -m src.description.name_types"

  run_step 5 "Construct Type Covariance (Observed LW)" \
    "uv run python -m src.covariance.construct_type_covariance"
else
  echo -e "${YELLOW}Skipping Steps 2-5: Type Discovery and Description${NC}"
fi

# Step 6: Train Ridge model
run_step 6 "Train Ridge Model (Pruned Features)" \
  "uv run python -m src.prediction.train_ridge_model"

# Step 7: Train ensemble model
run_step 7 "Train Ridge+HGB Ensemble Model" \
  "uv run python -m src.prediction.train_ensemble_model"

# Step 8: Generate 2026 predictions
run_step 8 "Generate 2026 Type Predictions" \
  "uv run python -m src.prediction.predict_2026_types"

# Step 9: Validate types and extract metrics
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 9: Validate Types and Report Metrics${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Command: uv run python -m src.validation.validate_types"
echo ""

VALIDATE_START=$(date +%s%N)
VALIDATE_OUTPUT=$(mktemp)

if uv run python -m src.validation.validate_types > "${VALIDATE_OUTPUT}" 2>&1; then
  VALIDATE_END=$(date +%s%N)
  VALIDATE_ELAPSED=$(( (VALIDATE_END - VALIDATE_START) / 1000000 ))
  VALIDATE_SEC=$(( VALIDATE_ELAPSED / 1000 ))
  VALIDATE_MS=$(( VALIDATE_ELAPSED % 1000 ))

  cat "${VALIDATE_OUTPUT}"

  echo ""
  echo -e "${GREEN}✓ Step 9 complete${NC}"
  echo -e "${GREEN}Time: ${VALIDATE_SEC}s ${VALIDATE_MS}ms${NC}"

  # Extract key metrics from validation output for summary
  HOLDOUT_R=$(grep -oP "(?<=Holdout r:?\s)\d+\.\d+" "${VALIDATE_OUTPUT}" | head -1 || echo "N/A")
  LOO_R=$(grep -oP "(?<=LOO r:?\s)\d+\.\d+" "${VALIDATE_OUTPUT}" | head -1 || echo "N/A")
  COHERENCE=$(grep -oP "(?<=Coherence:?\s)\d+\.\d+" "${VALIDATE_OUTPUT}" | head -1 || echo "N/A")
  RMSE=$(grep -oP "(?<=RMSE:?\s)\d+\.\d+" "${VALIDATE_OUTPUT}" | head -1 || echo "N/A")
else
  echo -e "${RED}FAILED at step 9: Validate Types${NC}"
  cat "${VALIDATE_OUTPUT}"
  rm -f "${VALIDATE_OUTPUT}"
  exit 1
fi

rm -f "${VALIDATE_OUTPUT}"

# Step 10: Rebuild DuckDB
run_step 10 "Rebuild DuckDB Database" \
  "uv run python src/db/build_database.py --reset"

# Calculate total elapsed time
PIPELINE_END=$(date +%s%N)
PIPELINE_TOTAL=$(( (PIPELINE_END - PIPELINE_START) / 1000000 ))
PIPELINE_SEC=$(( PIPELINE_TOTAL / 1000 ))
PIPELINE_MIN=$(( PIPELINE_SEC / 60 ))
PIPELINE_SEC=$(( PIPELINE_SEC % 60 ))

# Print summary
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Retrain Pipeline Complete                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Pipeline Status: SUCCESS${NC}"
echo ""

if [ $SKIP_FEATURES -eq 0 ]; then
  echo "Steps Run: All 10 (features + discovery + training + validation)"
elif [ $SKIP_DISCOVERY -eq 0 ]; then
  echo "Steps Run: 1-10 (full pipeline)"
else
  echo "Steps Run: 1, 6-10 (training only, features skipped)"
fi

echo ""
echo "Key Metrics from Validation:"
if [ "${HOLDOUT_R}" != "N/A" ]; then
  echo "  Holdout r:  ${HOLDOUT_R}"
fi
if [ "${LOO_R}" != "N/A" ]; then
  echo "  LOO r:      ${LOO_R}"
fi
if [ "${COHERENCE}" != "N/A" ]; then
  echo "  Coherence:  ${COHERENCE}"
fi
if [ "${RMSE}" != "N/A" ]; then
  echo "  RMSE:       ${RMSE}"
fi

echo ""
echo -e "${GREEN}Total Elapsed Time: ${PIPELINE_MIN}m ${PIPELINE_SEC}s${NC}"
echo -e "${GREEN}Finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo ""
echo "Next steps:"
echo "  1. Review metrics above against baseline (see CLAUDE.md)"
echo "  2. Check API endpoints: GET /counties/{fips}/prediction/2026"
echo "  3. Run tests: uv run pytest"
echo "  4. Commit changes: git add -A && git commit -m 'Retrain: <REASON>'"
echo ""
