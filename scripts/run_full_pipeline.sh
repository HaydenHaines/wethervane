#!/usr/bin/env bash
# Run the full analysis pipeline end-to-end
# Each stage reads from and writes to data/

set -euo pipefail

echo "=== Stage 1: Data Assembly ==="
python -m src.assembly.assemble_county
python -m src.assembly.assemble_networks

echo "=== Stage 2: Community Detection ==="
python -m src.detection.network_detection
python -m src.detection.nmf_detection
python -m src.detection.compare_methods
python -m src.detection.hierarchy
python -m src.detection.type_profiles

echo "=== Stage 3: Covariance Estimation ==="
python -m src.covariance.aggregate_to_types
python -m src.covariance.pca_factors
python -m src.covariance.fit_covariance
python -m src.covariance.stability_tests
Rscript src/covariance/variation_partition.R

echo "=== Stage 4: Validation (Baselines) ==="
python -m src.validation.baselines
Rscript src/validation/baseline_mrp.R

echo "=== Stage 5: Poll Propagation ==="
Rscript src/propagation/prepare_stan_data.R
Rscript src/propagation/fit_propagation.R

echo "=== Stage 6: MRP Integration ==="
Rscript src/propagation/mrp/fit_mrp.R
Rscript src/propagation/mrp/integrate_mrp.R

echo "=== Stage 7: Prediction ==="
python -m src.prediction.aggregate_predictions
python -m src.prediction.shift_decomposition
python -m src.prediction.turnout_decomposition

echo "=== Stage 8: Validation (Full) ==="
python -m src.validation.hindcast
python -m src.validation.metrics
python -m src.validation.falsification

echo "=== Pipeline complete ==="
