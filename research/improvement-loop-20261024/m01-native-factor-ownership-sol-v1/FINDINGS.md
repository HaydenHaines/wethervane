# Actual bank coordinates correct the earlier 2024 ownership claim

**H0 and F0 have identical event membership in every supplied fold, including 2024: both have 387 events.** The earlier root note, independent Luna review and continuation-v49 prose claiming H0=357 and a 30-key difference are incorrect. Their pinned compatibility bytes (SHA256 `53337fcbd640c1ece9c31f4c2c44cf107bbdcd60bc5bee912716109a3806ba05`) already show both=387. No prior artifact was changed. This correction concerns membership; the arm laws and coefficient dimensions differ.

This packet verifies all 32 existing raw-bank byte hashes, reads only their small `observed_data/source_race`, `historical_logit` and coefficient coordinate arrays, and checks exact ordered event agreement against both compatibility metadata and the frozen experiment plan. Posterior sample arrays are never read. The training outcome factor recorded in each NetCDF is `historical_logit`; the pinned aggregate factory defines it as StudentT(5), location offset+design beta+shared cycle effect, scale tau_race (lines 650–675). This is direct code and stored-observation evidence for the **aggregate fundamentals bank law**, not authentication of a selected native count graph. There is no need to load posterior draws or run a fit to establish this.

| Target | Each arm's stored training events | Used calibration events | Shared in each arm | Selected residual rows |
|---|---:|---:|---:|---:|
| 2018 | 299 | 59 | 59 | 930 |
| 2020 | 329 | 85 | 85 | 1,218 |
| 2022 | 357 | 106 | 106 | 1,514 |
| 2024 | 387 | 106 | 106 | 1,514 |

`per-arm-event-ledger.csv` has 2,744 arm/event rows, with membership authenticated against stored raw-bank coordinates and separately labeled candidate marginal/conditional owners. These columns are explicitly candidates, with `complete_native_binding=missing`; they do not assert compiler receipts or empirical independence. Metadata and stored logit arrays do not authenticate upstream event labels, source vintages, respondent overlap, or actual inclusion of a training law in a later selected native Problem.

# What the real residual implementation establishes

The pinned preparation forms `logit(poll D/(D+R)) - logit(actual D/(D+R))` (`momentum-calibration-preparation-sol-v1/prepare.py:128`). The incumbent covariance source's `gaussian_error_objective` (lines 79–100) evaluates a normalized multivariate Gaussian residual negative log density plus a Gaussian penalty in log-scale coordinates. It reads supplied residuals and kernels, not an outcome marginal. The regularization omits an additive constant and is an optimization convention, not a complete normalized prior or integrated covariance law. Splitting cycles is an asserted working independence; shared dependence within cycles remains in the kernels. The independent component owns sampling/weighting/rounding error and movement owns the same-race Brownian kernel. Adding another copy requires explicit allocation.

`check.py` compiles the unmodified objective AST and invokes it at supplied positive scales, without importing or invoking its optimizer. For the first three actual selected rows of each fold it reconstructs the five documented kernels, checks the Gaussian density against SciPy, finite-differences all five log-scale derivatives (maximum error 4.27e-10), and integrates the conditional Gaussian in whitened coordinates by tensor Gauss-Hermite quadrature (normalization error below 2e-15). The same first three records recur across folds, so four passes are not four independent scientific validations. These are deliberately small actual-data supplied-state checks, not full-vector or fitted-law evidence.

The Gaussian is normalized on **poll-logit measure**. A poll-share density requires the logit Jacobian, which the check records. Conditional use of a fixed observed outcome shifts its mean and does not introduce another marginal outcome density. This supports a possible working conditional adapter; it does not demonstrate the native count graph has adopted that adapter, that the independence assumptions hold, or that the outcome is fixed rather than uncertain there. A small ownership-contract check rejects two marginal owners and missing conditioned-event bindings; this is a supplied contract test, not actual native admission.

# Exact missing binding

The owner coordination file explicitly says no M01 binding, mixture implementation or fit is under way in that session. The inspected native `Measurement.observations` uses fixed block covariance and claims `poll:observation_id` tokens (components.py:541–542); native official counts claim `official:id` (438–443). Neither those tokens nor aggregate-bank event IDs link the aggregate `historical_logit` observations to a canonical native official-count atom, calibrator conditional, selected parent bank, or complete shared poll-vector owner. A poll token alone cannot represent the historical outcome labels used to derive its calibration bank.

A complete selected M01 Problem was not supplied or established by these artifacts. Missing are its immutable full provider/configuration manifest and source hashes; both arms' bank-to-provider law bindings; complete vector order and shared calibration component identity; canonical links between training official labels, conditional calibration uses and native official-count factors; a declared normalized conditional law with its conditioning variables, covariance uncertainty and cycle/shared-error assumptions; compiler receipts demonstrating one marginal outcome owner and one variance budget; and preserved protected-parent masses and conditional normalizers. Numerical equality on a small extracted method cannot fill these gaps. The existing evidence validator remains unchanged and admits no real ledger here.

Reproduce from the lab root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.improvement-ledger/runtime/calibration-v1-env/bin/python \
research/improvement-loop-20261024/m01-native-factor-ownership-sol-v1/check.py
```

The command reads existing sources and bank files and rewrites only this directory's ledger and `result.json`, which pins exact bytes, chain receipts and check values. No sampler, optimization, scientific fit, scoring, source admission, registration or promotion occurs. The project refine-model-layer and verify-election-model skills were read and applied to distinguish supplied-state arithmetic from retrospective predictive qualification. No web evidence is needed for this source-local implementation audit.
