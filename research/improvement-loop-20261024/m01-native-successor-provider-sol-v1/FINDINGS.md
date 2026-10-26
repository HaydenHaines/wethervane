# Lab-only actual native measurement successor

`provider.py` implements a separately versioned measurement provider against the actual SHA-bound prepared-v4 TX Problem and original eight population parents. It builds through the unmodified native Registry, typed port compiler and once-only observation gates, inside a local test registry only. It neither installs nor changes an owner provider. Its selected source configuration is computational research, not source/scientific admission, posterior inference or a 2026 forecast.

The provider delegates the incumbent `Measurement.observations` unchanged for all CES/report paths using a copied BuildContext with an empty poll list, then adds one `M01_complete_poll_vector` Potential. The new factor claims the exact six native poll tokens once and uses named candidate `category_rates` to form log(first rate)−log(second rate). The existing MvNormal poll factor is omitted. A single common calibration row governs the entire six-row vector; no independent per-poll mixtures, extra MOE/sampling term or extra Brownian movement kernel is introduced.

The actual bank contains all 4,000 joint five-scale rows, with their original uniform 1/4000 masses and original flatten order. The provider's reconstructed covariance and weight arrays are compared **bit for bit across every row** with `load_frozen_current_case(...)["mixture"]`, loaded independently in a clean calibration-runtime process. The reference is saved in `reference-mixture.npz` and SHA-bound by the selection; this catches kernel/order/weight errors that a SciPy oracle built from the provider's own matrices could miss. `reference-check.py` reproduces that clean-process reference and equality check.

`selection.json` binds actual Problem bytes/identity, exact native↔archive IDs through the earlier authenticated source map, complete poll covariance, all eight full parent states/IDs/masses, calibration source snapshots and numerical helper dependencies. Its expected SHA is a literal in `provider.py`, and is included in the ComponentSpec configuration/dependency receipts. A coherent selection rewrite is rejected before JSON parsing. This is an independently frozen local computational configuration, not an external scientific approval or tamper-proof storage anchor. Rewriting both source and selection remains outside that trust boundary.

The entire Problem identity must match. Extra blocks, duplicate rows, changed observations, reordered poll/candidate menus and doubled incumbent covariance fail closed. Every selected parent must exactly match its original complete state, including original mass 0.125; changes to the declared single total-error variance owner fail. These restrictive checks intentionally support only this authenticated actual fixture. Candidate reorderings are rejected rather than silently reinterpreted. Arbitrary new Problems require a separately versioned selection.

# Numerical and compiler scope

The executable builds the complete 662-dimensional actual role-aware graph at the first protected population parent, preserving the other seven providers and the incumbent survey/report observation methods. It tests single-component reduction at the actual graph's initial state against the normalized six-dimensional Gaussian density. It also builds the actual 4,000-component successor and compares its poll density at the same actual rate-derived means against an independent direct SciPy vector logsumexp, plus symbolic versus finite-difference mean gradients. All vector rows share one component index.

The check additionally evaluates the **entire** joint density as the complete native `compile_logp(sum=False)` list, numerically summing every contribution, including transformed-variable Jacobians and all incumbent factors. It checks that switching single-component to bank mixture changes this full density by precisely the poll-factor difference. Both models use identical transformed initial coordinates; no random prior/predictive states are substituted. Full-gradient and selected-coordinate finite-difference status is recorded explicitly in `result.json`; a poll-mean derivative alone does not establish the full joint gradient.

The original eight population masses are checked and remain outside each conditional logp. This is **not** a proof that all eight downstream conditionals normalize, a quadrature convergence result or a validation of posterior population-cut inference. Only the first parent's graph is numerically exercised. No normalization claim is inferred from finite fixed-state values or the prior tautological Gaussian-whitening exercise.

Compiler evidence tokens still describe literal native observations. Calibration training outcome labels remain conditioned provenance, not invented native tokens. The adapter adopts the declared retrospective normalized Gaussian logit working law and marginal-Bayes feedback within each population conditional for computational study. It does not establish respondent independence, original publication/vintage, common-error adequacy or empirical admissibility. TX2020/TX2024 calibration event exclusions are retained. Its bank uncertainty is integrated through one complete-vector finite mixture; it does not silently become a protected calibration cut. No separately fitted H0/F0 coefficient bank is spliced into the native fundamentals provider.

# Runtime failures preserved

`failures.json` records each failure and response. The original calibration environment lacks pydantic, and the owner virtualenv's PyMC import fails because its ArviZ lacks InferenceData. No environment was modified. A lab-only `schema-overlay` exposes only existing pydantic, pydantic_core, annotated_types and typing_inspection packages/metadata through read-only symlinks. NumPy, SciPy, PyMC, PyTensor and ArviZ remain from the established calibration environment; actual versions/import paths are recorded.

The first numerical harness incorrectly compiled a raw Potential, retaining RandomVariable nodes and failing on the native Flat variable's unavailable random generator. No fitting or sampler was invoked. The corrected harness uses native logp/value-variable replacement. An artifact serialization failure was also corrected without a model change.

With C compilation disabled, the full scalar joint density and later its full derivative encountered PyTensor's missing ufunc for fused additions with more than 32 inputs. Exact traces are retained in `full-joint-first-failure.log` and `full-gradient-first-failure.log`. The full logp is evaluated as its complete contribution list. A narrowly targeted exclusion of `local_add_canonizer` did not resolve the gradient failure; its exact trace is retained in `full-gradient-excluded-rewrite-failure.log`. Further C/compiler attempts were stopped. The final executable evaluates the complete native factor-list joint density but explicitly does **not** compile a full 662-coordinate gradient. It retains the poll-mean analytic/finite-difference check and central differences of the full joint density for two intercepts at two step sizes; these do not establish the unavailable full gradient. Final partial status is preserved in `check.log` and `result.json`.

All runs use nice15 and one BLAS/OMP/MKL/NumExpr thread, with compiler cache confined to this new directory. No C compiler, sampler, optimizer, owner file, owner job, global registry, source admission or public output is changed.

Reproduce independently checked reference, then final bounded computation from lab root:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 15 \
.improvement-ledger/runtime/calibration-v1-env/bin/python -B \
research/improvement-loop-20261024/m01-native-successor-provider-sol-v1/reference-check.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTENSOR_FLAGS='cxx=,floatX=float64,mode=FAST_COMPILE,compiledir=/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-native-successor-provider-sol-v1/pytensor-cache' \
nice -n 15 .improvement-ledger/runtime/calibration-v1-env/bin/python -B \
research/improvement-loop-20261024/m01-native-successor-provider-sol-v1/check.py
```

Final observed results: single-component density error 4.44e-16; all-4,000 mixture/direct-oracle error below 1e-15; six-mean gradient finite-difference maximum error 5.59e-10; entire-factor-list density delta error 2.77e-11. Nine mutation checks reject. Final computation took 30.29 seconds. Complete-density central differences are step-stable at the recorded tolerances, but are not analytic full-gradient validation. Full 662-dimensional gradient is unavailable in this runtime.

`provenance-check.py` separately checks actual calibration training CSV bytes against the incumbent Problem's retained-training SHA and exports `calibration-label-ledger.json`: 1,483 rows, 105 conditioning event labels, empty intersection with the two native official-count event keys. It uses no outcome values. No native token is fabricated for those labels, and sample identity remains unknown for all selected rows. The posterior bank remains separately byte-bound; this provenance export does not establish a new joint source-dependence law.
