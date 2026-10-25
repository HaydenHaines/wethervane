# Source-bound F0 and measurement adapter

This packet prepares constructors and source identities for the selected retrospective H14 aggregate experiment. It does not execute an updater, generate forecasts, load numeric target outcomes, summarize archived forecast values, or calculate scores. The selected scientific design is `checkpoints/momentum-forecast-design-v1.json`, SHA `45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678`.

`adapter.load_adapter(cycle)` returns a frozen `F0MeasurementAdapter` for 2018, 2020, 2022 or 2024. Its `prior_kwargs` are the exact constructor arguments for an `EmpiricalBasisFundamentals`: pinned target offset/design, six ordered coefficients, aligned positive cycle/race scales, complete bank race roster, original chain/draw coordinates, Student-t5 race law and explicit source identity. The original four chains each retain all 4,000 posterior rows in `chain*4000+draw` order. There is no thinning, permutation, refit, resampling, or replacement by predictive replicates. The starting empirical row mass is exactly uniform `1/16000`. Historical `cycle_z` and `cycle_effect` are retained separately for source reconciliation and never become future shocks.

`measurements['FS5']`, `measurements['FS6']` and `measurements['LP5']` contain immutable coordinates/shares, named accepted MAP scales, and Gaussian likelihood constructor kwargs. FS5 and FS6 have identical full-series observations; LP5 uses the already accepted latest-per-race/pollster operator. Their respective row counts are 202/202/170, 190/190/134, 220/220/171 and 351/351/226. All use exact admitted conditional D/R log odds and field-end time coordinates; covariance comes from the accepted five-term owner law plus the independently reviewed persistent within-race/cycle slope term for FS6. No nominal-sample variance, jitter, new poll, availability lag, or event expansion is added. PO5 consumes precisely the FS5 likelihood, with the independently selected final-logit `Normal(0, 2.5²)` prior constructed by a later runner; this adapter does not calculate that posterior.

## Constructor identity

`data.build_prior()` and `data.build_likelihood('FS5')` construct the original owner classes. Arrays inside the returned objects, including their native cached products, are backed by immutable bytes rather than reversible NumPy write flags.

The private observed sampler has a copied `EmpiricalBasisFundamentals` class. Use `data.build_for_sampler(module, 'FS5', module_pin, dependency_pins)` for it. The caller supplies the already authenticated module and dependency modules, loaded under explicit identities. The adapter checks the exact source files and hashes against its frozen inventory, the concrete constructor names/module bindings, and the sampler function's constructor globals. It returns the prior, likelihood and constructor binding metadata, without calling the sampler. Passing an owner prior to the copied class's `isinstance` gate would fail; the test demonstrates this distinction.

`test_private_sampler_constructors_use_exact_snapshot_classes` is an executable loading example: it imports the private response and polling snapshots under unique names, temporarily binds their explicit dependency keys only during module execution, restores those keys afterward, and constructs the copied prior and matching likelihood. It blocks invocation of the sampler with a profiler. Raw immutable kwargs also remain available for another explicitly pinned constructor. This is constructor compatibility, not independent acceptance or execution authorization of the sampler.

## Archived predictive reference

`archive_reference(cycle, data)` reports separately named `ArchivedF0PredictiveReferenceIdentityV1` metadata. The four original `predictions.npz` files remain read-only. Their full byte hashes, sealed generation receipts, exact F0 law/recipe, target feature-array hashes and ordered race IDs are bound. NPY headers establish four chains × 20,000 predictive realizations, with five contiguous replicates per original chain/draw row. Only the three race/draw/replicate identity vectors are decoded. No share, logit, cycle-shock, or race-shock numeric array is decoded; the tests enforce that read boundary.

This is identity by sealed bytes, metadata and pinned generator source. It does not independently recompute predictive values or demonstrate their numerical accuracy. The 80,000 predictive realizations are not 80,000 posterior parameter rows. Old economic-experiment masks, score floors and quantile conventions are not imported into the new experiment; later prelabel processing must follow the selected forecast design.

## Reproduction

From `/home/hayden/projects/ballotmancy-improvement-lab`, use the exact `runtime.json` interpreter/prefix and relevant package versions:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .improvement-ledger/runtime/economic-v1-env/bin/python -m unittest discover \
  -s research/improvement-loop-20261024/momentum-forecast-adapter-sol-v1 -p test_adapter.py -v

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .improvement-ledger/runtime/economic-v1-env/bin/python \
  research/improvement-loop-20261024/momentum-forecast-adapter-sol-v1/adapter.py \
  --output research/improvement-loop-20261024/momentum-forecast-adapter-sol-v1/another-exclusive-report
```

All direct sources and their accepted inventories are authenticated before scientific imports. Subsequent terminal/fold/chain reads use the immutable expected pin index, rather than accepting self-rebound mutable receipt hashes. The report path must be a new directory inside this packet. `development-run-01/` preserves the initial successful inspection before final hardening; `verified-v1/` is the final report. Previous logs and reports are preserved, never overwritten.

## Limits

The preparation uses the already admitted retrospective archive availability proxy, including native created-date plus one day and LV/RV/adult priority. It is not strict source-vintage/as-of qualification. The bank and scale calibration share earlier realized outcomes; this remains a conditional empirical-Bayes approximation. Scale uncertainty is not integrated. A finite original historical bank is not repaired by extra target simulations. Original F0 bank diagnostics are inherited from the authenticated accepted economic run, not recomputed here. No new source acquisition, count-model qualification, population cut, chamber-control statement, public activation, or empirical superiority claim is made.
