# Independent E01 preparation review — Sol

Reviewed 2026-10-24. This is an independent read-only review of `economic-experiment-v1/prepare.py`, `test_prepare.py`, and the frozen `prepared-v1` output. No producer files were changed, preparation regenerated, sampler invoked, or scientific/public provider activated.

**Verdict:** the frozen preparation supports the stated narrow aggregate experiment. I found no material error in its H0/F0 recipes, E01 nesting, earlier-only labels/scalers, or admitted CPI join. It is not yet an executable, qualified forecasting experiment. Approve preparation as evidence; review the fitting/prediction/scoring runner before launch. An aggregate improvement would not qualify the count graph.

The reviewed plan SHA-256 is `cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9`.

## Exact recipes and joint target

`prepare.py:206–230` uses the existing factory, whose bytes are hard-pinned before import (`factory`, lines 196–203). H0 is `legacy_recipe('senate_history_refreshed_v1')`; F0 is the default `standard_providers()` recipe, equivalent to `senate_relative_context_v1`. The owner factory's `standard_providers` and `legacy_recipe` at `src/forecasting/aggregate_component_factory_v1.py:245–353` confirm the fixed/learned state and national slope choices, signed incumbency, context terms, shared cycle normal, Student-t5 race law, and identity polling factor. E01 retains every F0 provider and adds exactly one mean term and Normal(0, 0.05) coefficient.

The complete normalized joint density includes coefficient priors, cycle latent priors, both scale priors and historical Student-t likelihood (`aggregate_component_factory_v1.py:617–634`). The supplied-point tests compare H0/F0 against the independent incumbent `fundamentals_v1` density in all four folds, rather than merely comparing factory recipes. At economic coefficient zero, E01 equals F0 plus the extra coefficient's normalized prior density at zero; it is correctly not asserted to be numerically identical without this constant. The analytic economic gradient also agrees with finite differences. This establishes computational nesting at supplied points, not posterior convergence or forecasting benefit.

Plan conditioning explicitly requires joint inference of all coefficients, historical cycle effects and cycle/race scales. No fitted-column splicing, post-poll shift or extra national shock is introduced. F0 and E01 must still be independently jointly refitted; the zero restriction does not justify recycling their posterior columns.

## Orientation, scaling and prior

The covariate is `z = (president_party * headline_CPI_yoy - earlier_mean) / earlier_sd`, with D president +1 and R president −1 (`prepare.py:157–193`). Inflation is in percentage points. Scaling uses one equally weighted observation per strictly earlier election cycle, population SD (`ddof=0`), not one observation per race. Training cycle counts are 10/11/12/13 for the four targets. Target observations never enter their own scaler.

A negative coefficient would penalize the incumbent president's party as inflation increases, holding other terms fixed. The symmetric prior does not assume that effect or a causal worsening. A coefficient SD of 0.05 is in Democratic-share logit units per SD of the **oriented** national covariate; it is not a 5 percentage-point share effect. Centering after orientation adds an intercept component. The alternative `party * (CPI - unsigned_mean) / unsigned_sd` is generally a different coordinate involving the party term; independent same-SD priors are not equivalent joint priors. The diagnostic acknowledges this rather than silently treating both as the same model.

Confounding remains substantial: the saved correlations with presidential party are 0.915/0.920/0.911/0.813; with national carryover, 0.892/0.883/0.884/0.678. This is not an omitted-party baseline: F0 already contains party, party×midterm, national carryover and a shared cycle residual. E01 tests incremental usefulness under shrinkage, with very few independent national realizations. Inspect coefficient/scale posterior dependence and predictive gains; do not infer an identifiable causal CPI effect from numerous race rows.

The 2022 raw CPI value is outside earlier raw support and its oriented standardized coordinate is 3.5453. Its CPI-only prior central 95% logit shift is ±0.34743, or approximately 41.40–58.60% at a 50% baseline. The other targets' corresponding intervals are 47.74–52.26%, 48.81–51.19%, and 48.45–51.55%. This large 2022 extrapolation is visible and merits the predeclared tighter 0.025 sensitivity. It is not itself proof the prior is unacceptable. The 4,096 prior draws are diagnostic only, include shared cycle and race variation, and do not constitute empirical calibration.

## Labels, source vintages and support

`earlier_outcomes` (lines 146–154) reads cycle metadata and only the strictly earlier response rows. Reading cycle metadata to select rows does not condition a model on future outcomes. The outcomes source is pinned in full; per-fold response hashes and exact race IDs are retained. All three arms share 299/329/357/387 earlier training labels and 30/28/30/30 target races. These are the admitted ordinary D/R Senate panel, not all Senate contests. Final roster and archive reconstruction limitations remain explicit.

`load_macro` (lines 87–132) verifies the pinned midnight-v2 receipt, its complete inventory, inherited source receipt/raw inventory and selection bytes. It requires headline CPI-U NSA US-city-average 12-month percent changes, and both release/vintage timestamps no later than exact election-date-midnight minus 14 days UTC. Feature forecast cutoffs must equal these cutoffs. The critical 2004 sentinel uses August 2.7%, not the later-on-cutoff-day September release; 2014 uses 1.7%. Target values are 2.3/1.4/8.2/2.4%. NSA/source revision and retrospective archive qualifications are inherited; this does not authenticate original website posting times or turn current-history CPI into vintage evidence. Core, gas, income and state interactions are not silently included.

I independently recompiled all 12 arm/fold laws and checked their recipes, law hashes, design and response hashes, target race IDs and scalers against the frozen plan. All 12 recorded source pins and copied source snapshots match current bytes. The frozen feature exports contain target covariates but no actual-share labels. This is reproducibility within the pinned checkout/runtime; the plan's absolute paths are not a portable source resolver.

## Gates before execution, and later evaluation

These are execution-contract requirements, not defects requiring a new CPI preparation:

1. Bind a functioning, reviewed PyMC/ArviZ runtime and runner to the frozen plan/source hashes. Validate diagnostics for all modeled coefficients, scales and latent cycle effects, preserve failures, and forbid incomplete draws from becoming an empirical pass. The preparation honestly says fitter runtime is not certified and `fit_authorized` is false.
2. Specify and test predictive generation: one new-cycle draw shared across target races, appropriate new race residuals, and complete joint fitted draws/scale dependence. Historical fitted effects cannot substitute for a new-cycle draw. Seal fold predictions before accessing that fold's score labels; later folds may legitimately train on earlier evaluation cycles. This is iterative retrospective replay, not a globally untouched holdout.
3. Freeze executable score definitions and evaluation masks before scoring: CRPS units/aggregation, WIS interval levels, coverage/width, clipping conventions for winner log score, competitive mask threshold and common forecast-time input, and the precise earlier-only-frequency baseline. `score_controls` currently lists intentions, not these algorithms. State what A-tier gain and material-regression guards would matter; no requirement that every metric or every cycle improve.
4. Keep initial execution F-only as declared. Later matched FP needs its own pinned poll factor/data/cutoff/alias/support plan, and the same polling information and conditioning algorithm across arms. Do not treat absence of FP bindings as an already approved combined forecast.

A2018/2022 and B2020/2024 follow the project verification skill for a 2026 midterm target. Report each cycle and A combined results, B transfer, uncertainty and near-even behavior; do not let pooled race counts masquerade as national-cycle replication. H0 provides a history reference; F0 versus E01 is the incremental macro comparison. Better aggregate CRPS with destructive competitive/winner or calibration changes requires explanation under prespecified guards, not an automatic pass. Conversely, no universal every-cycle-win rule is warranted. The tighter prior is a declared sensitivity, not permission to select a winning prior using target scores and report it as outcome-blind.

## Actual checks

From `economic-experiment-v1`, ran:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/hayden/projects/wethervane-redesign-20261006/integration/.venv/bin/python \
  -m unittest -v test_prepare
```

**7 tests passed (0.866 seconds)**. Tests use supplied densities/gradients, source checks and temporary copies only. A separate read-only Python probe recompiled and checked all 12 serialized laws and source/snapshot pins, as described above. First attempted the isolated calibration runtime; it lacks `pydantic`, so zero tests ran there. Re-running in the existing scientific `.venv` passed. No dependency was installed and no fitting was launched.

Review basis: scientific `AGENTS.md`, `docs/ARCHITECTURE.md`, the operational stack specification, and the `verify-election-model`/`refine-model-layer` skills. This report records preparation correctness and remaining execution gates only; no numerical fit, empirical qualification, calibration-cut validation, count promotion, or public activation is claimed.
