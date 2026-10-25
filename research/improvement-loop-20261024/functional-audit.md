# Eight-layer verification, October 24–25, 2026

All eight roles exist in research. No canonical layer needs to be invented to complete the inventory. The initial research-presence criterion does not activate these providers on the public site.

The selected modular catalog is preserved in [evidence/modular-catalog-v1.json](evidence/modular-catalog-v1.json), SHA256 `025d248741ca21d3f723528c0d3ac8f22a0a7c8f49e500018f8e8e077c13dcf0`. All nine catalog implementation entries match their source hashes. The default composition's roles are:

| Role | Existing implementation | Meaningful work |
| --- | --- | --- |
| Population/geography | `protected_full_parent_exposure_v1`, with upstream national population source/path models | Carries protected uncertain population states into the graph. Upstream correlated source law and dated paths model exposure uncertainty; the downstream passthrough alone is not the population model. |
| Community | `demographic_profile_drift_v1` | Demographic cells and religion/attendance profiles, state differences and temporal drift. |
| Fundamentals | `lean_carryover_context_incumbency_v1` | Earlier electoral lean, prior presidential context and incumbency. |
| Environment | `separated_lkj2_cycle_student5_race_v1` | Correlated cycle effects and race residuals. |
| Participation | `profile_turnout_response_observed_total_gamma_v1` | Profile turnout and response with a shared physical-event count multiplier. |
| Candidate choice | `candidate_iid_profile_choice_v1` | Candidate utilities and contrasts; the catalog also contains `candidate_TX_ballot_role_profile_choice_v1`. |
| Counts/rules | `single_mark_category_allocation_poisson_v1` | Candidate/noncandidate allocations and official-count likelihoods. Separate ranked/runoff implementations exist for expanded scope. |
| Measurement | `joint_CES_F1a_final_pair_total_error_v1` | Joint survey patterns, source discrepancies, administrative report error and polling measurement. |

Scientific source root: `/home/hayden/projects/wethervane-redesign-20261006/integration`. Main graph: `research/senate-stack-integration-20261020/modular-count-graph-astra-v1/`. Population source/path: `national-population-joint-astra-v2/` and `national-population-path-astra-v1/` under the same research parent. The current source hashes are in [the capture manifest](evidence/capture-manifest-v1.json); a commit ID alone would not capture the owner's uncommitted research.

## Fresh checks

**110 population checks passed** in 2.03 seconds. Root ran the source, independent source, dated path and independent path suites with Python bytecode/cache disabled, single-threaded numerical libraries and isolated temporary output. These include finite mathematical comparisons, supplied inputs, source/temporal support, shared shocks, source periods, projection, population-mass preservation and artifact replay. No sampler was invoked. [Preserved log](evidence/population-functional.log).

**31 independent downstream checks passed** in 202.17 seconds. Sol ran `modular-count-graph-review-panel-sol-v1/test_independent_v1.py` with an isolated PyTensor cache. Checks cover supplied-state density/factors/gradients and physical outputs, source duplication rejection, protected-parent weights, typed ports, and source/configuration/target replacement boundaries. The verifier recorded unchanged scientific source hashes, which root checked again at capture. PyTensor warned that BLAS was not linked; this is not a fitting-performance benchmark. [Receipt](evidence/downstream-verification.json), [log](evidence/downstream-functional.log).

The older actual-graph witness was not rerun. Its source hashes match the currently verified compiler, but its existence does not establish a new successful fit.

Population hashes were captured after the run; unlike the downstream receipt, this capture is not a separate before/after unchanged-source proof. The original `/tmp` evidence is copied here so cleanup will not erase the results.

## What remains unestablished

These 141 checks support meaningful implementations and coherent tested joint calculations. They do not establish all of the following:

- A numerically qualified actual count-stack posterior and forecast.
- Incremental predictive benefit for each role, or the weakest role.
- Complementarity under matched jointly refitted A/B ablations.
- Full national geography/candidate/rule coverage or a qualified public export.

The Senate owner retains the numerical repair and actual-fit work. Prior pilots failed or timed out, and retained diagnostics include an independently documented process-cleanup/evidence-integrity incident. New runs must preserve distinct attempt directories and verify child termination. No frozen failure was changed into a pass here.

The next scientific evidence is a qualified incumbent fit and matched component-replacement comparisons. Existing aggregate Senate and House historical results can inform research, but cannot be relabeled as evidence for the successor count stack.

The owner's later [deduplication note](evidence/senate-owner-deduplication-v1.md) explicitly identifies the actual candidate-role hierarchy and its exact numerical wrappers. The catalog's first baseline composition is a preserved interface reference, not a claim that its IID candidate option is the currently running pilot. That pilot has passed a repaired worker-startup boundary; posterior convergence and predictive qualification remain unestablished at this checkpoint.
