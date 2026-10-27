# 2018 NUTS divergence remedy research and predeclared candidate

Read-only research and candidate design, 2026-10-26. No fit, graph construction, replay, sampler diagnostic or owner-file modification was performed in this task.

## Existing failure and source-specific model facts

The owner-supplied 2018 audit reports 76 retained divergences among 32,000 transitions (0.2375%), with at least one divergence in each of eight protected-parent conditionals. Its predeclared gate requires zero divergences and zero depth saturation, so the completed run fails the gate even though the process exited successfully. This proposal retains the gate exactly.

The inspected 2018 plan selects `candidate_iid_profile_choice_v1`, `standardized=True`, `person_coordinates=hybrid_v1`; it does not select the TX ballot-role hierarchy. The parent-0 model contract records empty observed-total coordinate centers/scales. Therefore a TX role hierarchy change or observed-total coordinate rewrite is not an evidence-based 2018 remedy candidate here.

The actual selected source already uses several non-centered hierarchical blocks: pooled parameters are `raw ~ Normal(0,1)` multiplied by a `HalfNormal` scale; the correlated cycle vector is `cycle_raw ~ Normal(0,I)` multiplied by a Cholesky factor; the final-race effect is `StudentT_raw * final_race_sd`; and community drift is a raw Normal innovation path. A broad “non-center the model” prescription would repeat existing parameterization, not address a demonstrated defect. The baseline candidate-specific effects for this composition are independent Normals, not the TX hierarchical role prior. The observed-total coordinate list is empty.

The existing descriptive location audit reports flagged leading-coordinate families such as `community_intercept` (40), `final_race_raw` (12), `community_drift_raw` (6), and `source_community_state` (5). The nearby-control sample reports similar leaders (`community_intercept` 35, `final_race_raw` 12, `community_drift_raw` 3). Median maximum standardized coordinate magnitudes are 4.25 for flags and 4.20 for controls. These are descriptive nearest-coordinate summaries, not a finding that any named coordinate caused the divergences. The separate divergence-state exploration queue exited 1 and supplied no successful causal geometry result. Until the pending owner geometry analysis is available, this evidence does not justify a new prior, transformed coordinate, dense-mass block, or model rewrite.

The baseline already adapts step size and a **diagonal** mass matrix during 1,500 warmup iterations; `dense_mass=False`. Therefore “enable adaptation” is not a new treatment. The gate audit observed no depth-limit hits, so increasing maximum tree depth is unsupported as a divergence remedy. The highest-information, smallest-change candidate is a single target-accept change that directly tests whether discretization resolution is a material contributor.

An independent source challenge by Sol agrees on these target-specific limits: [`national2018-remediation-challenge-sol-v1/README.md`](../national2018-remediation-challenge-sol-v1/README.md), SHA-256 `522113b4762bd1ef4090f77210ee611a3d11cb84674e59d2c32971653d1ae646`. It additionally confirms the 2018 cycle implementation already separates scales/correlation and that previously validated TX/NV transport methods are not automatically portable to the nationwide graph. This review does not promote those other-state fixes as candidates.

## Candidate, declared before any follow-up run

**Candidate C1: hold the exact 2018 model and all other settings fixed; change only NUTS `target_accept_prob` from 0.97 to 0.995.** NumPyro 0.15.3 documents that increasing this target yields a smaller adapted step size and usually slower but more robust sampling. This is an established sampler adjustment for divergences when integration error is the issue; it is not evidence that this particular model's divergences have that cause.

The matched run, if separately scheduled after the owner geometry analysis and resource/source review, must preserve:

- All eight original protected-parent conditionals, exact authenticated inputs and original outer masses; no parent exclusion, evidence reweighting or changed cut.
- Four chains per parent, the original parent-indexed seed scheme and original initialization; 1,500 warmup and 1,000 retained draws; max tree depth 12; same NUTS implementation, backend, device, runtime, model/provider/data hashes, coordinate schema and storage protocol.
- The single declared difference: target acceptance 0.995 instead of 0.97. Do not simultaneously increase warmup, change mass matrix, alter initial values, change priors, or reparameterize.
- The original full gate without exception: each conditional must pass existing R-hat, bulk/tail ESS and MCSE checks, with zero retained divergences and zero depth saturation. Preserve every transition and every parent's outputs, including failures.

Primary falsifiable prediction: if finite-step integration error is a consequential cause and this single target change is sufficient, candidate C1 yields zero divergences across all eight parents and retains the existing diagnostics gate. Any retained divergence or any other gate failure falsifies the candidate as a qualification remedy. Compare divergence counts by parent and chain, acceptance, adapted step sizes, tree-depth/saturation, E-BFMI, ESS/sec, and elapsed time to explain the cost and mechanism; these secondary summaries do not relax the gate. A zero-divergence pass with unacceptable ESS/MCSE still fails.

If C1 fails, do not iterate target acceptance until the run passes. Use the pending geometry analysis to design one new, target-preserving reparameterization of the implicated 2018 block, with explicit prior/Jacobian and physical-quantity equivalence checks, then compare it against the unchanged baseline under a new predeclared version. If geometry instead indicates a log-density/gradient defect or bad initialization, address that defect as a new model/code target and do not call higher target acceptance a fix. No such geometry-driven second candidate is specified here because the identifying evidence is not yet available.

## Established remedies and limits

1. **Increase target acceptance / reduce step size.** NumPyro 0.15.3 says `target_accept_prob` controls dual-averaging step-size adaptation; increasing it produces smaller step sizes with slower, more robust sampling. Stan's reference manual likewise states this can eliminate divergences in some cases, but requires reparameterization in others. This supports C1 as a conventional, falsifiable sampler-only test, not as a pre-verified 2018 solution.
2. **Use geometry-guided reparameterization for varying curvature.** Stan's hierarchical examples and Betancourt's HMC treatment explain centered/non-centered transformations when hierarchical posteriors have funnel geometry. Because the 2018 graph already uses non-centered coordinates for its pooled latent effects and the current owner location summary is non-causal, no extra non-centering is justified without geometry results.
3. **Mass-matrix changes are conditional on cross-coordinate structure.** NumPyro supports dense or named block mass matrices, but current production settings already adapt a diagonal mass matrix. No source-reviewed evidence here shows which 2018 blocks have stable within-block correlations or that a dense matrix is feasible/useful at 3,101 unconstrained dimensions. Do not switch to dense/full mass by default.
4. **More tree depth treats saturation, not observed divergences.** The baseline audit reports zero depth saturation; tree-depth increase is not the candidate.
5. **Stronger priors or deleted/reweighted divergent draws change or misrepresent the analysis.** Neither is a sampler remedy. Any scientific prior redesign requires a separate scientific rationale and matched evaluation. Divergent transitions remain in the diagnostic record; do not drop them to manufacture a pass.

The model has 3,101 unconstrained coordinates in the actual parent-0 inventory. This dimensionality makes generic alternatives such as random-walk/ensemble methods poor plug-in replacements and leaves no basis here to change sampler family. The present candidate says nothing about posterior bias magnitude, predictive accuracy, model adequacy, or the cause of the observed divergences.

## Primary technical references

All linked technical references are primary documentation or the author paper:

- [NumPyro 0.15.3 MCMC/NUTS reference](https://num.pyro.ai/en/0.15.3/mcmc.html), especially NUTS arguments for target acceptance (lines 552–602 in the web-rendered reference), adaptive step size/mass matrix and dense/block options.
- [Stan Reference Manual: divergent transitions](https://mc-stan.org/docs/reference-manual/divergent-transitions.html), discussing smaller step sizes, varying curvature, and when reparameterization is required.
- [Stan User's Guide: hierarchical reparameterization](https://mc-stan.org/docs/2_27/stan-users-guide/reparameterization-section.html), with centered/non-centered hierarchical examples.
- [Betancourt, A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434), geometric context for HMC failure modes.

## Exact local evidence inspected

Original owner project is read only at `/home/hayden/projects/wethervane-redesign-20261006/integration`.

| Artifact | SHA-256 | Use |
|---|---|---|
| `research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1/gpu-production-v1/plan.json` | `fd44e4362c6c0a35e98aae23750b2266d4ea845aecea74de3970bb358a053209` | Target, all 8 parents/masses, 4 chains, 1,000 draws, 1,500 tune, 0.97 acceptance, max depth 12, NumPyro NUTS. |
| `.../actual-model-inventory-v2/parent0-contract.json` | `f4debf48cc2ed96281c2780a5dabd61d0abb7982625cba740f9cc3f08bf208cd` | Selected composition/coordinate contract, 2018 initial point and parent identity. |
| `.../gpu-production-v1/retained-divergence-audit-v1/result.json` | `21a7f534922c91618d8a92d5b2075dcc9d48b381b07a199f2a989d56d6d72089` | 76/32,000 divergence summary and zero depth-limit hits. |
| `.../gpu-production-v1/retained-divergence-state-exploration-v1/matched-controls-result-v1.json` | `720ff44675c0db40d0c1e4f06a222538b0d7eae464bfaaf7718c2ceb58099a29` | Descriptive flagged-vs-nearby-control coordinate leaders only. |
| `.../gpu-production-v1/retained-divergence-state-exploration-v1/failed-queue-v1.json` | `d5487e7b4676341e654d6b683b459e00121dd0d0cf0f6e41be8c57e428c26f6c` | Owner geometry/state exploration exited 1; no causal result. |
| `.../gpu-production-v1/attempt1/source-snapshots/034-batched_runtime.py` | `9d94b4a15bb309ccdd1228fd588bd5d8ea618555f8f2ed6c93e0e54f56ee5750` | Actual selected NumPyro NUTS config: adaptive step size, diagonal adaptive mass, settings forwarded. |
| `.../gpu-production-v1/attempt1/source-snapshots/139-components.py` | `59ff50bff5b99e79211d0592a68983f615d60a583a62ca8b2adcddfdd733786f` | Actual selected component implementations and 2018 candidate provider behavior. |
| `.../gpu-production-v1/runtime-record.json` | `d9b7f305242fc87e083d4c5bdd9e3198e8f814a46ef26ebc2ba8990e5f35208d` | NumPyro 0.15.3 runtime and device record. |

No actual run is admitted or initiated by this research packet. It contains a proposal, not a fit receipt or evidence that C1 works.
