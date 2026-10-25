# Independent economic execution review — Sol

Reviewed 2026-10-25, read-only. Frozen execution plan reviewed: SHA-256 `73c6bbf807f784d7afbee6914a56d95f358bf4f7396e8c60c84882358f2d8273`. Preparation review remains in `economic-experiment-review-sol-v1.md`. I inspected `economic-execution-v1/{runner.py,scoring.py,test_execution.py,README.md}` and ran only bounded tests that replace sampling. No actual sampler, `--run`, producer edit, or provider activation occurred.

**Verdict:** the fresh-attempt prediction/scoring implementation satisfies the preparation's principal outstanding requirements. No fresh-launch scientific blocker found within this review's scope. Root's independent runtime acceptance and explicit execution authorization remain separate gates. One integrity repair is required before relying on continuation's advertised no-retry guarantee; details below. Numerical and empirical qualification remain future evidence.

## Substantive finding: continuation does not pin numerical-failure status

In the reviewed version, `runner.py:571–606` seals the launch and complete-chain receipts, but not the presence/content of each job's `terminal.json`. `run` at lines 672–675 reads a prior job terminal directly and rejects a numerical failure only **if that file exists**. The resumed chain itself is well pinned; the decision that it may be reused is not. Deleting a prior `numerical_failed_preserved` terminal or changing its `numerical_passed` flag after the continuation manifest is sealed bypasses that guard without changing any externally pinned receipt. This need not involve changing a raw bank.

Bind each job terminal's presence and hash in the externally reviewed continuation manifest, authenticate before its status is used, and reject mismatch or disappearance of a previously present terminal. Add tests for changed/deleted failed terminals and a valid incomplete-chain continuation. The current bounded resume test checks a corrupted raw bank but not this failure-status case. Sent directly to producer and Root. This is a continuation integrity issue, not evidence that a fresh launch fits the wrong model. Continuation also re-exposes previously scored labels, as the README appropriately discloses; it cannot restore unopened-label status.

## Joint prediction and diagnostics

`predictive_arrays` (lines 458–484) keeps original coefficient/cycle-scale/race-scale rows aligned. For each retained row, five future replicates use one new standard-normal cycle shock shared across target races and independent Student-t5 race residuals, scaled by that row's jointly inferred scales. It does not reuse historical fitted cycle effects for the new cycle or splice posterior columns. The NPZ retains chain order, posterior-draw/replicate indices, shocks and exact race IDs. Replicates are contiguous; they are future simulation draws, not additional fitted posterior rows.

`fit_diagnostics` checks all coefficient, scale and cycle-latent coordinates, rank Rhat, bulk/tail ESS, divergences and depth hits. Nonfinite coordinates cannot silently vanish through a skip-NaN reduction. Prediction diagnostics include shares, winner indicators and the target-subset win total; this last quantity is explicitly subset numerical integration, not full-chamber control. Constant empirical winner tails remain flagged unresolved rather than treating zero numerical error as reliable rare-tail accuracy. Parameter ESS and predictive ESS have distinct interpretations. These are computational gates, not empirical calibration.

## Scores, masks and numerical error

`scoring.py` supplies concrete, bounded definitions: empirical share CRPS; WIS from median and central 80/95 intervals with denominator 2.5; winner Brier and log score with floor 1e-6; share coverage/width; margin summaries. The ordinary D/R aggregate target supports share > .5 as its winner rule; this must not be transferred to first-ballot or multi-group targets. Outcomes must be finite, interior and non-tied. Tests compare CRPS with the independent pairwise formula and its numerical influence with weight perturbation.

All arms use common forecast-time masks from the sealed F0 probabilities: competitive [.1,.9], near-even [.4,.6]. Empty masks stay unavailable. Race means are equal-weighted within cycle; A=(2018,2022) and B=(2020,2024) use equal cycle means. References include always-half winner scores and equal-earlier-cycle D-win frequency. They are winner references, not fabricated full share-distribution baselines.

Numerical score uncertainty uses the unchanged owner chain/batch routine with contiguous blocks corresponding to 50/100 original posterior rows. The empirical CRPS influence is centered by variance estimation; its raw mean is not asserted to equal CRPS. Independent-arm score-difference numerical variances add; independently seeded cycle variances combine before tier averaging. These errors do not represent uncertainty from having only two national elections per tier. Winner-score regression flags are unavailable for unresolved constant tails. Every cycle/stratum loss remains visible; the .0005/2% CRPS aid and two-MCSE flag do not automatically adopt a model. This follows iterative A/B verification, without a universal every-cycle win mandate or score-selected tighter prior.

## Label sequence, runtime and preservation

The loop compiles only the current fold's strictly earlier numeric labels, creates all four arms' predictive artifacts and seals, freezes the common F0 masks, then calls the target outcome parser. `load_labels_after_seals` checks plan/target/arm, prediction bytes, mask policy/support and mask reconstruction before parsing numeric outcomes. Target selection uses cycle metadata first, preserving the earlier-only training/scaler qualification. Later folds may train on earlier evaluation cycles. This is retrospective development, not an untouched global holdout.

Execution requires the externally supplied exact plan SHA and runtime acceptance SHA, inner acceptance pins, matching package lock/inventory and interpreter/prefix, and one-thread numerical settings. Root owns the runtime review, including its disclosed supplied-point normalizer discrepancy; this review does not replace that acceptance. Source and snapshot inventories are pinned. The 16 ordered jobs include 12 primary fits and four predeclared tighter-prior fits, all jointly fitted with matched support/budget. F-only scope remains explicit; no FP poll plan, calibration cut, count qualification or promotion is inferred.

Fresh directories and exclusive file creation preserve prior output. Each completed chain includes warmup and exact chain/coefficient/cycle identity and raw-bank pin. Failure records preserve completed-chain references and disallow automatic retry. Numerical-failed artifacts can still be scored descriptively but remain marked failed; overall terminal `all_numerical_passed` cannot become true if any arm failed. The top-level `completed_F_only_comparison` denotes workflow completion, not passing qualification. Downstream consumers must inspect the explicit numerical flag and future independent review, not that status string alone.

## Actual bounded validation

Ran from the execution package:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/economic-v1-env/bin/python \
  -m unittest -v test_execution
```

All **12 tests passed in 4.334 seconds**. The 16-job roundtrip replaces fits with synthetic banks, mocks diagnostics/MCSE where stated, uses removed temporary outputs, and exercises actual compilation, sealing and scoring. The chain interruption test mocks every sampler invocation. Expected warnings arise from a deliberately tiny four-chain/three-draw bank and a constant-coordinate diagnostic rejection fixture. These tests do not establish real sampler convergence, runtime performance, empirical skill, tail precision or a successful real forecast.

## Independent repair delta — 2026-10-25

The producer superseded the execution preparation with `prepared-v2/plan.json`, SHA-256 `e3faefeb89a5bca837bc7f8b01e666112d563b95fe16e909060edf5aa1cbfc50`. I independently confirmed both that digest and the unchanged original v1 plan digest above. The original finding remains recorded here; it is repaired in the reviewed v2 code.

The continuation manifest now includes every ordered job terminal path with either its byte pin or explicit absence. Before creating an output directory, `run` checks the exact 16-job inventory, terminal presence, bytes, plan identity and successful terminal status. `checked_continuation_terminal` repeats the presence/byte check immediately before reuse, and the status check is repeated there. Thus deleting a failed terminal, rewriting it as successful, keeping the unchanged negative terminal, or introducing a terminal after sealed absence cannot silently authorize continuation. Complete-chain receipt/raw-bank validation remains intact.

I independently ran these three bounded tests in the certified economic runtime, with all four numerical thread limits set to one and bytecode disabled: `test_all_fold_execution_roundtrip_with_explicit_synthetic_banks`, `test_continuation_absence_cannot_be_replaced_after_sealing`, and `test_failure_preserves_chain_and_resume_is_new_and_verified`. **All three passed in 4.273 seconds.** They exercise the changed/deleted/negative terminal rejection before new-attempt creation, sealed absence rejection, and valid complete-chain reuse after an interrupted mocked sampler. Every sampler invocation remained mocked; synthetic outputs were temporary and removed.

**Final review disposition:** accept v2 for a fresh launch and for the declared explicitly reviewed continuation mechanism, subject to Root's separate execution authorization and exact runtime/plan bindings. No remaining blocker found within this bounded review. Acceptance is of implementation and scope, not a claim of successful sampling, empirical skill, cut compatibility, count qualification, promotion or public activation. No fit was launched.
