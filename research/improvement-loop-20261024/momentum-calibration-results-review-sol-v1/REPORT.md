# Independent terminal calibration review — original attempt

2026-10-25. Reviewed `momentum-calibration-execution-sol-v1/attempt-root-v1` after Root reported original exec71863/PID1567524 exited0. I did not poll, stop, restart or control that handle/process. **The output preserves a real numerical-gate failure: five fits pass; the 2022 six-scale fit fails, and its 2024 alias is not qualified.** No threshold was relaxed, failed point replaced, optimizer rerun, target forecast generated or forecast score computed in this review.

External execution receipt SHA: `4c150bc181a474a64d75b121770bb9bf5da63e5cd1357ee950a8b3c4a480d79d`; run_id `m02-calibration-20261025-root-attempt-v1`. Runner manifest SHA `fcaccb33a483045e16d179ce5266370520db853f4c997cb3e0e5afcb4c991ef2`; design SHA `5c13a5d7552fdfe893bb0548a9205688c9a921a5d9faf2fd90147d5ff1cf5234`. Terminal status is `completed_with_calibration_failures_preserved`; summary SHA `6d53b91e4202d7246a3089f05cecb36d1af6444303d5cd8217f76323f1c230a2`. Exit0 denotes completed bookkeeping, not passing all gates.

`check.py` verifies launch/run/runtime/source bindings and hashes every original run file. All **six fits**, **eighteen original starts**, and **eight fold receipts** are present. Three starts retain the exact .5/1/2 prior-median initial points; all full coordinate/gradient/SD arrays have the expected five/six dimensions. Every started/returned evaluation sequence, wrapper count, iteration count, finite flag and final recomputation count reconciles. All eighteen starts report optimizer success, without silently becoming eighteen gradient-converged solutions. Fit/result/summary/alias pin chains verify. Sources remain the accepted earlier-only preparation; no target labels or new poll data were introduced. Alias2024→2022 preserves exact input identity, not forecast identity.

Independent checks recompute each saved best objective and gradient through the accepted wrappers, with an optimizer/RNG prohibition: discrepancies are exactly zero. A separate dense reconstruction from raw cycle coordinates, Gaussian determinants/inverses and frozen penalties agrees within **1.03e-12** objective and **4.75e-12** gradient. Natural SDs equal `exp(logSD)` exactly. Declared gate booleans, start agreement, inactive-bound distances, Hessian symmetry/eigenvalues/inverse/correlation and two-step stability reconstruct. Per-cycle covariance geometry is rebuilt independently and matches the stored eigenvalue/condition records. These are supplied-point computations, not another optimization.

| Input/arm | Best start (zero-based) | Recomputed gradient infinity norm | Fixed gate |
| --- | ---: | ---: | --- |
| 2018 five-scale | 0 | .0000155496302 | pass |
| 2018 six-scale | 2 | .00000408940039 | pass |
| 2020 five-scale | 0 | .0000867204429 | pass |
| 2020 six-scale | 1 | .00000296610784 | pass |
| 2022 five-scale | 0 | .0000188262984 | pass |
| 2022 six-scale | 0 | **.000188830305** | **fail: greater than .0001** |

The failed six-scale job passes finite, optimizer-success, bounds, agreeing-start, Hessian-finite/stable/positive and SPD geometry gates. Its other two start gradients are .000189833488 and .000797446145, so selecting another start would not satisfy the original gradient gate either. All three are within the declared coordinate/objective agreement tolerances, but agreement of nearby points does not establish stationarity. Some nonbest starts in otherwise passing fits also have gradients above the best-gradient threshold; the frozen contract gates the selected best point, while preserving every start. Do not summarize all eighteen starts as gradient-converged.

## Why the failed job stopped

All three failed-job messages are `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`. The installed, runtime-bound SciPy L-BFGS-B documentation states its ftol stopping condition as `(f_old-f_new)/max(abs(f_old),abs(f_new),1) <= ftol`; its separate gtol criterion concerns the projected gradient. Inspection uses the installed source, not an optimizer call.

From the last two retained iterations:

| Start | Final objective reduction | Relative reduction | Frozen ftol |
| --- | ---: | ---: | ---: |
| 0 | 6.5676886e-9 | 5.6691920e-12 | 1e-11 |
| 1 | 6.2391337e-10 | 5.3855852e-13 | 1e-11 |
| 2 | 5.4933480e-10 | 4.7418272e-13 | 1e-11 |

All satisfy ftol while failing the independent gradient gate. Minimum log-bound distance is about3.15145, so this is not a bound-active projected-gradient effect. The objective magnitude is about1158.49, making the absolute reduction threshold about1.16e-8. A local quadratic remaining-decrease estimate at the failed best point is about9.09e-10; it explains why an objective-change condition can fire while the raw log-coordinate gradient remains too large. This estimate is only scalar local algebra, not an executed Newton update or proof of a unique global optimum.

The saved/recomputed gradients and the independent dense oracle agree many orders more tightly than the failed margin. Thus the failure is not caused by reviewer arithmetic, stale solver jacobian or a source mismatch. It is an actual failure of the predeclared numerical contract, consistent with objective-reduction termination before sufficient gradient convergence.

## Interpretation and bounded refinement

The six-scale inverse observed penalized-Hessian Brownian/slope correlations are approximately −.540567 (2018), −.591278 (2020), −.597295 (2022). They are **local log-SD curvature approximations including the regularizer**, not empirical posterior correlations, integrated scale uncertainty, identifiable causal mechanisms or forecast usefulness. Kernel/SPD geometry and agreeing starts do not remove source/measurement confounding. The positive Hessians are numerical information about these penalized stationary neighborhoods, not evidence for promoting the slope mechanism.

As an additional control, the saved2022 five-scale natural SD vector exactly reproduces the original selected MAP calibration: independent .08438043897858434, cycle .06693935991701246, race .06560912002656763, house .07222665870699879, movement .08071632721709084. The original calibration JSON SHA is `474160120e00eb6ae6eadbf567434fad380dda64db7e1f096da7f240cbb3c85c`. This confirms the unchanged baseline target; it does not cure the six-scale failure.

Root's proposed v2 design (`dede87ca1843d8d98fd569effe3ddc81f84ccbb5111486ea3d1b19508b541f62`) tightens only ftol from1e-11 to1e-14 for both arms/all three unique inputs and all original starts. **That is an appropriate bounded numerical refinement to review**, because it addresses the observed stopping mechanism without changing scientific data/law/priors or weakening the gradient gate. It should remain a new externally pinned attempt with identical starts, no warmstarts, unchanged gtol/bounds/maxiter/gates and preserved original results. It does not guarantee success: actual v2 outputs must pass the same independent gate checks. This statement is not execution authorization and no v2 solve was run here.

Do not export the failed2022/2024 six-scale calibration as qualified, claim the failure is an empirical predictive loss, or rank models by their training objectives. The five passing fits satisfy their declared numerical gates only. Forecast/score evidence, parameter uncertainty and target-source/bank admission remain separate. All original failure evidence stays immutable before any refinement is selected or executed.

Exact inspected run/source SHA pins, complete start stopping ratios, fit/gate outcomes, local curvature diagnostics and independent numeric errors are in `results.json`. The only executed reviewer code was read-only hashing/parsing/linear algebra and saved-point objective evaluation with optimizer/RNG entrypoints blocked. Root retains ownership of original handle and any future authorized execution.
