# Independent refined calibration terminal review

2026-10-25. Root reported original exec65943/PID1569669 completed exit0. **All six fits satisfy the unchanged declared fit gates; one of eighteen starts remains an optimizer-reported failure and is preserved/excluded.** Accept the v2 results as numerically qualified conditional calibration under this exact scope, not forecast evidence or promotion. I did not poll/control either process, restart a fit, run an optimizer/sampler, generate a forecast or compute forecast scores.

Execution receipt SHA: `1854b4f9cd3a295a6e66f76528fe6fe69498c197733a2c10b01eac9d270aeca1`; run_id `m02-calibration-20261025-root-attempt-v2`. Runner manifest SHA `583ca465e8fea568843486542ad8d85832585661919aae974bcfb5c103e67160`; design SHA `dede87ca1843d8d98fd569effe3ddc81f84ccbb5111486ea3d1b19508b541f62`. Terminal status is `completed_calibration`; summary SHA `2bdbea1cfa7dd3749c56980a9a96496f6e3d89c375ea0e958c6f93a2ba29d131`. Exact runtime/source/preparation bindings remain the separately accepted versions.

The established read-only audit was reused only where unchanged law/code permits, with explicit v2 receipt/path/ftol and expected terminal bindings. Every run artifact is hashed. All **six fits**, **eighteen prescribed original starts** and **eight fold receipts** reconcile; no extra start/warmstart occurred. Launch run_id/PID matches Root's record. Evaluation-start/return sequences, full finite coordinate/gradient fields, wrapper/iteration counts and fresh recomputation counts agree with complete saved logs. Natural SDs equal exponentiated log-SDs. Fit/summary/alias byte chains verify. All2024 aliases point to the correct2022 arm terminal and exact accepted canonical input/coordinate/source identity; this does not alias target forecasts or banks.

At all six selected best points, accepted-wrapper objectives/gradients reproduce saved values **exactly**. Independent complete-cycle covariance construction and dense Gaussian algebra agree within **1.60e-12** objective and **6.58e-12** gradient. Direct declared-gate reconstruction matches every stored boolean. Hessian raw/symmetric arithmetic, eigenvalues, inverse/correlation and two-step stability verify; independently rebuilt covariance minimum eigenvalues/conditioning, active-kernel rank and Brownian/slope Frobenius cosine also match. This is saved-point algebra only, not another fit.

| Input/arm | Best successful start (zero-based) | Fresh gradient infinity norm | Unchanged≤.0001 gate |
| --- | ---: | ---: | --- |
| 2018 five-scale | 2 | .00000191912815 | pass |
| 2018 six-scale | 0 | .000000727885014 | pass |
| 2020 five-scale | 0 | .00000115646135 | pass |
| 2020 six-scale | 0 | .000000294824769 | pass |
| 2022 five-scale | 2 | .000000513032793 | pass |
| 2022 six-scale | 1 | .00000174980783 | pass |

The formerly failed2022 six-scale calibration—and hence its2024 calibration alias—now satisfies the same gradient gate after the ftol-only numerical refinement. Some solver messages still use relative-objective termination; passing the separate best-gradient threshold does not mean every optimizer stopped on gtol1e-6. The worst selected gradient is1.919128e-6, well below the frozen1e-4 qualification threshold. No threshold was changed to accommodate these outputs.

## Preserved failed start and audit adaptation

**2020 six-scale start1** has `optimizer_success:false`, `finite:true`, message `ABNORMAL: ` and fresh gradient2.96610784e-6. It remains in its original terminal/evaluation/iteration records and is **not** counted as successful merely because that gradient is small. The best point is chosen only among finite optimizer-reported successful starts; the other two successful starts agree under the original tolerances, as the frozen design permits. Therefore the correct counts are **17/18 solver-success, 6/6 fit-gate pass**. No numerical-record repair or retry is required to describe that outcome faithfully. The saved message alone does not establish a more specific line-search cause; no unrecorded solver explanation is invented.

The first reused reviewer assertion required all eighteen starts to report success, a fact of v1 rather than a declared gate. It failed on this retained v2 start. I inspected every start, then adapted only the new reviewer audit to the already-frozen **best-successful/two-agreeing-successes** contract, explicitly recording the failed start and17 count. This did not change producer code, scientific thresholds, start classification or the original v1 reviewer. The independent gate checks still require the complete original rules. This distinction is preserved here rather than concealing the initial audit mismatch.

## Preservation and scientific limits

Every exact input/run/source pin from the original v1 result review still matches, and that review receipt remains `76e224ced1ccd4eec2a6a591a628db86312e8a7023f9e0f01b3aba9dbc7c4a19`. Its failed2022/2024 six-scale gate is still a failure under v1's original output; v2 does not retroactively turn it into a pass. The new run used the original starts, identical data/law/prior/gtol/bounds/maxiter/gates and only the separately reviewed ftol1e-14 change. No old process was restarted or old file overwritten.

The refined five-scale2022 control SDs differ from the original selected MAP numbers by at most3.22e-8 in natural SD, consistent with a tighter numerical solution of the same objective; they are not claimed bit-identical fitted numbers. Slope/Brownian local inverse penalized-Hessian correlations remain about−.540567,−.591278,−.597294 for the three inputs. These are **local log-SD curvature approximations including regularization**, not empirical posterior correlations or proof of useful mechanism separation. Positive curvature, SPD geometry and optimizer agreement establish the numerical contract only. Scale uncertainty/shared-outcome dependence remain unintegrated conditional empirical-Bayes approximations.

The reviewed earlier-only inputs retain930/1218/1514 selected observations and the exact2022/2024 shared calibration identity, including all31 Texas2020 rows. They do not acquire a2022 training cycle for2024 or replace the source with the separate1483-row bank. Source/vintage and unknown respondent/sample-overlap limits remain unchanged. A lower training objective does not demonstrate forecast gains; this review computes none.

**Final disposition:** v2 calibration results may be referenced as passing their fixed numerical gates with the failed start visible. Any forecast comparison still requires its own Root-selected target/operator/bank/horizon/identity masks, Gibbs-appropriate diagnostics and matched A/B scoring contract. No current2026 adjustment, active-provider change, forecast superiority or public promotion is approved.

`check.py` and `results.json` preserve reproducible saved-point calculations, full exact inspected pins, original-v1 preservation checks, fit gates, source/alias identities, start statuses and numerical discrepancies. Reviewer optimizer/RNG entrypoints were blocked during all algebraic checks. No prior tests or actual fits were rerun. Ruff check/format pass on this review code.
