# Matched H14 forecast execution implementation

This is a runner implementation for the root-selected conditional empirical-Bayes aggregate D/R experiment. The author executes supplied synthetic fixtures only. Root launches every actual target job and the separate prelabel reference action after independent acceptance, using an externally SHA-bound receipt. Numeric target labels and scoring are absent from this packet.

The unchanged scientific selection is `momentum-forecast-design-v1.json` (`45707d4a…5678`). The before-run numerical clarification is `momentum-numerical-implementation-clarification-v1.json` (`0d6dd971…4024`). They govern the exact FS5/FS6/LP5 budgets, priors, data, streams, diagnostic thresholds and reference laws. The accepted adapter retains each complete original four-chain × 4,000-row F0 bank; the private observation-only sampler and trace collector preserve its constructor identities and native transitions. The paired-tail implementation is source-bound only after its separate v2 independent acceptance; the rejected v1 remains preserved separately.

## Root execution interface

Create a unique root receipt from `job-receipt-template.json` or `references-receipt-template.json`; fill the frozen execution-manifest digest, exact independent execution-review pin, unique run ID, shared matched execution-group ID, and new output path. Templates are not executable receipts. The root receipt itself must be externally SHA-bound on invocation:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/economic-v1-env/bin/python \
  /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/momentum-forecast-execution-sol-v1/runner.py \
  --receipt /absolute/root/receipt.json --receipt-sha256 EXTERNALLY_RECORDED_SHA256
```

The dedicated economic runtime's executable invocation path, prefix, Python version and relevant package inventory must match `runtime.json`. Frozen source inventories, exact dependency acceptances, private module identities and the execution plan are checked before scientific input use. The launch records authentic PID/PPID and thread environment. Root records the actual tool/session handle separately after launch returns; the runner cannot know that handle ahead of time.

The CLI exposes one selected action per receipt. A `job` is one cycle/arm/attempt with four chains. Attempt0 has warmup 1,000/draws 4,000 per chain; attempt1 has warmup 2,000/draws 8,000. The exact seed is `int(SeedSequence([2026102501,cycle,arm_code,attempt]).generate_state(1,dtype=uint64)[0])`, with FS5=1, FS6=2 and LP5=3. The unchanged sampler spawns its four child streams. Every output path is exclusive. No output reuse, automatic retry, seed search, score-triggered change or elapsed-time retry exists.

Attempt1 requires externally pinned terminal receipts for all three completed attempt0 arms. It checks their exact settings, shared execution group, current runner/source/runtime/review identities, complete bank source identity and per-arm target observation/calibration pins. The numerical trigger is recomputed from the selected core/bulk/bank gates. Exception/timeout, small-tail-only, interval-display-only, inherited-bank-only and post-label score failures do not authorize another attempt. All three prior terminal pins persist in the new provenance. Root must select the matched larger trio rather than concatenate attempts or choose by scores; a later score reader must independently enforce that selection.

## Per-job artifacts and failure handling

Each job writes `launch.json` before numerical input preparation. Source/axis failure after launch leaves a failed terminal. A successful sampler return writes every original array, including complete predictions for all bank races, to `original-predictions.npz`; non-array original fields go to `original-metadata.json`. `original-receipt.json` binds both. No race, coefficient, sampled bank row, original source-chain/draw identity, or raw unpolled prediction is dropped.

`trace/trace.npz` and `trace/provenance.json` preserve the admitted observer's exact phases and authentic per-chain final RNG states. `trace/receipt.json` is SHA-bound in `prediction-seal.json` before deterministic numerical processing. Original outputs and the complete trace remain preserved even if quadrature or diagnostic processing fails afterward. The sampler does not expose warmup trajectories; its selected warmup count is recorded.

Every 100 completed-sweep observer records, exclusive partial NPZ/JSON receipts preserve initialized prefix deltas and bank-mass aggregate updates (replace original batch positions rather than sum repeated updates); additional records preserve emitted final RNG states and caught failures. **The completed-sweep callback runs before missing-race RNG and original predictive output assignment.** A completed observer record does not certify a completed original raw draw. Partial schemas explicitly state this, identify saved delta ranges/current expected phase, and contain only actually emitted RNG states. They cannot restart a sampler. Missing terminal receipts after a hard process death remain incomplete; the parent process record is separate evidence, never a synthetic success.

`prelabel-components.npz` contains conditional expected shares and both component log tails on common-polled races, each with `[forecast_chain,retained_draw,common_race]` axes, plus ordered common race IDs/indices and seven quantiles `[common_race,quantile]`. Components are the pre-state conditional Gaussian laws: each conditions on incoming row/lambda and integrates Gaussian state noise. Their equally weighted empirical average integrates retained row/lambda states. Categorical row weights are never applied a second time.

`prelabel-summary.json` keeps these numerical scopes separate:

- Core state: observed logits/shares, all six coefficients, both scales, cycle/cycle-z and log post-gamma lambda; maximum rank/folded split Rhat 1.01 and minimum nonconstant mean-ESS 400.
- Bulk functions: share-mean MCSE≤.001 and each side's absolute win-probability MCSE≤.0025. Both batch 50/100 passes remain visible; the maximum MCSE is used.
- Per-side tails: relative MCSE≤.05 and chain log difference≤.15, using the accepted complementary smaller-side coordinate. Original one-sided statuses remain visible. No floor or large-side cancellation is substituted.
- Interval display: CDF MCSE≤.0025 and total mapped share bracket width≤.005. The selected method freezes pointwise MCSE at the nominal quantile and inverts at q±3MCSE. It is a local first-order display diagnostic, with no variable-MCSE envelope, confidence bound or calibration guarantee. It does not itself trigger sampling or determine score qualification.
- Bank integration: forecast-chain total variation≤.05 and original-source-chain mass difference≤.02. Nonuniform source-chain mass is allowed. Concentration warnings remain separate from integration gates and do not become a claim that extra forecast draws repair inherited bank support.

Every required gate key must exist and be true, except an explicitly declared valid structural exemption. Unresolved constants and missing/null gates do not pass. `terminal.json` distinguishes completed prelabel work (with a separate core/bulk pass flag) from caught failure; it binds original, trace, numerical and partial receipts. There is no single promotion flag.

## Separate prelabel references

The `references` action authenticates all four original archived F0 files through the accepted adapter and then, only on the root-authorized action, decodes their original numeric shares. It creates exact empirical CDFs/generalized-inverse quantiles (`sorted_x[ceil(N*q)-1]`) and freezes inclusive competitive [.20,.80] and near-even [.40,.60] archived D-win masks intersected with the fixed common-polled masks. It retains all original unsorted shares `[4,20000,bank_race]`, sorted pooled shares `[80000,bank_race]`, and five-replicate row-clustered functions `[4,4000,bank_race]`. The full original parameter bank still has 16,000 rows, never 80,000. Source-chain dispersion and constant-tail uncertainty remain explicit. No new F0 integration, draw, refit or mask change is performed.

PO5 uses exactly the FS5 observations/covariance and the selected independent final-logit `Normal(0,6.25)`. Cholesky solves compute its complete Gaussian mean/covariance; relative solve residuals must be≤1e−10. Mean shares use shared-node vector quadrature; they are not `logistic(mean_logit)`. It has no Monte Carlo Rhat/ESS or quantile-influence requirement. A later truth on a quantile remains a valid deterministic WIS evaluation.

Reference artifacts are `YYYY/archived-F0-{cdf.npz,summary.json}` and `YYYY/PO5-{analytic.npz,summary.json}`, bound by the separate terminal. F0 quantile arrays have `[quantile,bank_race]` axes; PO5 arrays contain `[bank_race]` means and `[bank_race,bank_race]` covariance. Completed reference artifacts stay pinned if a later fold/reference fails. Earlier-training frequency references are deferred to a separately bounded earlier-label reader; this packet contains no target or historical numeric label loader.

## Author verification and limits

Run the synthetic suite from the lab root with the runtime/environment above and `python -m unittest discover -s research/improvement-loop-20261024/momentum-forecast-execution-sol-v1 -p test_execution.py -v`. It blocks real adapter loading during synthetic sampler tests and supplies all archived-reference values synthetically. It exercises original/trace persistence, every observer failure phase, input-preparation failure, deterministic post-seal failure, external pins, source/runtime drift, seeds/budgets, matched retry rejection, scalar conjugacy, vector integration, gate presence, empirical source-row ordering and constant-tail handling. Initial logs and the deliberately retained synthetic failures remain evidence.

This is a retrospective aggregate conditional empirical-Bayes implementation. Original archive availability proxies, finite historical-bank uncertainty, shared earlier outcomes and fixed MAP-scale uncertainty remain inherited limitations. Revisited cycles are development evidence. No population-cut/count-stack qualification, chamber claim, public default change or forecast-superiority statement follows from author tests or computational acceptance.
