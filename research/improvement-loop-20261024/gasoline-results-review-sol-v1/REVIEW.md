# Independent G01 raw results review

Prepared at 2026-10-25 12:20 UTC. The frozen execution plan is `gasoline-execution-v1/prepared-v2/plan.json`, SHA-256 `63f6de0e57949857a48d02e5084af8cc9d576b4ffa96985b13afc938fe55b5ab`; its preparation plan is SHA-256 `ee3cdefc9b3a5f9bc98598acd47ef7db81d0393f32c874396843e587566849ef`. The owner runtime is read only. This directory alone contains this review's outputs.

`verify.py` is independent of the execution runner and model module. It derives target design columns from frozen label-free feature CSV fields and each job's frozen recipe. It recomputes the TWIP orientation and earlier unique-cycle scaler from the pinned gas panel, checks same H0/F0/G01 earlier race support, verifies that G01 adds only the prescribed gasoline mean term to F0, reconstructs every saved predictive logit from the original raw posterior rows and stored new-cycle/new-race random draws, and checks the share transform and F0 masks. Four raw chain banks per job are loaded with the certified ArviZ runtime to recompute rank Rhat, bulk/tail ESS, divergences and tree-depth hits. Stored predictive gate values are independently compared with the frozen thresholds, though their ArviZ estimates are not recomputed. Job/seal status and failed gates are preserved in `verification.json`.

The program inspects a fold only after all H0/F0/G01 prediction seals and the F0 masks exist and verify. It does not parse numeric target outcomes. It also defers even hashing the separate outcome CSV until the full attempt is terminal, so a partial-fold review need not open later targets' numeric bytes. File modification chronology is checked only as consistency evidence, not proof of first outcome access. Luna separately verifies score arithmetic. This is exposed retrospective A2018/22 and B2020/24 development evidence, not a current 2026 forecast, count-stack qualification or adoption decision.

Preflight at 2026-10-25 12:20 UTC: `verify.py` exited successfully with `reviewed_years=[]`, because no complete fold was yet available. The generated `verification.json` records the exact inspected source pins and actual UTC execution time. It will be replaced by the read-only verifier when a fold seals. To inspect a specific sealed fold, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/economic-v1-env/bin/python \
  /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/gasoline-results-review-sol-v1/verify-v2.py \
  --year 2022 --output verification-2022.json
```

The original `verify.py` and its `verification.json` are frozen with the 2018 receipt. Later reviews use `verify-v2.py --year YEAR --output NEW-FILENAME.json`, which creates the output exclusively and refuses to replace a prior report. It never samples, simulates, scores or modifies producer artifacts.

## 2018 sealed fold

At 2026-10-25 12:21:16 UTC, after all three prediction seals and the F0 mask were present, the independent 2018 review passed. It verified 98 source/artifact SHA pins, all twelve original raw chains (four per arm), 4,000 retained and 1,500 warmup rows per chain, and the eight unique earlier gas cycles with mean `0.06517138279346342` and standard deviation `0.1393466908752723`. H0, F0 and G01 share the same 236 earlier training races and numeric response binding. G01 contains precisely F0's providers plus one gasoline mean term with its Normal(0, 0.05) coefficient. The maximum reconstructed logit and share errors were both zero across all three predictive banks. Independently recomputed parameter diagnostics match stored diagnostics; each arm has zero divergences and depth hits and passes every fit gate. Stored predictive values also pass their frozen thresholds. `verification.json` contains exact values and pins. No numeric target outcomes were parsed; the outcome CSV pin remains deferred until the whole attempt is terminal.

## 2020 sealed fold

At 2026-10-25 12:26:51 UTC, the separate `verification-2020.json` passed the same raw review for 2020. H0/F0/G01 have identical 266 earlier training races and nine unique earlier gas cycles. All twelve 2020 raw chains preserve 4,000 posterior and 1,500 warmup draws; every independently recomputed parameter gate passes, with zero divergences and depth hits. Reconstructed predictive logits and shares agree exactly. G01's gasoline coefficient has posterior mean `0.0472064` and 5–95% interval `[−0.0064241, 0.0941860]`; these are descriptive joint-fit values, not causal or sign qualification. Numeric target outcomes were not parsed. `reviews-manifest-v1.json` pins the unchanged 2018 checker/receipt and the new 2020 checker/receipt.

## 2022 sealed fold

At 2026-10-25 12:31:24 UTC, `verification-2022.json` passed using unchanged `verify-v2.py`. All three arms share 294 earlier races and ten unique earlier gas cycles. Every raw chain and parameter gate passes, with zero divergences and depth hits; every reconstructed predictive logit and share matches exactly. G01's gasoline coefficient posterior mean is `0.0454331`, with a 5–95% interval `[−0.0005749, 0.0879413]`, descriptive only. No numeric target outcomes were parsed. `reviews-manifest-v2.json` pins the unchanged earlier reviews, their prior manifest, and this 2022 receipt.

## 2024 and terminal audit

At 2026-10-25 12:34:31 UTC, the frozen v2 checker passed for the 2024 fold alone. All three arms share 324 earlier races and eleven unique earlier gas cycles; 4,000 retained plus 1,500 warmup draws are present in each of twelve raw chains. Independently recomputed parameter gates all pass, with no divergences or depth hits, and every predictive logit/share reconstructs exactly. The G01 gas coefficient posterior mean is `0.0375902` with 5–95% interval `[−0.0070666, 0.0795347]`, a descriptive posterior summary. The separate outcome CSV byte pin is now verified because the attempt is terminal; no target numeric outcomes were parsed by this reviewer.

The independent `finalize.py` bound the four immutable fold receipts to all twelve completed job terminals, three seals and scores per fold, the original plan, and the top-level summary. It completed at 2026-10-25 12:35:11 UTC with zero failed jobs. The terminal SHA-256 is `f9922ed230ce99c83f88d8663f7ec1f3b97634e0707e4dd62ac1f91fc4ac4ec9`; the summary SHA-256 is `b72223532e63555332a3628ab4b9d4aeafd5a3f2d926f6111f126f7e8e673b22`. `final-audit.json` records every inspected pin, and `reviews-manifest-v3.json` (SHA-256 `f6fc7e60730598c8df70edf67ff1ef0508c2348cf2b8fde068a67d7d08c01d2c`) pins the full immutable review chain. No fitterror retry or failure artifact was found.

**Scope:** This review independently recomputed parameter rank Rhat, bulk/tail ESS, divergences and maximum-depth hits, and exactly reconstructed each stored predictive draw from raw posterior coefficients, label-free design and saved future noise. The stored predictive ArviZ estimates were checked against frozen thresholds but were not independently recomputed. Luna's separate work checks score arithmetic. Filesystem timestamps support the seal-before-mask-before-score order but do not authenticate first access to numeric labels. These are four exposed historical development folds in an aggregate fundamentals-only experiment; they do not qualify a count stack, current 2026 forecast, causal gasoline effect, or public-model change.
