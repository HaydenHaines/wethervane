# 2018 NUTS remedy research correction

This separately versioned addendum corrects the interpretation/decision language in the prior research packet. It does not overwrite or change that packet. No fit, replay, graph build, sampler diagnostic or owner-file modification was performed.

## Corrected wording and limits

The earlier packet's phrase **“highest-information”** is unsupported as a comparative ranking of possible remedies and is withdrawn. C1 should be described only as a **small, predeclared sampler-only probe**, conditional on review of the pending owner geometry analysis and a fresh source/resource admission. This ranking correction leaves the candidate itself unchanged: increase only `target_accept_prob` from 0.97 to 0.995, with all other model and experiment settings fixed.

A zero-divergence C1 run would meet the existing *observed-run divergence-count condition* for that finite run, if its other diagnostics also pass. It would not prove that the target has zero underlying divergence probability, establish that the model's divergence cause was finite-step resolution, or demonstrate a general causal cure. A future run with zero flagged transitions may still have unvisited regions with high curvature; independent chains, remaining diagnostics and the preexisting full gate remain required. Even a toy IID “rule of three” would not apply directly to these dependent and heterogeneous MCMC transitions, so this packet does not translate zero observed flags into an inferred posterior divergence probability.

The coordinator's current gate tests finiteness of `r_hat`, `ess_bulk`, `ess_tail` **and** `mcse_mean`; it additionally bounds R-hat by 1.01, requires bulk/tail ESS at least 400, and requires zero divergences and zero depth saturation. It has **no separate numerical MCSE threshold**. The candidate plan must not imply one.

The matched-control design holds scientific inputs, target, model, seeds/initialization scheme, chain count, warmup and retained draw counts, sampler, backend and diagnostics fixed while changing one sampler target. This is a controlled comparison of protocols, **not a pathwise paired-chain experiment**: changing step-size adaptation changes the trajectory, so sharing seed values does not yield identical or paired transitions.

At the proposed fixed sampling budget, there are 8 parents × 4 chains = 32 chains; each has 1,500 warmup and 1,000 retained transitions. The total is 48,000 warmup and 32,000 retained transitions, or 80,000 chain transitions. This does not fix leapfrog/evaluation counts or wall time: smaller adapted steps can lengthen NUTS trajectories, and max tree depth remains 12. Any depth saturation fails the unchanged gate, and runtime/ESS-per-second should be reported as secondary efficiency outcomes. Do not increase the depth cap in this one-factor candidate.

## Corrected candidate text

> **C1, conditional on the pending geometry review:** test one sampler-only change, `target_accept_prob=0.995` instead of 0.97, for all eight fixed-mass parent conditionals. Hold the original model, data, parent masses, seeds and initialization scheme, 1,500 warmup, 1,000 retained draws, four chains per parent, max tree depth 12, runtime/backend and all diagnostic gates constant. This tests whether a smaller adapted step size is sufficient to remove the observed divergences. The predeclared outcome is gate pass/fail, not proof of a zero underlying divergence rate or causal diagnosis. Any retained divergence, depth saturation, or other failed diagnostic means C1 does not qualify. Report the variable compute cost and ESS/sec; do not iterate target settings in the same candidate.

The task-specific independent challenge by Sol agrees that the A2018 graph already has non-centered pooled, cycle, race and drift blocks; selects no TX ballot-role hierarchy; has empty observed-total coordinate lists; and already adapts step size and a diagonal mass matrix. It cautions that current coordinate leader summaries do not establish cause. This is preserved at [`national2018-remediation-challenge-sol-v1/README.md`](../national2018-remediation-challenge-sol-v1/README.md), SHA-256 `522113b4762bd1ef4090f77210ee611a3d11cb84674e59d2c32971653d1ae646`.

## Cross-pins

- Prior Luna research packet preserved unchanged: [`m01-2018-nuts-remedy-research-luna-v1/README.md`](../m01-2018-nuts-remedy-research-luna-v1/README.md), SHA-256 `b1b9ae0870c72192998ea818da79c08537a77899b17c189836cf73280fb2f8ff`; [`evidence.json`](../m01-2018-nuts-remedy-research-luna-v1/evidence.json), SHA-256 `8f2b110e36223128b68d781930b88627a829e0965f140a6da69c29a9d2f262f0`.
- Sol independent challenge README, SHA-256 `522113b4762bd1ef4090f77210ee611a3d11cb84674e59d2c32971653d1ae646`; source pins SHA-256 `bb0927c36e869911e5d6ec55a8fd5d44229501e875b1c54c744cdcd743009a67`.
- Actual coordinator snapshot contains the exact finite-MCSE gate logic: `.../gpu-production-v1/attempt1/source-snapshots/156-coordinator.py`, SHA-256 `b195e21a307f6d19c9a6e486150a07c51a96159343602258140f4db7b6bb2c13`, `diagnose()` lines 72–89.
- Original 2018 plan SHA-256 `fd44e4362c6c0a35e98aae23750b2266d4ea845aecea74de3970bb358a053209`; retained-divergence audit result SHA-256 `21a7f534922c91618d8a92d5b2075dcc9d48b381b07a199f2a989d56d6d72089`.

This addendum only narrows claims and clarifies the design. It does not admit or execute C1.
