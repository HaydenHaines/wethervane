# Independent read-only review: M01 calibration consumer v1

**Review date:** 2026-10-25  
**Scope:** Supplied-mean likelihood consumption only. No posterior inference, fitting, activation, or producer edits.

## Frozen artifacts and entrypoint checks

Reviewed `/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/calibration-consumer-v1/` as received. The consumer SHA-256 is `5c01ff21de18bd9c2f3bda144fbb5497d73a90a0f622e9bdb8ee1b0d894d2399`; report SHA-256 is `e7b8f0285feb0987b6bb7b7d40917aa3037bfa416a6b8829bcb9efa02ff59306`; output receipt SHA-256 after replay is `6383c7b1b6e54ae188221056341b681507d666c3f2793ddec7e518c1a47a9cd6`.

`load_bank` requires the exact independent-review digest and passing review flags, checks review-to-plan/terminal/NPZ hashes, checks terminal diagnostics and artifact links, rejects historical target years, requires cycle 2026 and an explicit UTC cutoff at/after the source cutoff, and requires the literal marginal-Bayes target. The plan and terminal explicitly remain unqualified for forecast/publication. The bank’s `(4,1000,5)` joint rows are retained without resampling and receive equal `1/4000` input weights; arrays are bytes-backed immutable. The loader checks finite positive SDs and exact `exp(log_scales) == standard_deviations`.

The exact actual TX 2026 Senate vector is bound through copied owner receipt, target-coordinate and poll-component digests. The six coordinate IDs align exactly with the six model row IDs via `source_map`, and original IDs follow the same order. All cycles and race IDs are checked as TX 2026 regular Senate; the owner provider is imported from the pinned local source snapshot. The saved fixed covariance is reproduced within absolute tolerance `1e-15`. No second MOE/sampling-error term is added.

The math boundary is explicit: `M01_marginal_bayes_complete_poll_vector_likelihood_v1` is an ordinary marginal likelihood, not a cut. The mixture uses equal input row weights, while its log-sum-exp likelihood naturally reweights scale rows conditional on the observed poll vector. It must not be described as preserving calibration mass after a downstream posterior update. A cut or other feedback regime would need separately defined normalization and target.

## Independent numerical replay

I ran the documented CLI using the isolated `calibration-v1-env` and then ran `test_consumer.py`. The CLI rewrote the local `report.json` and `receipt.json`; both hashes matched the pre-run frozen values after replay (`e7b8…59306` and `6383…a9cd6`, respectively). The CLI reproduced all three reported supplied-mean checks; the seven tests passed in 0.225 seconds. The density oracle uses SciPy multivariate-normal component log densities and `logsumexp`; the gradient oracle independently solves each component covariance system and averages component gradients using the computed responsibilities.

The largest density discrepancy was `8.88e-16`; the largest gradient discrepancy was `2.66e-15`. The fixed-covariance check reproduces the owner’s saved covariance and has reported maximum gradient discrepancy `1.33e-15`. Four-chain sensitivity ranges are about `0.0130–0.0140` log-density units at these supplied points. These are computational integration checks, not posterior uncertainty intervals or forecast evidence. All values and result bytes reproduced the frozen report.

## Integrity limitation before reuse

The initial reviewed version had an integrity gap: it did not verify `input-manifest.json` against a fixed digest before using the copied covariance provider and poll inputs. For that initial packet the manifest hash was `0fbec6366c2640f30a17c2fc576b87303d9f4065a632b1488b9a72311345fcb8`, and its files agreed with it. A changed manifest could have rebound a changed provider/input set. This finding is retained as the baseline for the repair delta below.

The report is clear that no downstream posterior, calibration cut, forecast qualification, or public activation occurred. No MCMC was launched during this review.

## Repair delta review

After the initial finding, Sol added `SNAPSHOT_MANIFEST_SHA` in `consumer.py` with the reviewed manifest digest above. `verify_snapshots` now checks the manifest bytes against that constant before reading its file pins. `actual_inputs` calls this guard before importing owner code. The new test copies the packet, corrupts the covariance provider, updates that provider’s hash/size in a copied manifest, and confirms rejection specifically at the reviewed-manifest digest before provider import. The saved `manifest-repair-before.json` and `manifest-repair.json` record the before/after hashes and unchanged numerical report.

I independently inspected the diff and reran only `test_consumer.py` in the isolated environment (no CLI replay or report rewrite): **8 tests pass in 0.239 seconds**. The new consumer hash is `f2f28207125b9474ddc92b22b75b814c962e415c3152df6a9170a15e4dd1fd70`; test hash is `92d51ff0953379f6888d1da8977a5d0ad0f4fea3a763c1c55cfb267799db530b`; the new receipt hash is `0ca1e62b7b4987ab7ea51bbab75a2983cdb58355de20db6809086467d58ca47b`. `input-manifest.json` remains `0fbec6366c2640f30a17c2fc576b87303d9f4065a632b1488b9a72311345fcb8`, and `report.json` remains `e7b8f0285feb0987b6bb7b7d40917aa3037bfa416a6b8829bcb9efa02ff59306`. The previously noted manifest-rebinding gap is closed for this exact reviewed snapshot; all other scope boundaries remain as stated above.
