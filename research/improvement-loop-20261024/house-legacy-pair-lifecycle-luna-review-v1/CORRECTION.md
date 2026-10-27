# CORRECTION — runner version and receipt claim

This addendum corrects a version-binding error in `REVIEW.md`; that file is retained unchanged. Its line claiming that the runner pinned as `66924c1f…` wrote `receipt.json` is wrong. The source hash in the review and that observed behavior came from different runner revisions.

## Exact versions and evidence

- The current `runner.py` SHA-256 is `66924c1faf1641384ff8c35f3774e3e3dc68dea8d95929565444b6db216629f1`. Its source writes `checker-pending.json` with status `checker_completed_pending_scope_teardown`; it explicitly says this is not a validation receipt and does not write `receipt.json`.
- The owner reports the earlier receipt-writing version had SHA-256 `1ad3a27f…` (full hash was not provided to this review). That earlier implementation is not the current runner.
- My original disposable probe used ordinary `import runner`; I did not capture the imported module's code hash at that moment. Python may load timestamp/size-based cached bytecode, so I cannot reliably bind the probe behavior to either the source then on disk or a specific source hash. It observed a successful path that wrote a receipt while its separate-session disposable child remained live, but it does **not** establish that SHA `66924c1f…` did so. The prior review's pinned attribution is withdrawn.
- Current `test_runner.py` SHA-256 is `dc3c8b1c93d0f788b80266ffc0c24f4be70dea2f23995f6c49501d158bd6d246`. It has seven tests, including an escaped live descendant case. I re-ran all seven while injecting a `runner` module compiled directly from the current `runner.py` source (bypassing `.pyc`); all seven passed. The escaped-child case confirms the current runner emits the pending record and no `receipt.json` while the separate-process-group child is still live.
- The earlier reported six-test run was against an earlier six-test `test_runner.py` revision and was not a source-pinned validation of the current runner. Its result must not be applied to current source.

## Corrected lifecycle conclusion

The underlying process-lifecycle observation remains: the current runner can return its **pending** record while a checker descendant in a separate process group is still alive. It does not thereby issue a validation receipt. The current tests explicitly document this behavior and clean up their disposable child. External cgroup/service-manager teardown remains necessary before treating the scope as fully stopped; the separate systemd fixture supports only its own tested service configuration. No claim about current validation-receipt creation follows from my unpinned probe.
