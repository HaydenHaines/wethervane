# Independent follow-up review: House weighted research wrappers

Reviewed commit cd5dec180a8786dd7acb5ed67d573a7ad9802cec against 9a4dc1adaefafb8d6979d805a239dcbe1ad5ed0b, using a git archive in /tmp. The owner worktree and running forecast jobs were not modified or read as execution inputs.

The targeted joint and pair stage suites pass: 18 tests, 0 failures. The follow-up fixes the two V1 lifecycle findings: it serializes JSON before spawning, and adds an intrinsic two-hour default/max deadline with an optional shorter positive deadline. Cyclic JSON tests verify both wrappers reject before their marker decoder starts. The silent SIGTERM-resistant fake-child tests use a 500 ms deadline and verify that SIGKILL cleanup leaves no live direct child.

I also ran a separate stdin-failure probe against the joint wrapper. A fake decoder closes stdin while a 2 MB request is being sent; the wrapper rejects with EPIPE, and the controlled child is reaped (ESRCH). The pair wrapper uses the same handling path, but I did not separately force pair stdin EPIPE.

No material subprocess blocker found for the bounded research wrapper use. The default two-hour duration is source-checked, not waited out in tests. Cleanup escalates for the direct child and reports if it still cannot close; it does not establish control of arbitrary descendants or OS-uninterruptible processes.

This does not change the prior identity boundary. The wrapper checks that the supplied forecast identity matches the stage preflight, but the result summary itself has no independently authenticated production identity. The separate identity audit found existing replay stages without the new identity object and leaves legal map/scenario reconciliation unresolved. These wrappers remain research consistency checks; outputs have publicationAdmission:false and do not qualify or release a forecast.

Exact source and artifact pins are in source-pins.json; full test output and the reproducible stdin probe are retained alongside this review.
