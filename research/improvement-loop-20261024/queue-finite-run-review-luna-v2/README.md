# Independent review: finite-run queue integration V2

V2 closes the inherited queue-admission path-containment gap for the tested queue path. I reran the integrated suite (22 tests, 9.576 seconds) and ran an independent admission-only probe. Same-plan outer/sibling/child combinations reject; accepted and crash-left `ADMITTING` claims reject ancestor, descendant and exact overlaps in either direction; the lexical sibling `/owned-other` remains admissible. No output paths or worker jobs were created by the custom probe. The inherited authenticated prelaunch-resume test still passes.

The protection is explicitly queue-admission-only. Direct calls to the unchanged executor still use a hash of the exact canonical output path for claims, so they do not prevent ancestor/descendant overlap. That limitation is correctly documented and remains material only if direct executor use is in scope.

The prior V1 lifecycle review still applies: finite-run state, sealing, reconciliation and final receipts are bounded research code, not production readiness or a scientific claim. Review pins and the reproducible probe are in `review.json`.
