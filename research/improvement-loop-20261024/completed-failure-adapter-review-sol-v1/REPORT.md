# Independent review: completed first conditional failure adapter

## Exact native run

The pinned terminal (`FullActualFitTerminalRootV1`, SHA-256 `32b7c1de3904fefed9be105cb11fe68f4fd1e97dc183c710fec623868cca1fb3`) and exact plan (`adef60442a7097dc581323f9e361bb2a75441e9292eb92912c2fa1a0900b61b1`) pass `inspect_completed_failure`. All eight terminal inventory hashes match their source bytes. The plan contains eight distinct parents, each mass 0.125; the native fit completed only parent 0. The retained archive is byte-pinned but never unpickled by this review. The 1,376 diagnostic CSV rows independently yield max R-hat 1.3208674745084075, minimum bulk ESS 9.879545388653126, and minimum tail ESS 11.409113169190775. The plan gates are max R-hat 1.01 and minimum ESS 400. The authoritative terminal exit is 1; both terminal and sampling budget say the budget was not exceeded. The adapter reports numerical fit failed, mixture incomplete, qualification false, no automatic retry, and publication unauthorized. It does not infer a timeout or a completed eight-parent fit.

The CLI accepts the exact versioned request in `request.json` and emits `cli-output.json`. The revised adapter pins 14 files: the eight terminal inventory files, terminal, plan, admission, and three observer/progress files declared by the fit. A small synthetic fixture, with inert placeholder pickle bytes, attached those 14 source artifacts plus one normalized report to a separate ledger under this review directory; repeating the same command returned the same artifact and added no checkpoint. No native pickle was executed and no owner source was modified.

## Generic contract findings

The initial adapter accepted these four *rehashed synthetic fixtures*. The author then added guards, and the final adapter rejects each. They do not change or contradict the pinned native run above:

1. `fit.returned_inference_preservation_error="archive failed"` alongside an archive marked complete now fails with `inference preservation failed`.
2. `result.coordinate_version` differing from `plan.provider.coordinate_version` now fails with `coordinate identity mismatch`.
3. A terminal JSON stored under an unrelated directory but pointing to the eight attempt files now fails with `terminal attempt directory mismatch`.
4. Removing every terminal diagnostic key now fails with `terminal diagnostics keys mismatch`.

The adapter now verifies the declared fit progress and observer inventories as bytes. It does not verify the 95 plan source pins or the semantic contents of the pickle. Its report correctly says transitive scientific ancestry is not replayed and diagnostics are not recomputed from draws. Downstream users should read `verified_artifacts` as the exact 14 verified files, not as an ancestry audit. The terminal's tool-handle observation remains an owner claim; the normalized report explicitly says it was not independently polled.

## Files

`independent_checks.py` reproduces native byte/CSV checks and the four synthetic counterexamples without unpickling. `independent-checks.json` records their final rejection. `fixtures/`, `attachment-journal/`, and `attachment-journal-repaired/` contain only synthetic adapter checks; `attachment-checks.json` records the repaired attachment result. `receipt.json` pins this review and the adapter code version inspected.
