# Independent review: A2018 source-evidence resolver

## Verdict

No blocker found for the stated narrow result: **prepared mappings audited; full atom admission blocked**. The resolver pins and checks the supplied frozen inputs, verifies their internal plan/context/closure relationships, and returns aggregate-only digests. It does not authenticate original survey/poll measurement atoms, sample independence, historical public availability, or an executed model graph. Those limits are both explicit and reflected in its false status flags.

## Reproduction and challenge

The frozen source hashes are recorded in `source-pins.json`. I copied `resolver.py`, `bindings.json`, and `test_resolver.py` into fresh scratch and ran:

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest test_resolver -v
```

All 10 tests passed. I separately ran the resolver from scratch; parsed output exactly equals the frozen `result.json` (SHA-256 `4088421f06d61d0762176a3f2d4c716bfe10bc71d9db6d6a0871b9e3482b1403`). That reproduces the reported 65 survey blocks / 7,968 memberships, one covariance block with 274 ordered rows (20 AZ rows), and the explicit `covariance_positive_definite_verified=false` status. The reproduction output is `reproduced-result.json`.

The additional scratch probe (`fake-metadata-probe.py`) distinguishes the helper from the resolver path. `inspect_prepared()` accepts a well-formed in-memory edit and computes a digest, as its docstring and packet documentation warn; passing the same edit as changed bytes to `resolve()` is rejected as `pinned_source_bytes_differ`. An out-of-contest candidate is rejected as `poll_candidate_target_mismatch`. The probe emits only status codes/booleans and no row identifier or value; its output is in `fake-metadata-probe-result.json`.

Existing tests also exercise row ordering (digest changes), malformed/nonfinite/asymmetric covariance, duplicate keys, unsupported candidate targets, all pinned-role tampering, and privacy of the returned JSON. An indefinite symmetric covariance with positive diagonal is accepted by the limited shape inspector, while the result correctly leaves positive definiteness unverified. This is a declared limitation, not a false PD claim.

## Source and claim boundaries

The nine-role binding pins the exact plan, context, problem, selected component source, rank/open contributions, aggregate, checker and closure adapter by byte length and SHA-256. The resolver checks context/problem equality, plan identities, source closure and aggregate counts. The prepared-vector digests bind the serialized mapping/order representation that was inspected; a hash is not evidence that the original source's questions, people, poll sample, collection date, or release vintage were authentic or independent. `available_utc` is only a pinned supplied metadata claim. Likewise, module imports and reads are not an atomic hostile-workspace snapshot; the packet states a stable trusted-workspace assumption.

The result does not overclaim graph or scientific evidence: `actual_compiled_factor_graph_authenticated`, `original_source_observation_atoms_authenticated`, `exact_source_version_publication_times_verified`, `respondent_or_sample_independence_verified`, and `scientifically_qualified` are false. The full generic row audit remains blocked for missing raw-field lineage and chronology. Covariance conditioning/positive definiteness and transformed-density execution are not established. The fixed eight parent masses remain an aggregate-contract check, not evidence that a posterior or compiled likelihood was executed.

No row-level tokens, masks, measurements, covariance entries or respondent data were copied into this review packet. Only aggregate result metadata and hashes/statuses are retained.

## Scope

Review and reproduction only. No model import/build/fit, owner job, private respondent data export, or change to the frozen resolver packet.
