# Private exact computation preflight, version 2

This isolated frontend experiment uses invented inputs only. It does not qualify or expose the Nevada persistent-turnout challenger, create catalog options, parse a scientific packet, authenticate an owner, fit, queue or execute anything. No production module imports it. Production catalog, composition types and resolver remain unchanged.

`contract.mjs` exports `matchPreflight(metadata, catalog, request, expectedOwnerPin)`. The optional metadata schema is `ComputationPreflightMetadataV1`; the readable invented fixture in `contract.test.mjs` specifies its exact shape. It binds the catalog ID/source SHA256, owner handoff ID/SHA256, per-slot provider/version/instance identities, target, law/source-manifest/output-contract ID/SHA256 references, evidence descriptions and explicit `joint_refit` requirements. Artifact references are opaque: these field names describe frontend metadata, not a proposed producer packet.

Each binding declares exactly one year, date and complete slot selection. Multiple rows support multiple dates/years without cross-product authorization. Metadata cannot introduce options: every selection must already exist in the supplied catalog and support the declared year. The matcher accepts arbitrary slot counts, rejects extra/missing keys, and requires complete matching instance keys. A supplied preset is rejected because only the unchanged saved resolver can verify it.

All supplied declarations are checked before matching. Missing metadata yields `none`; malformed declarations or provenance mismatches yield `invalid`; duplicate exact tuples anywhere yield `ambiguous`, including duplicates outside the requested tuple. A valid but unmatched selection yields `none`. Only one exact valid row yields `match` with lifecycle `requires_computation`. Every outcome has disabled execution and null configuration/result/run identities. Matched metadata is cloned before return.

The caller-supplied expected owner pin establishes string/digest consistency only. It does not prove author identity, artifact availability, source vintage, admission, numerical validity or forecast skill. The helper does not fetch referenced bytes or cryptographically authenticate their content. Final producer schemas, admission and authenticated source-bound adaptation remain required.

Future integration must preserve saved-result/preset precedence, refuse to rescue stale saved results, reject overlaps with legacy computation metadata, and explicitly handle dates absent from saved registry runs. None of that production wiring is implemented here. Existing legacy preflight supports exactly base/polling/uncertainty; this experiment leaves its behavior intact.

## Validation

Run from the lab root:

```sh
node --test research/improvement-loop-20261024/native-preflight-contract-v2/contract.test.mjs
node --experimental-strip-types --test tests/composition-resolver.test.mjs
```

68 private checks pass, including exact four-slot matches, multiple dates, key ordering, unsupported selections, all identity pins, source references, duplicate ambiguity, malformed unrelated declarations, explicit joint refit, immutable inputs/outputs and disabled execution. All 7 existing resolver checks pass. Only these small suites ran; no build, fit or large scientific test ran. `tests.tap` and `legacy-tests.tap` preserve output. `evidence.json` records artifact hashes and identical before/after production file pins.

## V2 runtime boundary

V1 is preserved separately. Independent review found that sparse arrays and function-valued extra fields could throw under its broad runtime-input claim. V2 adds a bounded plain-JSON-tree boundary before semantic validation. It rejects sparse arrays, extra array properties, accessors, symbol/nonenumerable properties, undefined/functions/bigints, nonfinite numbers, cycles, non-plain objects and hostile proxy failures. Ordinary shared references and null-prototype records are allowed. Each input is limited to 10,000 visited nodes and depth 64; these conservative private parser limits are not scientific limits or final API requirements. Absent metadata still returns `none` without inspecting unused context.

The metadata schema identifier remains `ComputationPreflightMetadataV1`: V2 versions this private implementation and its evidence, without changing the JSON metadata shape. The top-level exception boundary converts malformed runtime failures into disabled `invalid` results. No actual owner packet or implementation is qualified by these tests.
