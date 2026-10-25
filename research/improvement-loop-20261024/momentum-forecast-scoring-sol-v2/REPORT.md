# Score reader v2 — strict constructor source-pin contract repair

2026-10-25. **40 synthetic tests pass in25.18s; independent exact-v2 acceptance pending.** Sealed scoring v1 manifest `cd9eb4b4e8bd6869e6a34d45b96ba137624df271d44e57c0aaae99b5f5ff3b9c` remains unchanged. This is a source-contract integration repair only: `score_math.py`, `score.py` and `runtime.json` are byte-identical to v1. No probabilities, scores, statistical thresholds, masks, streams, selected attempts or model/source-fitting law changes.

Root's actual v1 scoring invocation stopped before label access with `ValueError('Trace source pins mismatch')`. Root's preserved failure record SHA is `59b84a975e27e25b6407d8034e0a5f1c9e4e5fba8fb6a65a5fd172cfdf0c09de`; no output directory/launch/label-access was created. No sampling is repeated or authorized by this repair.

## Metadata-only diagnosis

Root explicitly authorized reading actual metadata for this investigation. I read only the SHA-bound admission, selected terminal JSON, trace receipt JSON and trace provenance JSON. **No actual prediction NPZ, actual forecast/reference numeric arrays, target outcome columns or actual scores were opened.** All numerical/scoring tests remain invented supplied fixtures.

Every one of the12 selected jobs has16 job-specific caller pins in `terminal.provenance.trace_source_pins`, while the stored trace map contains exactly those16 plus the4 fixed `REQUIRED_SOURCE_PINS`. There are no missing caller values or conflicting overlaps. `actual-metadata-only-diagnosis.json` records the pin-only comparison and verified JSON hashes without any numeric prediction arrays.

The collector does **not** add fixed pins automatically. The sealed runner explicitly builds `dict(REQUIRED_SOURCE_PINS)` then updates it with its caller map before invoking the collector, whose constructor requires those fixed identities. V1's consumer passed only the caller map to the loader's exact equality check. V1's synthetic gateway fixture had copied the complete stored map back into terminal provenance, hiding the actual producer/consumer distinction. Independent review also missed that fixture escape; it is preserved explicitly rather than attributed to failed scientific inference.

## Strict repair

`expected_trace_source_pins(caller_pins)` reconstructs the exact runner constructor union from the immutable fixed source/design constants and caller-declared job pins. It validates SHA256 values, rejects any conflicting fixed-key collision and passes the complete expected map to unchanged `load_trace`. Stored metadata is never used to define expected identities; no subset or wildcard source check is introduced. Extra/missing caller keys still fail complete stored-map equality.

The revised synthetic fixture independently retains its caller-only terminal map and forms the fixed+caller map before invoking the real collector constructor. Added tests reproduce v1's mismatch under that realistic contract, verify full gateway success through the repair, reject fixed conflicts and reject unrecorded/omitted caller pins before label access. An already-complete consistent caller map remains supported without weakening identity.

All previous37 mathematical, mask, source/runtime, immutable-capability, dimension, retry-proof and orchestration tests remain. Final40 tests pass under the exact accepted economic interpreter/package profile with no installation. Existing lab pytest is appended read-only after economic site-packages, plugin autoload/cache disabled, bytecode disabled, one BLAS/OMP thread. Ruff check/format pass. V2 had no failing numerical test; prior v1 failure/test evidence remains intact.

## Handoff

Use the v2 README/CLI and admission template, not the v1 paths. Root must create a new externally hashed admission/action with the **same12 forecasts/reference/mask/selection pins** and new scoring manifest/review pins. Root's planned prelabel-only `verify_admission` run can validate actual inputs without calling `read_labels_after_admission`; labels remain closed until its separate scoring action.

Root admission validation must preserve the already agreed exact runtime closure: v2 `scoring_runtime` equals this packet's `runtime.json` pin SHA `01ff4314a314fcb315719c0efb30062d9ee81878fa12ae428f7f1645f56cd354`, equal to the manifest runtime entry, independent review runtime SHA and accepted execution runtime receipt. Runtime JSON values are also checked transitively by the gateway. This condition does not require changing the byte-identical CLI or runtime.

All v1 scientific/scoring limits remain: conditional aggregate D/R share comparison only; no whole-chamber/count-stack claim, no promotion, no score-triggered sampling, inherited-bank/calibration-scale uncertainty separate, archived F0 WIS source-chain dispersion only, empty required cycles unavailable. The original mathematical verification remains applicable because its code bytes are unchanged. Independent v2 acceptance precedes Root's actual admission/scoring retry.
