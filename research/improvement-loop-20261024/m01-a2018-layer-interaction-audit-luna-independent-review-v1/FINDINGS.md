# Independent review of A2018 layer-interaction audit v2

Reviewed frozen v2 hashes: `check.py` `21b77fb5…cc87b3`, `test_check.py` `854165e3…bb920ea0`, `result.json` `da8b2f49…c980e22e8`, `source-pins.json` `5bf3fc92…b81b7d76`. All eight external inputs and all self-pinned packet files matched at review time. I copied v2 into `scratch/`, ran `check.py` and the eight tests under both normal Python and `python -O`; both suites passed. Running the check from its original directory reproduced the frozen result byte-for-byte. No source/owner files were changed and no model/graph/prediction arrays or row data were read.

## Material defect: large common offsets break the claimed stable log-ratio

The fixed ±1000 test covers overflow-safe exponentiation, but not cancellation when the log normalizer is formed as `max + log(sum(exp(...)))` and then subtracted from a utility near the same large magnitude. In a separate scratch probe, with weights `(1,9)`, pair utilities `((4,-4),(-4,4))`, and other utility `0`, v2 returns `-2.1942472351374973`. Adding the exactly representable common offset `1e16` to every utility yields `-1.9975697137614865`, a spurious `+0.1966775213760108` change. Per-profile gauge-centered arithmetic returns `-2.194247235137497` before and after the shift. The exact inputs and output are in `offset-counterexample.py` and `offset-counterexample.json`.

Thus v2's moderate invented fixtures reproduce but its general numerical-stability claim is false. Do not rely on these synthetic interaction magnitudes beyond the tested moderate scale. A follow-up should subtract each profile's maximum from its utilities **before** computing the profile log normalizer and candidate log-rate terms, then retain this offset-invariance regression.

## Source closure, leakage, and scope

The computation pins and reads the selected plan, aggregate ledger, selected `117-components.py` snapshot, preserved legacy result, and prior review/test artifacts. It uses explicit `require`/`ValueError` guards rather than `assert`; an updated test mutates bytes after hashing and verifies the stale hash fails in an actual `-O` subprocess. All declared external source pins matched. Result JSON contains no poll/survey identifiers, respondent tokens, raw values, or row IDs.

The closure check verifies pinned aggregate counts/flags and source-expression anchors; it does not independently recalculate the ledger from raw observations or build/evaluate the compiled graph. The identified formula is the correct aggregate ordinary D/R poll ratio when response and event-G are common positive scalar multipliers. This does not transfer to other factor paths. The `(1,1)` and `(9,1)` relative profile weights, utility vectors, and resulting interaction are invented, not fitted participation states. Centering pairs is not evidence of posterior support or native attainability. The packet appropriately disclaims source lineage, calibration, qualification, and layer ranking.

One minor manifest clarity issue: `source-pins.json` lists the prior luna-v1 result and prior review `source-pins.json` as inputs, while `check.py` reads a pinned byte-copy of the former and does not consume the latter. The actual files it does read are pinned in code, so this is reference/manifest clarity, not a bypass of the current calculations.

The frozen v2 packet is unchanged. Reproduction logs, source copy, exact counterexample script/output, and machine-readable findings are preserved alongside this report.
