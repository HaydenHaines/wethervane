# Independent strict trace-metadata contract repair review

This separate v2 review preserves sealed scorer v1, its original independent acceptance, and Root's failed actual v1 action. The failure occurred in admission before numeric label access. I missed the producer/consumer mismatch in v1: the invented reader fixture copied the expanded stored trace map into terminal provenance, masking the runner's caller-only provenance contract. The earlier 37 passing tests did not establish that native contract. This review records that escape rather than rewriting the original receipt.

## Observed cause, metadata only

Frozen Root admission `1cfc43cbab0a1f809d52f75831223846999489c5bce34ad26e5112c16c60dffd` selects 12 jobs. I independently verified their terminal, trace-receipt and provenance-JSON hashes and compared only source-pin maps. Every terminal has 16 caller pins; every stored trace has exactly those 16 plus the four fixed REQUIRED_SOURCE_PINS. No caller value is missing or conflicting. The four additions bind the observed sampler, marginal math, forecast design and correlated-MCSE source. `actual-metadata-contract.json` records every inspected metadata pin. No actual NPZ, forecast/reference numeric array, outcome source or actual score was opened.

The exact producer is run_prepared_job: it copies trace.REQUIRED_SOURCE_PINS, updates with provenance.trace_source_pins, then constructs the collector. The collector requires the constants already present; it does not invent them. V1 reader incorrectly passed only the caller map to load_trace's exact stored-map check. Source ordering confirms that failure prevents the verified admission capability and later label reader from being reached. Root preserves the failed invocation separately; no forecasts, masks or numerical thresholds require regeneration.

## Bounded production change

Math, CLI and runtime files are byte-identical to sealed v1. The only production reader delta imports the fixed source-pin constants, adds expected_trace_source_pins, and uses its result at the unchanged load_trace call. The helper reconstructs exactly fixed-plus-caller from authenticated constants and caller provenance, validates hash strings, rejects conflicting fixed-key collisions, and never derives expected pins from stored trace metadata. Native exact equality remains intact: this is not subset acceptance or removal of source checks.

My independent check_metadata_contract.py matches the helper's expected map against all 12 actual stored JSON maps. Conflicting fixed-key injection is rejected. Added or omitted caller keys do not match unchanged stored metadata. Those checks never call an actual array or label loader.

The revised synthetic fixture independently emulates the runner's constructor union while preserving caller-only terminal provenance. Its positive full gateway reproduces the contract that escaped v1. Regressions reproduce old caller-only exact-check failure, accept the reconstructed full map, reject collisions, and reject added/omitted caller keys without label access. The unchanged math/CLI proofs from v1 are reused; no broader scientific audit is inferred from this repair.

## Handoff and limits

README and template point to the v2 manifest/review paths and the exact runtime SHA `01ff4314a314fcb315719c0efb30062d9ee81878fa12ae428f7f1645f56cd354`. Root admission must make scoring_runtime, accepted runtime_receipt, scorer-manifest runtime entry and review runtime hash equal that value. The full gateway also compares runtime JSON values with the actual process. The prior direct-byte cross-link condition is unchanged.

Actual Root admission/label access remains a separate action. Root will dry-run complete verify_admission without labels before another scoring invocation. This acceptance is a strict source-contract repair and synthetic computational finding; it does not qualify actual scores, change a model, authorize a forecast rerun or threshold relaxation, or promote anything.

Final exact seal: `9cea17abf7b52a32f6817f210cab66e28f117a59a53b8d1032f2200c69c75eb1`, all 55 artifacts and 68 upstream bindings verified. All 40 sealed synthetic tests independently pass in 33.59 seconds under the accepted Economic runtime; the focused gateway/pin subset also passes (3 tests). The constructor-realistic regression is included in the complete 40-case run. Math, CLI and runtime byte comparisons and all 12 actual metadata-only map checks pass. No numerical prediction array, label source or actual score was opened by this review. Accepted exact strict metadata-contract repair at the reviewed runtime.
