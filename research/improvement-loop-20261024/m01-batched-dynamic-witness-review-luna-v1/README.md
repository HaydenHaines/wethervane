# M01 batched dynamic-witness design review

A single dynamic graph with exact exposure replacement can produce the same pointwise values as separate per-parent dynamic-witness builds. Batching changes execution and provenance scope, not the target, provided each call binds the right full parent record/exposure to the same immutable saved point, and the comparator requires the complete roster. Dynamic witnesses remain pointwise checks, not normalized conditionals or posterior inference; each original parent mass stays at 0.125.

The minimum six-parent roster is 42 calls: per parent ordinals 2 through 7, compare three saved baseline points and capture four new signed finite-difference calls (two variables, coordinate 0, step 1e-5, ± signs). A useful state-isolation extension is parent0 at points 0–2 before and after the six-parent sequence, adding six calls (48 total); the repeated A outputs should exactly match. This does not replace each parent's separate fixed-native comparison.

A batch needs a new exact roster/parent/source contract. One success/ACK must require all calls, all six per-parent baseline comparisons, all FD checks and any chosen A controls. Partial/error artifacts remain inspectable but cannot imply success or authorize all fixed stages. The V2 per-parent ordinal2 review object is insufficient for this wider batch.

Compile once may avoid five repeated graph compilations, but all 42 (or 48) calls and durable receipts remain. No actual timing, memory or disk scaling has been measured. A new independently reviewed and freshly resource-assessed version is needed; no batch code or model run occurred here.
