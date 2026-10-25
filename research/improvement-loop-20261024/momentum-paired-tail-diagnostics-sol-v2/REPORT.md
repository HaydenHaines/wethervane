# Paired tail diagnostics v2 — recoverable input-consistency correction

2026-10-25. **Eleven supplied-only tests pass; independent review pending.** V2 preserves the paired smaller-side arithmetic of v1 and adds Root-selected validation of recoverably contradictory paired inputs. No statistical threshold, probability law, source, data, mask, random stream or attempt budget changes. Actual historical target data, numeric labels, empirical scores, Gibbs chains, optimizers and fits remain unused.

V1 manifest `522b89bcb9629805735109f86da781451d77463ff978ddfa082df48677426603` remains unchanged. Its independent review receipt `614f4fecde2a694e8b299d107c6792677d773b148c9e659ce6cd577011650564` requires this correction. The original positive arithmetic tests and initial contradictory-input fixture/failure remain preserved there. This v2 binds unchanged Root design `45707d4a933ded2f7d4bffcbaab0fb6fb9a4ab7b1d62ad4494a27422b7d75678` and clarification `0d6dd97153687df2583c0a086e92e8a42c22eb0c850ed0a0133acdcb87c14024`, plus Root's subsequent selected recoverable-input guard.

## Corrected input boundary

An absolute complementary mass tolerance alone can accept a material relative contradiction in a tiny side. In the exact Root z9..10 fixture, adding.1 to log-lose leaves total normalization drift far below1e-12, but the original normal-representable negative log-win still stably determines the expected small-side log via `log(-expm1(log_win))`. The two supplied tails are recoverably inconsistent.

V2 checks the **raw supplied logs before any shared normalization correction**. For each component, if the dominant log is negative with magnitude at least float64's smallest positive normal value, it reconstructs the small log using stable log1mexp and requires absolute log disagreement at most1e-12. Otherwise it explicitly records relative cross-validation unavailable because the dominant log is subnormal or zero. The existing absolute complementary normalization check still applies in every case. This new check is an arithmetic consistency boundary, not a looser/tighter electoral MCSE gate.

Order matters: subtracting a common logaddexp normalizer first can adjust the tiny negative dominant log to fit the altered rare side and erase the very contradiction being checked. The negative regression keeps the raw dominant log unchanged, perturbs the rare side by+.1, and rejects in both win/lose orientations.

For subnormal/zero dominant logs, V2 does not invent relative precision that their representation cannot provide. It retains the supplied informative smaller side as authoritative and reports recoverable/unavailable component counts, maximum checked log discrepancy and the reason. Acceptance of a skipped cross-check is not proof of the rare input's source law; caller provenance and validated Gaussian/empirical construction still matter.

## Unchanged paired method and API

`probability_coordinates` supplies immutable centered scaled smaller-side probabilities, log scales, signs for exact complementary deviations, pooled/chain stable log probabilities and input-consistency scope. `paired_tail_diagnostics` provides JSON-safe `win`/`lose` summaries with old-compatible measured/explicit-structural statuses and original one-sided summaries retained separately.

The informative smaller coordinate supplies identical absolute MCSE to both sides; larger-side relative and chain log behavior use stable complements. Numeric underflowed MCSE remains missing with finite log measurements/gates, never zero-error certification. Unexpected constants/all-minus-infinity remain unresolved. Raw indicator handling,50/100 batch passes, mean-ESS naming and all selected thresholds remain untouched. No retry, label, score or promotion policy is implemented.

Score consumers must use the shared centered deviation/sign/log scale for Brier/NLL influences. Normalizing near-one probabilities and subtracting them again would reintroduce cancellation; the supplied coordinate API avoids that operation.

## Evidence

`attempts/pytest-v1.txt`: **11 passed in0.72s**. Retained tests cover exact Root z9..10 direct-survival MCSE and original-status preservation; pooled/chain complement logs; side/race symmetry; stable endpoint arithmetic; relative failure separated from bulk absolute gates; z50..51 underflow status; constants/raw zeros; malformed inputs; and analytic Brier/NLL deviation identities.

Added regressions reject the raw+.1 rare-log contradiction in both orientations, and check concrete z9/10/37.5(normal-dominant) versus z38/40/50(subnormal/zero-dominant) scope. Underflowed z50..51 rows still have measured absolute log MCSE/gates with ordinary MCSE missing and cross-validation availability explicitly false. The independent reviewer separately examined a larger supplied native log_ndtr grid and found the1e-12 recoverable bound compatible with normal-dominant arithmetic; that is review evidence, not additional model tuning or chain execution.

Ruff final check/format pass. The initial Ruff log records one long prose string, corrected only by equivalent string formatting; no numerical test failed in v2. Runtime remains NumPy2.4.4/SciPy1.17.1/pytest9.0.2, one BLAS/OMP thread, bytecode disabled. Source snapshots and upstream manifests are bound in SOURCES/artifact manifest. Independent exact-v2 acceptance and runner/scoring source binding are required before actual execution.
