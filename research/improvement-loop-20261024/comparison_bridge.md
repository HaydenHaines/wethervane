# Reviewed M02 comparison bridge

`comparison_bridge.py` connects the existing momentum score/review dossier to the
experiment queue and journal. It supports one native comparison family and the
explicit **retain FS5** recommendation. Promotion, defer, other decision schemas,
different scientific scopes, activation and publication are rejected or absent.
It does not fit a model, score a prediction bank, read original outcome columns,
restart an owner process or establish a current/full-count forecast.

## Pinned request and metadata boundary

The request has exactly `schema`, `source_root`, `cycle_id`, `cycle_binding` and
`sources`. Its schema is `ReviewedMomentumComparisonRequestV1`.
`cycle_binding` is the complete existing journal binding: incumbent configuration,
scope, target layer and hypothesis. The supported scope remains H14 conditional
aggregate D/R, A2018/2022 and B2020/2024, with the original common-race support,
`aggregate_comparator_only=true` and `strict_asof_qualified=false`.

`sources` contains these seven named SHA pins:

| Role | Native record |
| --- | --- |
| `admission` | `MomentumForecastScoringAdmissionV1` |
| `scoring_terminal` | `MomentumScoringTerminalV1` |
| `scoring_receipt` | `MomentumScoringOutputReceiptV1` |
| `scores` | `MomentumMatchedForecastScoresV1` |
| `independent_review` | `MomentumActualForecastResultsIndependentReviewSolV1` |
| `root_acceptance` | `MomentumActualForecastResultsRootAcceptanceV1` |
| `root_decision` | `MomentumForecastDecisionV1` |

Validation authenticates the named JSON bytes and follows typed links for the
frozen selection/masks, three fitted arms over four cycles, forecast terminals,
seals, numerical summaries, execution receipts/launches, runtime and source-review
metadata, and scoring launch/label-access metadata. The original label source is
compared by pin and never opened. Manifest JSON is authenticated; source code and
raw tensors referenced by those manifests are not revalidated by this bridge.

The five score arms are FS5, FS6, LP5, analytic PO5 and archived F0. Only the first
three require fitted forecast terminals. Every cycle/tier cell retains its value,
status and **conditional numerical precision** flag, together with its full native
cell including conditional MCSE/uncertainty fields. No uncertainty is recalculated.
An empty cell has a null value and zero count; a missing constituent cycle remains
unavailable in its equal-cycle tier. That flag does not qualify a
scientific model. Core/bulk/bank/tail gate records, the separate LP5/2020 interval
display limitation, archived F0 unresolved/dispersion-only cells and empty A
near-even summaries stay visible. The independent review's 19 point regressions,
18 numerically resolved regressions, original check count and limitations are
preserved as prior review evidence; those arithmetic checks are not rerun.

The deterministic output is `ReviewedMomentumComparisonReceiptV1`, with
`status=verified_scoped_retain`, the original request pin, all checked metadata
pins, normalized cells and explicit scope limitations. It distinguishes:

- Scorer completion and recorded numeric-label access.
- Prior forecast numerical gates and metric-specific precision.
- Prior independent scientific review and separate root acceptance/decision.
- Full-count qualification, model promotion and publication, all false.

The report lists the raw outcome/influence-array links that were not reopened.
`raw_bank_revalidated=false` is deliberate. Byte authentication of metadata and
preservation of a prior review are the evidence provided by this command.

```sh
python3 -B research/improvement-loop-20261024/comparison_bridge.py validate --request REQUEST.json --sha256 REQUEST_SHA --output FRESH_VALIDATION_OUTPUT
```

The request SHA hashes its exact file bytes. This differs from the queue plan's
canonical-JSON digest convention.

## Journal reconciliation

Reconciliation independently receives the original request pin and the validator
receipt pin. It reconstructs the normalized receipt from original native sources
and requires exact equality; a caller-authored report is insufficient.

```sh
python3 -B research/improvement-loop-20261024/comparison_bridge.py reconcile --comparison-receipt VALIDATION/receipt.json --sha256 COMPARISON_SHA --request REQUEST.json --request-sha256 REQUEST_SHA --journal-root JOURNAL --cycle CYCLE --expected-revision REV --command-id STABLE_ID --output FRESH_RECONCILIATION_OUTPUT
```

A missing database/cycle fails without creating a journal. The initial snapshot
uses SQLite read-only mode and checks the full expected cycle binding.

For the actual M02 cycle, already `decided` at revision 7, the only accepted no-op
has `outcome=success`, `recommendation=retain` and exactly the original evaluation
(root decision) and review references. Their registered object sizes and bytes
are authenticated. `already_recorded` opens no `Ledger` writer, copies no journal
artifact, records no command and adds no checkpoint. The native decision and all
existing journal rows remain intact.

For a future supported cycle in `evaluating` or `reviewing`, all compact metadata,
the original request and reconstructed report are staged and reverified before
opening a journal writer. One compare-and-swap checkpoint records `decided/retain`
with the original decision/review references. This records the reviewed scoped
recommendation; it cannot update an active-model pointer. The deterministic note
binds the normalized comparison/request digests. No new scientific cycle is
created by reconciliation.

Use the same journal command ID, expected revision and pinned comparison/request
when recovering a caller crash after journal commit but before output publication.
The **executor retry must use a new attempt and fresh output directory**, preserving
the partial earlier output. Reconciliation authenticates the recorded command,
checkpoint and artifact objects and returns `recorded_idempotent` through a
read-only snapshot. It does not append another checkpoint. Changed payloads,
advanced journal state, corrupt objects or reused/foreign output directories fail.
In-place execution/output resumption is not supported.

## Queue recipe and review boundary

The concrete capacity-one recipe under `comparison-bridge-sol-v1` runs:

1. Metadata validator with the exact request/source pins.
2. Journal reconciliation with the validator's exact receipt path and dynamically
   bound SHA, plus an independently pinned original request and expected revision.

The actual recipe expects `already_recorded` and `journal_read_only=true`; it is
an integration check against the existing completed M02 result, not a new fit,
comparison, decision or journal mutation. Root owns its explicit queue invocation.

For future scientific work, the existing native scorer remains an earlier,
separately authorized queue phase. Independent review is an external evidence
boundary: admit a validation/journal plan only after the actual score outputs,
independent review and root decision have their own pinned records. The bridge
does not invent review approval inside a dependency graph.

Tests combine actual metadata-only checks with temporary software-journal fixtures
for writes, stale revisions, cross-bindings, unsupported decisions, corrupt objects
and post-commit recovery. The actual journal check forbids a `Ledger` constructor
and compares its database bytes and complete M02 history before/after.

```sh
python3 -B -m unittest discover -s research/improvement-loop-20261024 -p test_comparison_bridge.py -v
```
