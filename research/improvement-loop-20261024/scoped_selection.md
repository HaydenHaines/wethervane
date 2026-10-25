# Scoped count research selection

`scoped_selection.py` connects an independently authorized journal decision to an append-only research selection registry. It authenticates that decision and its declared evidence; it does not derive a winner, recompute scores, solve repeated-benchmark overfitting, establish prospective calibration, or authorize publication. No actual count bootstrap or promotion has been performed: the pinned Texas 2026 native request still requires computation and has no complete count artifact. M02's H14 aggregate `retain` decision and the separate retrospective Texas 2018 GPU run cannot establish this count selection.

## Native boundary and task identity

`prepare(spec_pin, fresh_output)` invokes the fixed `native_selection.inspect_count_candidate` adapter itself. It never accepts serialized readiness flags or an arbitrary adapter callable. The adapter reopens the exact owner request, complete eight-role configuration, and optional complete result through the native application. A pending request produces `blocked_requires_computation`, `candidateRef=null`, and `selectionEstablished=false`; preparation neither creates a registry nor writes the journal. A ready result produces a complete `NativeCountResearchReferenceV1` containing the independently pinned native spec and the full native receipt, including source/configuration/problem/support/law/graph identities and native result, parent, and common-draw evidence. Native numerical readiness remains separate from scientific review.

The CAS task key hashes the native-derived physical prediction task (target, count output contract, election/geography, contests and ballot IDs), the information cutoff, and the exact unit, denominator and estimand from the unique matching authenticated native output contract. This profile requires `integer_counts`, denominator `declared_event_participants_and_contest_category_universes`, and estimand `joint candidate/category counts and shared event participants`; missing, ambiguous or unsupported contract semantics fail closed. Provider/configuration/family/support/law/graph/parent/draw identities and admitted training/observation support are excluded from this key and retained in the complete selected reference. A measurement replacement or poll-off experiment therefore cannot evade the predecessor check by changing input support. The independent review must explicitly address native encoding compatibility and matched inputs or a declared data arm. This adapter supports only `population_cut_v1` / `whole_parent_joint_candidate_counts_v1`.

## Registry operations

Public Python interfaces are:

```python
prepare(native_spec_pin, fresh_output)
initialize(registry_root, domain)  # explicit "research" or "synthetic"
apply(registry_root, preparation_pin, expected_native_spec_pin, decision_pin,
      expected_revision, expected_predecessor, command_id, fresh_output)
resolve_selected(registry_root, exact_task_scope)
history(registry_root, exact_task_scope)
```

The CLI has matching `prepare`, `initialize`, `apply`, `resolve`, and `history` commands; `--help` gives exact arguments. Pins have canonical absolute regular paths, SHA-256 and optional size. Evidence is compact JSON/metadata (maximum 32 MiB per file), not raw posterior tensors or outcome loads.

`initialize` explicitly creates a separate registry with source/domain pins; it does not initialize the journal. `apply` requires an existing registry, a fresh output path, a fresh native reinspection matching the pinned preparation exactly, and a reviewed decision. Research stores reject synthetic native results; synthetic stores are explicitly labeled in every event and resolution. Test inspector doubles cannot be supplied through the public CLI. The synthetic namespace is software-mechanics evidence only.

The typed `ScopedCountSelectionDecisionV1` names the operation, domain, task key, complete candidate-reference hash, expected registry revision and predecessor, exact journal root/cycle/revision/full binding, independent review pin, empirical status, reason, optional restored event, primary scientific layer, changed roles and optional interaction scope. `ScopedCountSelectionReviewV1` independently echoes these identities and disposition, binds native problem/support compatibility, and supplies nonempty pinned review sources. The journal binding preserves its genuine hypothesis and target layer while naming complete native configurations.

Supported operations are:

| Operation | Required authority and resulting status |
| --- | --- |
| `bootstrap` | Absent predecessor; genuine numerically passed complete native result; explicit reviewed baseline authorization and journal `retain`; empirical status remains `unassessed`. |
| `promote` | Exact current predecessor; a different complete candidate; journal `promote`; typed matched comparison and reviewed target transfer; empirical status `reviewed_comparative_support`. This is scoped reviewed support, not general forecast qualification. |
| `retain` | Exact existing complete reference and unchanged empirical status; independent retain decision; appends an auditable event without changing the selected configuration. No first-time implicit bootstrap. |
| `rollback` | Journal `revise` plus explicit reversal review; exact older event in this task's history; native result reopened; new revision restores that immutable full reference and its exact original empirical status and evaluation basis. No evidence upgrade or pointer rewind. |

All operations keep `publicationAuthorized=false`. A run binds the revision and full predecessor it started from. If the selection advances, its later apply fails CAS; the stale candidate and staged evidence remain available for research.

## Matched evidence and transfer

Promotion requires exactly A=2018/2022 and B=2020/2024, with each evaluation scope stating target/output, cycle, office/geography, matched race IDs, cutoff, horizon and information budget. Historical configurations and data differ from the 2026 target; explicit reviewed mapping connects them to the candidate/incumbent target definitions. A `ScopedCountTargetTransferReviewV1` binds both exact target references, target model definitions, every historical definition/score/input-review mapping, target observation-support hashes, and a declared compatibility/transfer disposition. There is no assumption that identical family names prove transfer or that one current-election result proves historical skill.

The comparison graph uses `ScopedCountModelDefinitionV1`, `ScopedCountEvaluationConfigurationV1`, `ScopedCountEvaluationResultV1`, `ScopedCountMatchedEvaluationPlanV1`, `ScopedCountMatchedEvaluationScoresV1`, `ScopedCountNumericalReviewV1`, and `ScopedCountInformationMatchReviewV1`. Configuration/result/plan/score/review links must match exactly, including the native `composition-result-v1` configuration hash and passed numerical status. Target definitions match the full provider selections and pin their sources. These typed normalized documents are supplied under independent scientific review; this module is not a replacement for an owner-native historical validator.

Historical `ScopedCountEvaluationTerminalV1` explicitly binds scope, native configuration SHA, native-record pin, complete status, and an `outputIdentity` pin. The latter is `ScopedCountEvaluationOutputIdentityV1`, binding the same scope/configuration/record to the native prediction artifact kind/ID, nonempty parent IDs and common-draw identity. These compact links are verified as metadata. Historical raw banks are not reopened by the registry; numerical and completion authority remains in the source-bound native numerical review. A terminal file's existence alone is insufficient.

Each frozen plan specifies unique metric IDs and a required primary metric. Every score cell states availability, support count/unit, paired finite candidate/incumbent estimates and difference when available, uncertainty status/standard error/reason, and a reason. Missing, nonnumeric or unavailable primary evidence rejects promotion. Unavailable secondary/subset cells remain explicit, with null estimates and unresolved uncertainty. Full cells and their uncertainty are preserved in the event's evaluation basis. No improvement threshold is invented: the independent decision supplies adoption authority. Every evaluation retains `retrospective_development`, its as-of review status and stated limitations.

## Transactions, retries and reopening

All compact evidence is read, byte-verified, staged durably as content-addressed immutable objects, and native preparation is rechecked before mutation. One registry `BEGIN IMMEDIATE` transaction authenticates the journal's exact decided/successful checkpoint, full cycle binding and exact decision/review reference set, checks both journal object bytes, verifies registry predecessor/revision, and appends the event and payload-bound command together. UPDATE, DELETE and duplicate-key replacement are rejected by append-only triggers. Unused staged objects after a failed CAS are preserved rather than deleted.

The source journal is attached read-only. In supported SQLite rollback-journal modes (`delete`, `truncate`, `persist`), the journal read lock lasts through the registry commit, preventing a competing journal writer from committing a changed head during CAS. WAL is explicitly rejected rather than converted. The actual journal is currently `delete`; this module never changes its mode. This is a local SQLite coordination contract, not a distributed transaction protocol.

A stable command ID binds the exact preparation/spec/decision pins and CAS payload. A crash after commit but before output receipt can be retried as an explicit new executor attempt in a **fresh output directory**, preserving the earlier partial directory. Exact retry authenticates the immutable original event, response hashes/reference, original journal checkpoint and source objects, and returns `already_applied` without a new event. Later journal or registry revisions do not erase this historical acknowledgment. The response separately reports today's registry head. Reusing a command ID with different payload fails.

`resolve_selected` authenticates the stored event and retained objects, loads the selected original spec, and reruns the trusted native adapter; it returns a usable scoped reference only when the full current native receipt still matches. A missing/corrupt native artifact blocks resolution. `history` is an audit view explicitly marked `nativeResultsReopened=false`. Neither is a public default or publication action. Source pins intentionally fail closed after source changes; portable archive/restore, code migration and new owner profiles require separate reviewed integration.

## Verification and actual pending recipe

`python3 -B -m unittest -v test_scoped_selection` uses temporary registries/journals and an explicitly invented inspector double. It covers full configuration selection/reversal, evidence-status preservation, retain, pending/forged/mis-scoped rejection, historical link corruption, missing primary evidence, unavailable subsets, concurrent CAS losers, journal-writer locking, immutable object/history corruption, WAL/domain rejection, selected native reopening, and post-commit idempotency after later heads. The positive production-policy fixture exercises only the admission predicate with an invented capability; it is not actual native or scientific evidence. The real native adapter is independently tested separately.

Root's permitted actual check uses only the existing pinned native metadata and a fresh preparation output, without initializing or applying a registry:

```sh
python3 -B scoped_selection.py prepare \
  --native-spec native-selection-sol-v1/attempt2-spec.json \
  --native-spec-sha256 a72024df6626224cfaba6e0d2d704d7bc4cfc05a2e81b6df2f5aef405247c233 \
  --output scoped-selection-root-v1/actual-pending-preparation-v2
```

Run from this research directory. Expected output is `blocked_requires_computation` with no candidate reference or selection, not a bootstrap. Root owns this expensive actual invocation and the before/after journal/registry proof. No registry or scientific journal mutation is authorized by this recipe. The v1 source/author packet and actual pending preparation are retained as superseded evidence: that version used an incorrect denominator marker. V2 takes denominator, unit and estimand from the native output contract and intentionally yields a different task key. No actual registry was initialized under v1, and no selection migration is implied.
