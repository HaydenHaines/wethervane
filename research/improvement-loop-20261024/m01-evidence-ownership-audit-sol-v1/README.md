# M01 evidence ownership: current inventory and admission contract

The available files establish source identity mappings and event exclusions,
but do **not** establish respondent disjointness, fully independent calibration,
or historical availability. `inventory.json` records the actual field schemas,
counts and source hashes. `inventory.py` regenerates the read-only audit.
No selected full M01 Problem is asserted by collecting these pilot artifacts.

| Evidence | Available identity/provenance | What remains unresolved |
|---|---|---|
| Current TX2026 polls | Six original `sample_id`, `release_id`, `overlap_group_id`, `selected_sample_owner`; model-ID map; field dates, publication date/basis, source URLs and pinned preserved-rendering files; universe, candidate pair, raw percentages, nominal N, mode and provider | All six lack exact publication instants. Each says cross-survey respondent overlap is unknown. Effective N is missing in all six. Sample IDs identify releases, not people. Source evidence is preserved rendered text, not universally original document bytes. |
| Calibration | 1,483 rows, 105 race keys; `source_row_number`, `source_id`, `source_path`, candidate joins, pollster, sample/overlap fields, residual outcome `actual_dem_share`, available-date fields; calibration plan/source pins and explicit TX event exclusions | All 1,483 rows have `unknown_no_author_sample_ID_or_field_start`; publication and historical-version verification are false. `available_date` is a field-end proxy. Source IDs and invented `unknown:...` sample IDs are not deduplication proof. |
| CES2020 TX | 4,294 unique `CES:2020:case_id` records in 33 time blocks; raw `year,case_id` join, source receipts, demographic and observed-answer masks; weights retained diagnostic-only | Internal one-record ownership is checkable; cross-provider respondent identity is not. A case ID is not a global panel identifier. Missing political items remain marginalized. |
| Official results | TX2020/2024 canonical race keys and candidate counts; FEC sheet/row coordinates or Clerk PDF page/line, source IDs; model contest/candidate mapping | Original-result byte and publication-vintage chain is not fully authenticated by this audit. These are earlier training counts for TX2026, explicitly not 2020/2024 evaluation forecasts. |
| Fundamentals/other inputs | Actual modular graph supplies shared states and compiler once-only evidence tokens | No complete M01 selected-Problem manifest and fundamentals label-use ledger is bound here; cannot assert coverage or absence of repeated training labels. |

The recomputed intersection between calibration race keys and current pilot
official race keys is empty. This supports that precise exclusion check.
Calibration and downstream official observations both include cycle 2020;
same-cycle shared shocks remain, and different-state keys do not prove
conditional independence. Survey reports and official results about the same
event are different observations; they should share latent event behavior,
not automatically be declared duplicated official labels. Calibration residuals
using official results and a separate official-count factor can reuse the same
label; this needs a joint law or event exclusion.

## Falsifiable contract for a native successor

1. Bind the exact selected Problem, parent bank, full provider composition,
   complete poll-vector order, candidate/contest/universe mappings and every
   source byte hash. Define the whole vector before splitting it into numerical
   blocks; preserve one calibration component identity where uncertainty is
   shared. Name once-only ownership of sampling/movement/error variance.
2. Build a canonical evidence ledger across modules. Each physical record or
   source-derived statistic has one canonical atom and one numerical factor
   owner. Aliased model observation IDs must map back to that atom. Official
   outcome labels use canonical event/contest keys independent of spellings.
   Inventory which labels trained calibration, fundamentals, representations,
   tuning and downstream count factors. Do not classify survey self-reports as
   official labels merely because they concern the same event.
3. Assess respondent reuse **and** source/shared-error dependence between every
   ledger bundle. Each assessment is `unknown`, evidence-established disjoint,
   evidence-established not applicable, or jointly modeled with a pinned
   conditional adapter and law ID. Missing metadata stays `unknown`. A provider
   name, date gap, event exclusion, or different sample ID alone cannot prove
   person-level disjointness. Bundle rows only when its internal dependence
   inventory is reviewed; bundling cannot hide unresolved overlap.
4. Reuse a known shared sample through one joint observation law; otherwise
   remove duplicate releases or version an explicit overlap-sensitivity model.
   Unknown overlap may motivate sensitivity research, but cannot pass a claim
   of established disjointness. Shared cycle dependence needs an admitted
   conditional model or an explicitly narrower independent-data design.
5. For a historical replay, every training input must have verified availability
   no later than cutoff and a verified historical version. Do not invent a clock
   from a date or promote field-end proxies. For present retrospective research,
   keep approximate availability labeled; it need not block the current
   fixed-state or provenance audit. Never use an evaluation event's labels to
   fit its replay. A/B comparisons require identical admitted information and
   tuning opportunities, with earlier-only calibration for each target cycle.
6. Then execute full-compiler fixed-state checks: one-component incumbent
   reduction, multi-block shared-index fixture, reordered candidate IDs,
   duplicate evidence under aliases, unknown sample IDs, source hash changes,
   target-label reuse, late vintages, variance double addition, and protected
   population parent masses. Only subsequent owner-coordinated matched joint
   refits can assess predictive benefit; no approval barrier is added here.

`validator.py` implements a small supplied-ledger contract, not a source
authenticator or complete native integration. It rejects duplicate canonical
atoms across numerical factors, official-label reuse across factors, evaluation
label leakage, missing dependence assessments, unsupported joint-adapter
claims, missing vector binding, and unsupported historical dates/vintages.
For a genuine joint likelihood, represent shared evidence under its one joint
factor owner rather than recording two independent factors and relying on a
shared-error annotation to excuse duplicate labels. This is a conservative
ledger representation, not a restriction on valid conditional statistical laws.

Its evidence pins are syntax-checked; they do not authenticate file contents or
establish scientific sufficiency. Arbitrary metadata can pass. A human/source
review must verify the referenced evidence, canonicalization, bundle-internal
dependence and completeness. Current missing provenance should yield an
unresolved admission status, never a fabricated passing ledger.

Eleven synthetic unit tests pass, including same atom under different IDs,
unknown overlap, missing pair assessment, target-label and calibration/count
label reuse, publication proxies, late availability and missing conditional
adapters. Separate retrospective mode deliberately makes no strict-as-of
claim. The fixtures are invented; no actual ledger is admitted by these tests.

Overlap with existing work: this audits the current bank/pilot provenance and
extends the earlier native supplied-state parity artifact. It neither repeats
the M01 mathematical implementation nor changes the owner's current
Nevada/national workflows. Owner follow-up v2 remains the written coordination
basis; no new source-dependence decision is inferred from it. No model fit,
sampling, native registration, promotion or owner-tree mutation occurred.
