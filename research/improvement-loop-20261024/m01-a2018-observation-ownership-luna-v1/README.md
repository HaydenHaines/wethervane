# A2018 observation-ownership ledger

This packet inventories the frozen A2018 context's **likelihood factors and their observation identifiers** against its selected provider plan and `117-components.py` source snapshot. It reads no posterior/predictive arrays and runs no model. The builder parses the pinned prepared context/problem to emit metadata only: factor/block IDs, opaque record tokens, event/contest IDs, source roles, covariance-block membership and aggregate counts. It does not copy survey masks, poll logits/covariance entries, counts, or prediction values.

## Frozen inventory

- 54 events and 37 contests, cutoff recorded as `2018-10-23T00:00:00Z`.
- 65 survey likelihood blocks and 7,968 survey record tokens: CES2014 has 31 contest-specific blocks / 3,662 rows for the 2014 Texas event; CES2016 has 34 no-contest blocks / 4,306 rows for the 2016 Texas event.
- One correlated poll MvNormal block with 274 paired-candidate logit rows and a 274×274 covariance. The Arizona 2018 regular contest has 20 poll rows, identified in the machine ledger.
- Zero official count vectors and zero participation reports in this prepared A2018 problem.
- The A2018 plan's selected providers are the IID candidate choice, profile turnout/event-Gamma participation, single-mark category counts, and joint survey/report/poll measurement. Its outer eight parent masses remain 0.125. Plan flags say `national_complete=false`, `publication_authorized=false`, and `forecast_qualified=false`.

`factor-membership.csv` has one line per source-scoped survey record membership or poll-row membership (8,242 data rows). It is an ownership map, **not** 8,242 independent likelihood factors: the 7,968 survey rows group into 65 Potentials and all 274 poll rows group into one covariance likelihood. Opaque survey `record_id` uniqueness is checked only within source; the ledger does not infer unique people, independent respondents, or absence of cross-source overlap.

## Units, absent factors, and cancellations

- The survey row schema carries an opaque record id and community/turnout/answer masks. It is a survey-record unit; unique respondent status, sampling/design weights, and cross-source overlap are not established by this ledger. Survey source roles are CES2014 and CES2016.
- Each poll row has an observation token and paired candidate IDs and enters one shared 274-row covariance block. Pollster, field dates, sample ID, population, mode, and sample size are not present in the prepared problem schema. The row token is not evidence of an independent poll sample.
- Official ballot-mark count vectors and participation reports have no observation IDs here because both input lists are empty. A separate Arizona candidate-mark score receipt exists outside this problem's likelihood; it is not a voter-participation observation.
- In the selected ordinary paired-poll branch, the mean is `log(category_rate[first]) - log(category_rate[second])`. Since `category_rate` includes a common event multiplier and the same contest-level scalar candidate response for each candidate, both cancel algebraically. Cell-profile weights remain in the aggregate rates, so turnout composition can still affect a poll ratio indirectly. This does not make a poll ratio a measure of total participation. The prepared A2018 poll rows have no ranked-channel assignments.
- With no official count vectors, no observed-event Gamma-coordinate factor is constructed; all 54 event scales use the `Gamma(kappa,kappa)` unobserved-event prior. The target AZ event has no direct survey, count, or report factor. Shared survey parameters and profile-weighted polls can still inform model components indirectly; the ledger does not claim complete lack of information.
- The separate score review is hashed as an external evaluation artifact only. It is not inserted into the training likelihood by this packet.

`ledger.json` records all 54 event identifiers, 65 survey factor/block identifiers, all 274 poll observation identifiers, all source identifiers/roles, parent identities/masses, the AZ target identifiers, factor families, important prior families, and explicit unknowns. `source-pins.json` binds plan, context, prepared problem, component snapshot and external score receipt. `build_ledger.py` checks those exact hashes and context/plan/problem identities before producing the ledger. `test_ledger.py` exercises those guards and the factor counts/source membership.

## Reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B build_ledger.py
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest test_ledger -v
```

Six bounded tests pass. The source registry's `available_utc` values are recorded as supplied; this ledger does not independently verify original source-release dates. Nothing here ranks layers, claims source independence, establishes a forecast score, changes outer parent weights, or qualifies publication.
