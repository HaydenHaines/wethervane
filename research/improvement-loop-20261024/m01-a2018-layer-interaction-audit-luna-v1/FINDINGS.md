# A2018 layer complementarity: source-bounded verification

## Why this check

The goal inventories eight roles: population/geography, community, fundamentals, national environment, measurement, participation, candidate choice, and counts/election rules (`research/improvement-loop-20261024/GOAL.md`). Existing evidence and the prior source review do not identify an empirically weakest layer. The highest-value low-cost verification available without touching posterior rows is a source-bound check of the *mechanism connecting* candidate composition and participation composition to the same ordinary paired-poll factor, while confirming which observation blocks are actually present for A2018.

## Verified aggregate/source facts

`check.py` hashes the exact A2018 unadmitted plan, V2 aggregate ownership ledger, and selected `117-components.py` snapshot before reading them. It confirms the selected eight roles, `candidate_iid_profile_choice_v1` rather than the role-hierarchy alternative, eight protected parent masses of 0.125, one 274-row poll covariance block (20 AZ rows), and zero official-count and participation-report observations in the prepared problem. It also checks the ledger's source-closure flags for no open-person/count/report/ranked/runoff observations, an eventG-free survey loop, ordinary-poll formula, and scalar contest response. The plan remains unqualified, incomplete, and unauthorized for publication. These are exact pinned-plan/aggregate claims, not independent source-vintage validation.

The selected source expressions establish this conditional mechanism: cell intensity is exposure × community share × turnout propensity; candidate softmax probabilities feed category rates after one scalar response probability; eventG multiplies the base category rates; ordinary poll means use log(rate[first])−log(rate[second]). Thus eventG and scalar response cancel from ordinary pair log-ratios, while profile weights and profile-specific candidate softmax allocations can remain relevant. The A2018 closure says no counts/reports/ranked poll channels and survey loop eventG-free; it does not prove a compiled graph or historical sample independence.

## Bounded invented sensitivity result

The deterministic fixture uses two profiles with different D/R utility gaps and an invented O-candidate utility shift from −1 to +1. It evaluates the exact source equation shape, not actual poll rows or posterior values. With equal relative profile intensities, the poll logit shift is `+0.0264773`; with a 9:1 relative profile-intensity shift, it is `+0.0122397`. The difference-in-differences is `−0.0142376`. Changing scalar response and eventG together leaves the poll logit unchanged to floating-point tolerance; when D/R utility contrast is the same in both profiles, the O utility change cancels from the aggregated D/R log-ratio.

This is a counterexample to assuming the layers are algebraically orthogonal: candidate softmax denominators and participation/composition cell weights can interact in an ordinary pair-poll prediction. It does not identify the size or sign of this interaction in A2018, demonstrate that A2018 posterior states occupy the fixture, or show predictive value. Utility values and relative weights are invented, and no data records or arrays were read.

## Evidence boundaries and next useful step

The existing source ledger verifies prepared mappings/vector digests and factor closure, but source lineage, question/sample identity, first-publication vintage, cross-source independence, positive-definiteness/conditioning, and compiled graph execution remain unverified or blocked. The `eventG` claim is conditional prior algebra, not evidence its predictive role is negligible. The 2018 plan is unqualified, so no selection or weakest-layer claim follows.

The next scientific test should be a small, source-authorized, aggregate-only sensitivity extraction from a **qualified** jointly fitted baseline: for an identical set of common physical parameter states, report selected-poll logit contributions while changing only (a) a prespecified candidate-utility contrast and (b) a prespecified profile-intensity contrast. Keep each protected parent at 0.125; do not reweight parents by evidence. Before that test, the owner must provide a validated, non-exported parameter summary or a clean read-only interface plus proof the chosen baseline is qualified. Do not use the failed/unqualified A2018 banks as if they were posterior-valid. If such a qualified interface is not presently available, the source-equation fixture here is the bounded result and the next empirical test is blocked on that prerequisite.

## Reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B check.py > result.json
```

The script reads three small pinned files only. It does not import model code, read context/problem arrays, inspect raw survey or poll rows, or run an owner job. `result.json` contains aggregate counts and invented sensitivity outputs only.
