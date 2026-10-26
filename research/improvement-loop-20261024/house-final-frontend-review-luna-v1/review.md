# Ballotmancy House current-release frontend review

Reviewer: Luna  
Date: 2026-10-26  
Scope: read-only review of the opt-in Playwright release spec, live House projection/rendering, native release packet shape, and pending edits to six historical browser specs. No source edits, typecheck, or browser runs were made.

## Overall assessment

The new opt-in test is bound to the staged website publication, not to route mocks. It reads `data/house-live/publication.json`, checks path and SHA-256 pins before parsing artifacts, checks the reviewed decision and all 16 catalog/result bindings, and requires 435 ordered races and joint-draw identities. For each of the 16 API choices it compares the selected identity/configuration, ordered race IDs, district keys, poll counts, group outcome IDs/labels/parties/probabilities, joint-draw metadata, chamber control chances and expected seats, limitations, and selected poll-question IDs. Its browser assertions then compare the map and ribbon colors against all 435 native race probabilities for every model, and compare control probabilities, default expected seat values, selected polling details, and saved controls against native data.

The release-v6 native packet has the fields and identities assumed by the spec: 16 entries; `house-current-polling` default; 435 races per run; native outcomes use D/R/O candidate groups; and the fundamentals/standard-polling/standard-uncertainty result has genuinely mixed Other groups. For example, AK-00 combines `Nonpartisan` and `Libertarian`; AZ-01 combines `Independent`, `None`, and `Libertarian`. The test predicate checks distinct nonempty source party labels, then the browser check verifies every such member name and label. When no such group exists it records explicit conditional coverage instead of silently passing.

## Review findings

1. **API projection comparison is broad but not exhaustive.** It does not compare the API's candidate roster and candidate labels/ballot status, outcome kind and member-to-group mapping, conditional-pair estimates, rule family/method, source URLs, race qualifications, source cutoffs/status fields, chamber seat quantiles, or detailed polling records. A wrong value in one of those projections could pass the API portion while the hashes and the tested summary fields still match. This is the clearest integration-test gap if “compares all 16 API results” means every published user-facing field. Suggested extension: compare the fields rendered by the selected-race detail and methodology to their native counterparts, especially candidate/member associations and conditional-pair values; compare polling metadata beyond selected IDs if those records are part of the release contract.

2. **Browser result checks are strongest for map, ribbon and chamber headline values.** The detail view checks one poll-positive race's count and group members, but does not check its displayed winner probabilities or any conditional share against the native values. The default control assertion checks expected seats but not the displayed modeled 90% seat range or numerical standard-error text. This is useful additional real-result coverage, not a mismatch found in the renderer.

3. **No scientific wording mismatch found in the live result copy inspected.** It distinguishes conditional pair shares from group win chances, says Other seats remain unassigned, names “Neither party reaches 218” as unresolved rather than a tie, and says accuracy cannot yet be measured. The release limitations also explain the polling subset/availability proxy, incomplete roster, independent-candidate proxies, and RCV/runoff approximations. This matches the science-review framing: current aggregate forecast, 435 regular districts, 2026 empirical skill unobserved, strict-as-of false, and no turnout or vote-count forecast.

## Historical-spec pending edits

The six modified historical browser specs now explicitly select the historical edition where needed, and fixture helpers stub live availability to false and target only `/api/chambers/house` so the new current endpoint cannot capture historical mocks. Fixture IDs ending in `-h14` keep their route classification explicit. The changes appear consistent with the intended edition split and do not widen mocks to the live API. The two fixture suites were not run per instruction; the existing real browser suites were not rerun.

## Execution status

No old suites were rerun and no port was started. Review was read-only aside from creating this report. The current test remains opt-in through `BALLOTMANCY_LIVE_HOUSE_RELEASE=1` and will require the operator-staged website directory before enabled execution.
