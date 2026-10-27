# Independent review: observation-ownership-checker-sol-v2

## Verdict

V2 resolves the V1 chronology gap. It distinguishes an exact pinned version's availability from a product/series first-release date and from the target election date. It requires the actual observed collection interval to end no later than exact-version availability, which itself must be no later than the composition cutoff. Future intention targets remain separate from the collection interval. I found no new blocking inconsistency in the stated claim-only contract.

The central limitation remains: all times and source/vintage claims are supplied metadata. This is necessary chronology consistency, not authenticated historical availability or scientific admission.

## Frozen inputs

- checker.py: 65c28c3dbf0a5d6e1f1cd7dc60e8cccc47dce2b336a660ddd73769aac3bde8c2
- test_checker.py: be9adc877f4c94297fc4830006c2f1206d53ab8a5167cb961b1adde3e609f7df
- evidence.json: 7006bb1794aa3963334ce2b60059abc80df9933162fa42fc4a75c65d7fe4c8c1

## Reproduction

Copied checker.py and test_checker.py into a fresh temporary directory and ran:

    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest test_checker -v

All 24 tests passed. The output is retained in tests.log. No private IDs, model jobs, raw arrays, or owner-tree changes were involved.

## Reviewed chronology cases

The fixtures exercise the specific distinctions requested:

- Earlier product first release with a later rolling exact snapshot containing later collected observations: accepted when collection end <= exact-version availability <= cutoff.
- Earlier exact snapshot that predates its claimed observations: rejected with observation_after_source_version_available.
- Revised exact snapshot after cutoff: rejected even where the product first release predates cutoff.
- Equal observation/version endpoints and timezone-equivalent instants: accepted; a one-second earlier version time rejects.
- Missing exact-version availability or observed collection interval: rejected explicitly. Unknown product first-release date is allowed.
- First-release timestamp after exact-version availability: rejected.
- Future election intention target with pre-election collection interval: accepted; moving the collection interval into the future rejects. This correctly prevents target time from masquerading as when an observed answer was collected.

## Remaining limits

1. The checker does not authenticate exact-version bytes, release timestamps, source lineage, or first-release claims. A caller can reauthor a coherent manifest and expected digest. The correct result remains explicitly unauthenticated.
2. The chronology rule only applies when period_start/period_end truly mean the measurement/data-collection interval. Forecast horizon, target date, publication date, and retrieval date need separate typed fields/contracts; V2 correctly makes no automatic exception for a future target.
3. This does not resolve respondent overlap, shared source error, factor normalization, actual graph binding, or source-derived atom identity. Sampling attestations, joint-law hashes, and vector/source pins remain caller claims. A2018's aggregate helper still does not reconstruct row-level ownership or prove survey/poll respondent independence.

Under the documented timestamp meanings, the temporal ordering rule is internally consistent and the old counterexample now fails. No claim of real-source verification follows from this review.

