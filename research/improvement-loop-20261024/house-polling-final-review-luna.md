# Independent review: House polling frontend packet

**Review date:** 2026-10-24  
**Reviewer:** Luna, independent read-only review  
**Scope:** Final historical race-level packets and published score exports. No model fits or source edits were made.

## Pin and packet integrity

The reviewed catalog is `frontend-results-v1/catalog.json`, SHA-256 `9b35f2a63c895cf6b01de9ac3542b2f7abcb68a555aabdc73fedcb5d55683b0c`. The reviewed self-audit is `output-audit-v2/review.json`, SHA-256 `86e537ae80784dd937e4c90dc6655afe3f0de1b6cb737ce3c5bd89011282b1e9`.

I recursively checked 327 file pins reachable from the catalog, audit, and the 32 final full-roster and conditional-share packets: all referenced files were present and all hashes matched. This establishes byte-level binding for the reviewed packet set, not scientific truth or prospective validity.

## Matched F/FP comparison integrity

All 16 baseline-F / combined-FP full-roster pairs (four model configurations across 2018, 2020, 2022, 2024) preserve the same 435 race IDs, group IDs, member IDs and member kinds. Their event/outcome definitions also match: rule family, nominal cutoff, comparison key, support mode, score eligibility, vote-share reference availability, outcome status, group/member outcome, source basis, and unavailable reason. I found no pair mismatch. The 2018 NC09 unavailable outcome remains unavailable in both sides; it is not silently scored as a loss.

I independently recalculated the multiclass categorical Brier score from each FP packet's race probabilities and observed group outcome, using `sum_g (p_g - 1[y=g])^2`, then averaged over observed outcomes. All 16 results agree with the packet metrics and exported cycle-score table to floating-point rounding (under 1e-16). The score counts are 434 in 2018 and 435 in each later cycle. The verification script/output summary is captured in `house-polling-final-review-luna-evidence.json`.

## Score masks and comparison with original F strata

The group-score export has these masks:

| Cycle | all435 groups | polled groups | unpolled groups | noncore groups |
|---|---:|---:|---:|---:|
| 2018 | 434 | 115 | 319 | 63 |
| 2020 | 435 | 82 | 353 | 52 |
| 2022 | 435 | 65 | 370 | 62 |
| 2024 | 435 | 75 | 360 | 60 |

The `all435_groups` mask is the same scored-race mask as the original F public `all_scored` stratum. I recomputed its F scores from the original F result packets; they agree with the original exports. `noncore_groups` corresponds to the old F `newly_included` stratum, likewise with matching race counts and recomputed scores.

The new `competitive_F_probability_0_1_to_0_9` mask is **not** the original public `competitive` mask. The old mask used prior presidential Democratic share from 45% to 55% (see `research/house-full-roster-20261022/score.py`); the new mask uses F's forecast win probability from 0.1 to 0.9. They must not be labeled as a continuous historical series or direct comparison. `polled_groups` and `unpolled_groups` are new poll-availability masks. F-versus-FP comparisons for those masks need F recomputed on the exact same race rows; the old all/core public strata do not provide that matched comparison. `all_original_share_eligible` is the legacy share roster, while `same_polled_races_F_P_FP` is a newly matched polling subset.

## Metric interpretation, especially 2020

The 2020 result is mixed and should be described by metric and mask. For `senate_relative_context_v1`, the all-race group Brier worsens from F `0.0498772` to FP `0.0533612`; the 82 polled-race group Brier also worsens (`0.157304` to `0.196103`), while the 353 unpolled-race Brier improves (`0.024923` to `0.020203`). Across the original 383 share-eligible races, share CRPS worsens (`0.0213182` to `0.0240597`) and share Brier worsens (`0.0256751` to `0.0288408`); 80% coverage moves from `0.9347` to `0.7911`.

On the 82 same polled races, share CRPS improves (`0.0246296` to `0.0174785`), while share Brier worsens (`0.0786115` to `0.0980521`), log loss slightly worsens (`0.30680` to `0.30904`), 80% coverage falls (`0.9512` to `0.8415`), and point-call accuracy falls (`0.9146` to `0.8659`). These are not contradictory: proper scores measure different distribution properties, and the restricted mask differs from the full legacy roster. Across configurations, same-polled share CRPS improves but same-polled share Brier worsens; all-race group Brier is mixed. Do not claim simply that polling “helps” or “hurts” 2020, and do not explain the pattern causally from this retrospective packet.

## Names and identity

The documented exceptions are display-name aliases, not candidate/member identity changes. The seven unique race-party label changes are:

- 2018 WA-03 R: `Beutler, Jamie Herrera` → `Jaime Herrera Beutler`
- 2020 IL-13 D: `Longdrigan, Betsy Dirksen` → `Betsy Dirksen Londrigan`
- 2020 NH-02 D: `McClane Kuster, Anne` → `Ann McLane Kuster`
- 2020 PA-10 D: `DePasquale, Eugenio` → `Eugene A. DePasquale`
- 2020 TX-03 D: `Seikaly, Lulu` → `Helane Seikaly` (exception record ties FEC fullname to Clerk “Lulu Seikaly”)
- 2020 WY-00 D: `Grey Bull, Lynnette` → `Lynnette GreyBull`
- 2022 WY-00 R: `Hageman, Harriett` → `Harriet Hageman`

IDs, groups, maps, and outcomes are unchanged. For three exceptions where the alias candidate won (2018 WA-03, 2020 NH-02, 2022 WY-00), the observed-result winner string retains the baseline FEC spelling while the forecast candidate label uses the updated display spelling. This is not an identity mismatch, but the UI should avoid presenting the two strings as different people; align display-only outcome names or identify them as the same member.

## Readiness boundary

The catalog is explicitly marked `reviewed_experimental_research`; it declares `current2026: false`, `public_chamber_control_qualified: false`, and `website_activation_requested: false`. The output audit's passed status describes its own output checks, and its scope says this is a self-review with separate formulas and actual-output checks, not independent scientific validation or a prospective test. The packets use final historical rosters/returns and a final poll archive with an availability proxy rather than exact contemporaneous publication vintages. The model is aggregate: it does not model turnout or vote counts, allocate within third-party groups, or simulate ranked-ballot/runoff outcomes.

Publication should therefore be limited to an explicitly retrospective, actual-race historical comparison with the score masks and limitations stated. The packet does not qualify a 2026 forecast, chamber-control probability, or website activation. The manifest and hashes establish reproducibility of these outputs, not model qualification.

## Reviewer conclusion

No material packet-pin or matched-pair integrity issue was found. The categorical Brier exports reproduce from packet probabilities. Material presentation risks are mask-name equivalence (especially “competitive”), the mixed 2020 score changes, and winner display aliases. Address those in the frontend and accompanying text; keep the claim historical and descriptive.
