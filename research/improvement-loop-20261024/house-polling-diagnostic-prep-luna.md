# Read-only House F/P/FP diagnostic preparation

Snapshot reviewed 2026-10-24 from `/home/hayden/projects/ballotmancy-house-benchmark/research/house-polling-benchmark-20261024`. This is preparation for the owner’s review packet, not an acceptance review. Primary cell score tables exist; the benchmark README still says intake is in progress and lists delivery/review as remaining, while sensitivity runs and repair logs were changing during this read. Do not treat these intermediate tables as final or cite them as public results.

## What the arms test

`protocol-v1.json` calls the arms `F`, `P`, `FP`, but per-cell outputs label the combined `FP` arm `primary` (see `run.py`, `selection.py`, `scores.py`). Interpret that mapping explicitly.

- **F** is the exact saved predecessor fundamentals forecast. It is a locked reference, not a newly fitted fundamentals-only treatment.
- **P** is a common flat-logit Gaussian poll-only posterior, and only predicts poll-covered districts (`run.py::polls_only`, `scores.py::share_cell`). It is not the same full-roster categorical winner model as F/FP; group-score P tables are absent by design. It contextualizes how much the poll data alone says on the polled mask, but P-vs-F is not the incremental poll contribution conditional on fundamentals.
- **FP / `primary`** conditions the existing F mixture on the aggregate D/R poll likelihood. The same F prior predictions, likelihood implementation, and earlier-only plug-in error scales are used; it does not refit the fundamentals component or error scales. The protocol maps the updated distribution to all 435 D/R/O groups using the frozen winner law, loading 1, and common keyed categorical draws. Polls directly inform only eligible core D/R shares; noncore groups have no direct poll likelihood, though the shared national factor can affect them.

Thus matched F-to-FP score deltas on the same polled races are the cleanest available estimate of *incremental value of this specified historical aggregate poll update conditional on this locked F model, source selection, cutoff proxy, and error calibration*. They do not identify whether polls or fundamentals are generally superior, establish causal complementarity, validate all the separate polling/fundamental pieces, or qualify the broader modular/count model. The four election cycles—not hundreds of within-cycle districts—are the main scope for transport. A-vs-B equal-cycle summaries and prespecified per-cycle guards should be shown alongside any pooled presentation.

## Intermediate 2020 illustration (not an explanation or ranking)

For `senate_relative_context_v1`, the current share scores on the same 82 polled districts show F→FP (`primary`) CRPS 0.02463→0.01748 and margin MAE 0.06647→0.04849, but Brier 0.07861→0.09805 and 80% interval coverage 0.9512→0.8415 (with narrower width 0.1475→0.0912). Log loss is nearly unchanged/slightly worse, 0.30680→0.30904. Current group scores on 82 polled districts show categorical Brier 0.15730→0.19610, binary D Brier 0.07853→0.09781, log loss 0.30758→0.30994, and accuracy 0.91463→0.86585. On the 353 unpolled labeled groups, categorical Brier moves 0.02492→0.02020. This descriptive mix demonstrates why “polls help/hurt 2020” cannot be one scalar claim: share-distribution distance improved while event-probability calibration and group scores worsened on polled districts, and unpolled group scores moved differently. It does not explain why. The listed per-cell CSVs are interim and their hashes below are snapshot pins, not a final immutable analysis receipt.

## Reviewer checks needed before interpreting the 2020 pattern

1. Confirm each F/FP delta uses identical race IDs, label masks, outcome panel, and score definitions; check race-level paired deltas, not just separate means. Confirm sample counts distinguish poll-covered share-eligible districts, labeled group outcomes, and the full 435 forecast roster (including any unavailable truth label).
2. Verify that `primary` maps to the largest numerically qualified completed attempt without selecting by score; inspect selected receipts, sampler diagnostics, seeds, and MCSE. Verify F predictions really are the exact predecessor and group predictions share the frozen categorical law and keyed draws. Do not mistake shared random numbers for inferential independence.
3. Report all preregistered materiality guards and score families together: CRPS/proper share scores, win Brier/log loss, interval score and coverage/width; group categorical and D Brier/log loss. Show paired per-race losses, worst losses, calibration, competitive-F subset, unpolled/noncore subsets, and A/B equal-cycle aggregates. A 2020 regression triggers investigation/guards; it is not a reason to select a new metric or to discard the challenger.
4. Check `release_lag3/7`, both error-scale variants, and single-district-only sensitivity against primary, including which 2020 races/rows change and whether scale/source vintage or shared-panel dependence plausibly accounts for movement. These are still source/cutoff proxies; archive availability is not an original publisher-publication ledger. No sensitivity establishes prospective validity.
5. Examine paired 2020 districts descriptively before proposing mechanisms: poll residual sign/magnitude versus F baseline and final truth; poll count, field dates and latest-poll selection; pollster composition, repeated/shared samples, LV/RV and candidate/ballot roster, district competitiveness; posterior shift/width; and whether errors concentrate in high-leverage common national-factor updates versus direct race observations. These diagnose candidate mechanisms but are not causal attribution.
6. Keep alternatives as research questions, not explanations: differential turnout/sample composition and nonresponse by district; correlated pollster or common national error; stale/late movement and archive availability; missing district/candidate effects (incumbency, candidate resources/quality); and third-party/O/coalition ballot mass or identity/mapping mismatch. The existing two-candidate conditional D/R share likelihood and frozen group-winner law do not make separate turnout, full vote-share, candidate-level or exact ballot-transfer predictions. Test one component at a time against the same archived data/cutoff with joint refitting only for the declared affected parameters and retain material-regression guards. Never infer these causes from the cycle-average score pattern alone.

## Artifact pins for this read

Protocol and source-interface hashes (SHA-256):

- `protocol-v1.json`: `c6bfb2f5b3c51ddc9d393241d0152725efbec38651a9768a64066d4392ad7006`
- `README.md`: `94c28e78e668cf5a24e48e99acf5b078b16b449cdf572ddb55741b561a1d6372`
- `run.py`: `8f6cefd1a830bc8e105e471f9152223dce5b3dc6876c7433b82f9ef269bd12cb`
- `selection.py`: `30e0ee4dc11c767aefe83ee29d29e718c16ee6ec863f6de1c77ba3300046bff1`
- `scores.py`: `9cd6817d4cd867cac7ef9587cb9749905b0fe370a84f4573cca033424f4576bc`
- `complete_poll.py`: `1f3e9ed8a94debd56e35a0a6fcf331046d12bc5fdf4619ead826f3cd9f5c701a`

Current 2020 relative-context score snapshot hashes (SHA-256):

- Share F `summary.csv`: `a786b744ba502913cd5a700e03236564403f8c2cf069cf5ec1fe14b5d56267`
- Share P `summary.csv`: `86abd5da03184aa4b33bfbcd72dc95e5a1154b1640d6c1851fd024d469d44ff6`
- Share FP/`primary` `summary.csv`: `ed76895f22e3f3d96bdc85cd450fb6450e03304245a852dc84269741b16579fc`
- Group F `summary.csv`: `1d49b6c135f5c9334d75dbe8a33afd8583ed6d8056f75ebcb06952cdbb1fe5bd`
- Group FP/`primary` `summary.csv`: `53414760ad9b5ec30d08eda8baac1b8bf02568d84b8a66cc01db8a0bfa721408`

The receipts next to these summaries also hash outputs and inputs. Owner sensitivity/review work can replace these score tables; recompute hashes from the eventual packet before making any quantitative claim.
