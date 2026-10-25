# Descriptive Senate polling momentum audit

2026-10-24. Retrospective descriptive evidence only; no forecast fit, causal test, model qualification or activation.

Under the plan-defined primary definition, **12 race cases** had a rising candidate gaining at least one D/R-conditional **share percentage point** between the preceding and final polling weeks, with at least two selected samples and two pollsters in each window. Relative to their final-week poll average, four rising candidates finished more than one share point below it, six finished within ±1 point, and two finished more than one point above it. Counting even tiny misses, eight finished below and four above. The mean oriented miss was −0.97 share points. These frequencies do not establish a systematic momentum advantage or disadvantage.

| Cycle | Eligible ordinary D/R roster races | Sufficient two-window coverage | Rising ≥1 pp | Below −1 pp | Within ±1 pp | Above +1 pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 28 | 7 | 5 | 2 | 2 | 1 |
| 2020 | 28 | 9 | 4 | 1 | 2 | 1 |
| 2022 | 28 | 5 | 1 | 1 | 0 | 0 |
| 2024 | 29 | 8 | 2 | 0 | 2 | 0 |
| Total | 113 | 29 | 12 | 4 | 6 | 2 |

The denominator is selected races with adequate polling and supported truth, not all Senate races, all recent elections, all polls, or two independent candidates per race. One-point errors in conditional share correspond to two points in D-minus-R conditional margin. These are poll-average errors, not errors in the fitted production forecast.

## Definitions and source boundaries

[plan.json](plan.json) records the definitions and is fingerprinted with the script. These artifacts do not independently establish an outcome-blind preregistration. The primary comparison uses field ends 1–7 days before election against ends 8–14 days before election. Selected samples/questions receive equal weight in each window; averages are not nominal-N precision-weighted or quality-rating-weighted. Trend is final minus preceding Democratic share, oriented to whichever principal candidate rose. Exact-zero trends are excluded. Actual performance is oriented observed share minus the final-week mean. The ±1 point band is inclusive.

The only poll source is the existing pinned FiveThirtyEight Senate historical archive at `.redesign-runtime/forecast-20261007/polls/senate_polls_historical.csv`, not a concatenation with Tierney/Stanford copies. The existing `src/forecasting/senate_data.py::senate_polls_as_of` applies the supported eligible race/candidate-ID gate and availability before `src/forecasting/poll_data.py::normalize_538` chooses one question per sample/race, preferring LV, then RV, then adults, then largest lineup, then lowest question ID. Its tracking/hypothetical/ranked-choice exclusions remain unchanged. This audit retains LV/RV only after native question selection; it does not add adult-only questions. Partisan labels remain in the selected rows; the sensitivity excludes explicit labels, without claiming missing labels establish nonpartisanship. Distinct publisher poll IDs can still hide shared panels.

**Availability is an unverified proxy:** archive `created_at` date plus one day, with field end also on or before the election-minus-one-day cutoff. There are no verified original publication dates or historical revision ledgers. No post-election-created poll row is admitted, but this does not prove today's reconstructed archive was public then. Source snapshots/receipts were acquired or reconstructed later. The audit cannot establish actual publication lag.

The source authors' older consistent panel is unsuitable for this definition: it lacks verified sample/universe/start/publication fields, and its 2016 series ends nine days before election after the documented election-date correction. It was not mixed into this audit.

The ordinary eligible roster excludes unsupported special, simultaneous-special, runoff/RCV and non-D/R settings according to the existing canonical gates. It is not a full-chamber or all-candidate analysis. Third-party/undecided mass is excluded from D/(D+R); nomination/field changes are not inferred from names alone.

## Truth joins and exclusions

2018 truth uses the existing separate allowed-cycle outcome export; 2020 and 2024 use the existing separate cross-context outcome exports. These are eventual labels, not historical predictors. Their underlying retrospective official/archive qualification remains inherited, including the 2024 Clerk compilation/text-derivative provenance.

2022 uses the existing official event candidate-vote export and admitted candidate support file. A source D or R group must have exactly one named member, an exact lexical full-name match to the fixed publisher poll roster (existing normalizer removes punctuation/accents/terminal party tags only), and complete official rows for its FEC identity. Fusion ballot lines for that identity are summed. This does not assert an unprovided publisher-to-FEC person crosswalk. Every party join is retained in [truth-2022-event-party-joins.csv](truth-2022-event-party-joins.csv).

Four 2022 races are excluded from primary truth rather than repaired: Indiana (Thomas McDermott versus Thomas M., Jr. McDermott), Oregon (Jo Rae versus Joe Rae Perkins), Pennsylvania (John versus John K. Fetterman), and North Carolina (the support-file Republican FEC identity has no complete matching official vote row in this export). These may warrant an explicit source-reviewed alias/identity correction, but none was invented here. There are 2,050 native-selected LV/RV poll rows before the truth join and 1,960 afterward; 90 rows lose the required candidate-truth join. These are full-history selected rows, not 1,960 final-week independent cases. Missing-truth and sparse-window exclusions are recorded separately in [coverage.csv](coverage.csv).

## Sensitivities

| Definition | Rising threshold | Cases | Below −1 pp | Within ±1 pp | Above +1 pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary final-minus-prior field-end trend | Any nonzero | 29 | 11 | 9 | 9 |
| Primary | ≥1 pp | 12 | 4 | 6 | 2 |
| Primary | ≥2 pp | 3 | 2 | 1 | 0 |
| Same-pollster matched means | ≥1 pp | 1 | 1 | 0 | 0 |
| Independent earlier trend, same final benchmark | ≥1 pp | 10 | 3 | 3 | 4 |
| Actual field-midpoint windows | ≥1 pp | 6 | 1 | 3 | 2 |
| Exclude explicitly partisan samples | ≥1 pp | 10 | 4 | 4 | 2 |

Same-pollster matching averages within each shared pollster first, then weights shared pollsters equally in both windows; at least two shared pollsters are required. Only four races support any nonzero matched trend and one meets the primary rising threshold. This is insufficient to isolate genuine movement from pollster composition. The independent earlier-window matched version has zero adequately matched cases; its zero denominator remains explicit in [summary.csv](summary.csv).

The independent sensitivity measures trend from field ends 15–21 to 8–14 days before election, while retaining the final 1–7-day poll average as the error benchmark. All three windows must have two samples/two pollsters. Its mean oriented error is +0.63 pp rather than the primary −0.97 pp, with different race membership. This is not a paired estimator or proof of reversal. It reduces the direct shared-late-window noise below, but common pollster/race/national errors remain.

**Mechanical association:** the primary trend is `late−prior`, while error is `actual−late`. The same late-poll noise appears with opposite signs. Conditioning on a large observed rise can therefore select positive late noise and create apparent underperformance/regression toward the mean even without a true reversal. Selecting the rising candidate also changes orientation case by case. Same-pollster matching, independent earlier windows, and fixed-D exports are descriptive checks, not causal identification.

[fixed-D-comparisons.csv](fixed-D-comparisons.csv) retains signed Democratic trend and signed Democratic miss for flat/falling/rising comparisons without double-counting opposite candidates. No significance claim treats the many races as independent national cycles; there are only four cycles and selection/coverage vary substantially.

## How old are the final polls?

Among selected samples whose **field end** is in the final week:

| Cycle | Samples | Median field-end age at election | Median field-midpoint age | Median field duration |
| --- | ---: | ---: | ---: | ---: |
| 2018 | 33 | 6 days | 7.5 days | 3 days |
| 2020 | 46 | 5.5 days | 6.75 days | 4.5 days |
| 2022 | 30 | 6 days | 6.75 days | 3 days |
| 2024 | 82 | 5 days | 7 days | 4 days |

Some long field periods have substantially older midpoints (maximum 12.5/16/11/20 days respectively). [metadata.json](metadata.json) includes minima/medians/p90/maxima. These data support roughly week-old median underlying interviews in this selected late-poll subset; they do not establish that most polls are uniformly one to two weeks behind, nor that older interviews imply an unmodeled momentum effect. The aggregator availability proxy is typically about two days after field end here, but it is not verified pollster publication lag.

Midpoint sensitivity uses actual starts, with continuous age intervals [1,8), [8,15), and [15,22); half-day midpoints do not disappear into gaps between integer-day labels. The primary field-end integer windows remain unchanged.

## Reproduction and checks

From the parent lab directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/hayden/projects/wethervane-redesign-20261006/integration/.venv/bin/python \
  momentum-audit-v1/audit.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v momentum-audit-v1/test_audit.py
```

The script imports the canonical read-only helper and writes only in this directory. It does not invoke either intake's `main`, fit a model, change owner data or inspect/control a live pilot. [receipt.json](receipt.json) fingerprints the source archive, roster/truth exports, helper source, plan and executable audit. Four independent export checks pass: source hashes/no-fit claims, native sample/universe/availability boundaries, race-once primary arithmetic independently reconstructed from selected samples, and every threshold/cycle frequency recomputed from race rows. The CSVs preserve raw selected rows and aggregates so further reviewed sensitivities can be recomputed without qualifying this as predictive evidence.
