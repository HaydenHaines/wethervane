# Paired gasoline challenger score concentration check

This is a small, source-pinned reanalysis of already scored retrospective development folds. It does not fit a model, sample, reopen prediction banks, change which arm is incumbent, or qualify a forecast.

## Question and rule

The existing G01-vs-F0 review reports cycle-level all-race share CRPS, but not whether each aggregate difference is broad or driven by a few races. For each original fold, this analysis joins every `F0/scores.json` row to the same `G01/scores.json` `race_id`, asserts exact one-to-one support, and calculates `G01 share_crps − F0 share_crps`. Lower CRPS is better, so negative favors G01. Every originally scored race remains in the equally weighted per-cycle mean. Cycle means are checked against the independently reviewed per-cycle aggregate to 2e-10. For 2018, the corrected `independent-score-review-2018-r2.json` is used; its correction note says race/stratum scores were already correct in r1, while r2 fixes the missing primary per-cycle comparison field. Four cycles receive equal weight in the final descriptive mean.

## Results

| Cycle | All races | G01 lower / higher | Mean delta | Top-3 signed sum | Remaining signed sum | Remaining mean | Leave-one-out mean range |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 30 | 22 / 8 | −0.002269 | −0.022152 | −0.045921 | −0.001701 | [−0.002573, −0.002077] |
| 2020 | 28 | 16 / 12 | +0.000147 | +0.015224 | −0.011104 | −0.000444 | [−0.000051, +0.000276] |
| 2022 | 30 | 12 / 18 | +0.001970 | +0.034901 | +0.024189 | +0.000896 | [+0.001615, +0.002284] |
| 2024 | 30 | 18 / 12 | −0.001112 | −0.005287 | −0.028067 | −0.001040 | [−0.001320, −0.000965] |

Top-3 means the three races with the largest absolute paired deltas within that cycle; “remaining” covers every other originally scored race. The separate top-three absolute-magnitude shares are 17.7%, 19.6%, 23.0%, and 18.5% of total absolute deltas for 2018, 2020, 2022, and 2024. That magnitude statistic alone cannot establish that the signed net is not top-three-driven because effects may cancel. In 2020, the all-race signed sum is +0.004121 (mean +0.000147), but the top three contribute +0.015224 and the remaining races sum to −0.011104 (mean −0.000444): removing the top three reverses the net sign. In the other three cycles, the top-three signed contribution and remaining signed sum share the all-race sign. The 2020 near-zero cycle mean also changes sign in the leave-one-race-out range. These are descriptive influence diagnostics; they do not explain *why* effects differ by cycle.

## Model-layer interpretation

This makes one weakness in the evidence clearer: the gasoline addition's matched CRPS behavior is cycle-dependent. It improves the all-race CRPS in 2018 and 2024, worsens it in 2022 and is near zero in 2020. In the midterms, opposite signs in 2018 and 2022 prevent a stable midterm-direction conclusion. In presidential cycles, 2020 is close to zero and 2024 improves. Thus these results do not support a general gas-price correction, nor do they establish that the economic layer is the weakest role in the count stack.

The model's empirical weakest layer is not identified by the evidence reviewed here. Momentum has two distinct existing comparisons: (1) a separately reviewed, matched FS5/FS6 aggregate forecast comparison at the H14 cutoff on 97 common-polled race-events across 2018/2020/2022/2024, with the decision to retain FS5; and (2) the descriptive rising-poll audit. The FS5/FS6 result compares a conditional aggregate model component, not the full eight-role count stack; it reports conditional numerical error rather than electoral-generalization uncertainty and inherits separate bank/scale uncertainty. The rising-poll audit is not an arm comparison. Neither ranks the full eight-role stack or establishes a weakest role. See `checkpoints/momentum-forecast-decision-root-v1.json` (SHA-256 `de6d022c5b7948e8421b011fa6f4aca09f103e8a3dc1a046474de27f58757584`) and its acceptance source `checkpoints/momentum-forecast-results-root-acceptance-v1.json` (SHA-256 `f6b1a20f93e1d6a0df5fd4ba134af8ac08a1f95be76aea62917938e4ba7a5f1a`). This corrects the prior description; no new momentum calculation was performed here.

## Reproducibility and limits

Run `python3 analyze.py`. It reads only the 8 small per-arm score JSONs and 4 independent per-cycle reports (2018 r2, which supersedes the incomplete r1 aggregate); `result.json` records every SHA-256 plus the independent audit-seal hash. It uses the original G01/F0 pairing and full all-race masks; there is no re-selection, dropped race, bootstrap, p-value or independent-race inference.

This is exposed retrospective development evidence over four national election cycles, not prospective validation. Races within each cycle share national conditions and forecast error. The decomposition is descriptive, not causal; it cannot isolate gasoline from correlated cycle changes or show that a different source/vintage or feature would improve the model. The retained F0 decision and all forecast qualification limits remain unchanged.
