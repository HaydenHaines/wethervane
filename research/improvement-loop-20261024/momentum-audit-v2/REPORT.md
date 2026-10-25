2026-10-25. Descriptive Senate momentum audit v2, using only the independently reviewed 2022 identity repairs. The [source review](../momentum-identity-review-sol-v2.md) documents exact raw cells and limitations; [v1's report](../momentum-audit-v1/REPORT.md) remains the full inherited methodology. No fitting, causal inference, model qualification or activation is authorized.

The primary rising-candidate threshold of at least one D/R-conditional **share percentage point** remains **12 race cases: 4 below −1 pp, 6 within inclusive ±1 pp and 2 above +1 pp**, mean oriented miss −0.965644 pp. These counts are unchanged from v1. There are still four national cycles, not independent national experiments for every race or candidate.

| Scope | v1 | v2 | Reason |
| --- | ---: | ---: | --- |
| Native-selected LV/RV samples before truth | 2,050 | 2,050 | Poll selection unchanged |
| Selected samples with supported truth | 1,960 | 2,050 | Four reviewed joins restore 90 rows |
| Races with primary field-end window coverage | 29 | 31 | NC and PA now have truth; original coverage bar retained |
| Any nonzero primary trend cases | 29 | 31 | Below/within/above become 12/10/9 |
| Primary ≥1pp cases | 12 | 12 | Both new field-end trends are below 1 pp |
| Primary ≥2pp cases | 3 | 3 | Unchanged |
| Matched-pollster ≥1pp cases | 1 | 1 | Unchanged |
| Independent earlier-window ≥1pp cases | 10 | 10 | Unchanged |
| Actual field-midpoint ≥1pp cases | 6 | 7 | NC added; below/within/above 2/3/2 |
| Exclude explicitly partisan ≥1pp cases | 10 | 10 | Unchanged |

Exactly 90 selected historical sample/question rows are restored: IN 2, NC 34, OR 4 and PA 50. IN/OR remain below the original two-window sample/pollster coverage bar. NC's field-end trend is D +0.122228 pp and oriented miss +0.042125 pp. PA's trend is R +0.560321 pp and oriented miss −2.795031 pp. Neither enters the primary ≥1pp denominator. NC's midpoint trend is R +1.911026 pp with oriented miss −1.470719 pp, explaining the additional midpoint case. These are descriptive changes from supported truth joins, not new polls or a relaxed selection rule.

[plan.json](plan.json) is byte-identical to v1. The audit retains original eligible ordinary D/R roster, canonical FiveThirtyEight helper/sample selection, LV/RV universe, election-minus-one-day created-plus-one-day availability proxy, field-end/midpoint windows, equal selected-sample weights, matched-pollster rules, explicit-partisan sensitivity, exact-zero exclusions and thresholds 0/1/2 pp. The algebraic shared-late-window association and original archive/publication limitations remain. One conditional-share point is two conditional-margin points. No original publication or archive revision chronology is newly verified.

The three IN/OR/PA name variants are event-specific reviewed crosswalks. NC uses only surrounding-whitespace trim of the source FEC-ID key; source bytes remain preserved. Complete official category rows and workbook totals for the four races were independently checked, including non-D/R/write-in categories. D/R conditional truth uses the principal D/R counts. Identity keys with conflicting raw IDs after trimming reject rather than merge. No generic fuzzy-name matching is introduced.

Reviewable outputs are [race-deltas.csv](race-deltas.csv), [comparison-deltas.csv](comparison-deltas.csv), [summary-deltas.csv](summary-deltas.csv), [source-review-evidence.json](source-review-evidence.json), and the full selected poll/comparison/truth-join exports. All prior selected poll cells and shared comparison values are unchanged. v1 remains intact, with all 12 original files inventoried and verified by hash. `receipt.json` binds source inputs, the exact v1 plan, code and reviewed source packet; `delta-receipt.json` binds the delta exports. Outputs use exclusive creation guards; rerunning the producer into this populated directory rejects.

From the lab root, independent checks are:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s research/improvement-loop-20261024/momentum-audit-v2 -p 'test_*.py' -v
.venv/bin/ruff check research/improvement-loop-20261024/momentum-audit-v2/*.py
.venv/bin/ruff format --check research/improvement-loop-20261024/momentum-audit-v2/*.py
```

The tests independently recompute primary race arithmetic, every summary/threshold/cycle cell, sample/universe/availability constraints, all restored-row identities, raw official category totals, unchanged v1 bytes, native computation AST equality, whitespace collision rejection and crosswalk rejection for a wrong publisher identity. They do not fit a model.
