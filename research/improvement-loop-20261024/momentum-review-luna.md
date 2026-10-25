# Independent review: Senate momentum audit v1

Reviewed `momentum-audit-v1/{REPORT.md,plan.json,audit.py,test_audit.py}` and the exported CSV/JSON artifacts read-only. I did not alter or rerun the audit generator. The four included tests pass (`python3 -m unittest -v .../test_audit.py`).

## Findings

- **Headline arithmetic checks out.** The selected rows feed one oriented case per race and comparison; the final-minus-prior D-share trend, candidate orientation, and actual-minus-final performance formula match the plan. Independent test reconstruction confirms the primary race rows; `summary.csv` recomputes to 12 rising cases (cycle counts 5, 4, 1, 2), with 4 below −1 pp, 6 in the inclusive ±1 pp band, and 2 above +1 pp. The mean oriented miss is −0.966 pp, reported as −0.97. Cycle-eligible and coverage counts also sum to the report’s 113 and 29. These are descriptive poll-average comparisons, not fitted forecast results.

- **Plan provenance supports “documented definition,” but not a strong preregistration implication.** The plan records the primary choices and is receipt-pinned; its filesystem mtime is only about 40 ms before `audit.py`, and there is no prior signed/committed plan history in these artifacts showing that choices were frozen before inspection of outcomes. “Under the prespecified primary definition” is technically possible but invites a stronger inference than the evidence supports. Safer wording: “Under the plan-defined primary definition”; state that the plan and script are receipt-pinned, while prior outcome-blind preregistration is not established.

- **Cutoff wording needs one small precision fix.** `senate_polls_as_of` retains `end_date <= election_date - 1 day` and proxy availability `created_at(normalized)+1 day <= election_date - 1 day`. The report’s phrase “field end also before the election-minus-one-day cutoff” can read as strict `<`; say “on or before the cutoff.” More substantively, the report correctly labels this proxy unverified and says the current reconstructed archive is not an exact historical revision ledger. The source cannot prove what was actually public on the cutoff date.

- **Coverage and sensitivity caveats are adequate.** The primary needs two selected LV/RV samples and two distinct pollster labels in each window; selected samples/questions are equally weighted. The report explains 2022 lexical-match exclusions, names all four affected races, and preserves the joins. It also makes the late-window shared-noise reversal explicit, distinguishes the independent earlier trend’s changed membership from a paired estimate, and reports only four cycles with variable sparse coverage. “Independent earlier trend” refers to non-overlapping trend windows, not independent polls/races; the caveat appropriately says residual errors can remain shared. The matched-pollster analysis is correctly described as too sparse to isolate pollster composition.

## Recommendation

No material correction to the numerical headline is indicated. Keep it explicitly descriptive and selected-case scoped. I recommend the two wording edits above and avoiding any implication that the available plan artifact proves outcome-blind preregistration. The report already avoids a causal, model-qualification, or production-forecast claim.
