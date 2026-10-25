# Independent G01 score audit

verify_scores.py implements CRPS, WIS, Brier/log scores, interval summaries, F0 masks, equal-cycle A/B aggregates, and numerical-error combinations without importing producer scoring or runner code.

Default mode is label-blind. Do not pass --go until root explicitly says requested fold(s) are ready. In scoring mode, it first verifies all requested folds' three prediction seals and the mask bindings, then checks each mask from the F0 prediction bank. It opens saved target labels only after those checks. Without --target it requires all four folds; with root notification for a subset, pass --target YEAR once per ready fold. Output is written in this directory as independent-score-review.json.

The report labels A/B cycles as exposed development evidence. It keeps constant-probability precision limits, empty near-even masks, and cycle-level replication visible.
