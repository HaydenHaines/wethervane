# E01 aggregate inflation comparison: retain the incumbent

October 25, 2026 UTC. The sixteen H0/F0/E01/E01_sd025 historical fits completed under original session 38077 with exit code 0. All declared fit and predictive gates pass. Independent review supports accepting the computational results and **retaining F0 for this comparison**: neither inflation variant earns promotion. The standing improvement goal remains active, and every incumbent, challenger and result is preserved.

The primary question was whether adding one regularized, president-party-oriented headline-CPI coefficient improved the existing relative-context aggregate fundamentals law. All existing coefficients and cycle/race scales were jointly refit. H0 is the electoral-history reference; F0 is the nested control; E01 has coefficient prior SD .05 and the predeclared sensitivity SD .025. This is a fundamentals-only aggregate experiment, with no polling contribution, protected-population count graph, full chamber outcome or current-2026 qualification.

| Historical cycle | F0 share CRPS | E01 minus F0 | Tighter prior minus F0 |
| --- | ---: | ---: | ---: |
| 2018 midterm | .038001598 | +.000544697 | +.000213098 |
| 2022 midterm | .028472070 | +.002250319 | +.000496747 |
| 2020 presidential | .034593891 | −.000045030 | −.000110017 |
| 2024 presidential | .027122186 | +.000135050 | +.000058072 |

CRPS rewards accurate predictive distributions; lower is better, so positive differences are losses. The equal-cycle midterm loss is .001397508 for E01, about 4.2% of F0's score, and .000354923 for the sensitivity. Their numerical integration errors are .000067948 and .000055112. Both midterms worsen individually, as do their competitive-race comparisons. E01's midterm interval, winner-probability and log scores also worsen. The presidential-cycle net differences are small relative to numerical error and offer no clear compensating gain. Near-even midterm aggregation remains unavailable because the fixed 2022 mask is empty; it is not assigned a zero score.

This is evidence against adopting this specification. It does not establish that inflation never matters, isolate a causal economic effect, identify the weakest full-stack layer, or establish F0's superiority on every metric. Two national elections per tier remain only two national realizations. Numerical integration error is not election-to-election uncertainty. The CPI coordinate overlaps party/carryover terms, and 2022 extrapolates beyond earlier raw inflation support. A new gas-price, purchasing-power or trend specification needs its own versioned, matched comparison; tightening the prior after seeing these outcomes cannot turn this run into an outcome-blind success.

[Luna's 2018 review](economic-2018-output-review-luna-v1/REPORT.md) reconciles all transition rows, four chains per arm, warmup/retained coordinates and reconstructed joint predictions. [Sol's all-cycle review](economic-results-review-sol-v1/REVIEW.md) independently checks all sixteen score, source-truth, mask and score-MCSE calculations and the remaining twelve raw-chain parameter diagnostics. Scores agree within 9.91e-15. Stored predictive diagnostic gates pass; the independent review did not rerun every predictive ArviZ calculation. No failed gate was observed. Root inspected both reviews, verified their receipt inventories and separately checked the per-cycle score differences.

The prediction-sealing rule is four arms **within each chronological target fold**, followed by its fixed F0 masks and numeric target scoring. Earlier target outcomes legitimately enter later training. This is revisitable retrospective development evidence, not a globally unopened holdout or independently verified original publication archive.

The [complete output inventory](checkpoints/economic-run-inventory-v1.json) binds all 376 run/log files, totaling 1,416,906,623 bytes. Original paths remain intact, and each exact file is also stored at `.improvement-ledger/objects/<sha256>`. Git retains JSON metadata, scores, logs, reviews and the inventory; large NetCDF/NPZ arrays and transition journals are excluded from Git and preserved in the local journal backup. Recovery copies each inventory hash to its declared path and verifies its byte count and digest. This is local durable storage, not an external backup service.

The decision is **retain**, with no public configuration or forecast change. The original completed process must not be restarted. This completes the bounded E01 comparison, not the continuous improvement goal. Full-stack complementarity and an empirical weakest-layer ranking still require qualified matched joint fits from the scientific owners.
