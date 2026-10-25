# Exact historical calibration baseline supplement

This is the unchanged historical polling-calibration selection, shown separately from the H30/H14/H7 cutoff-truncated coverage sensitivities in REPORT.md. The original runner uses every earlier-cycle panel observation with lead in [0,100] days, then retains the latest by end date and observation ID in each cycle/race/pollster/14-day lead bin. It does not apply the target-cycle H30/H14/H7 cutoff to historical training cycles.

The fold sets are strictly earlier: 2018 uses 2012/2016; 2020 uses 2012/2016/2018; 2022 and 2024 use 2012/2016/2018/2020. The 2022 and 2024 F0 folds have the same historical panel years because this source ends at 2020. This is coverage only, not a covariance fit or forecast.

Coverage counts and within-pollster time-span support are in historical-calibration-baseline.csv. Every row ID, fixed bin, selected/excluded disposition, and F0 training-race membership is in historical-calibration-baseline-selection-rows.csv. Source and runner hashes are pinned in the receipt. Historical availability is only field-end proxy; original publication time and exact vintage are unavailable.

- 2018 fold (2012,2016): 1178 raw rows to 930 fixed-bin rows; 59 races with raw polls, 59 with retained bins; F0 training race intersection 59/299.
- 2020 fold (2012,2016,2018): 1515 raw rows to 1218 fixed-bin rows; 85 races with raw polls, 85 with retained bins; F0 training race intersection 85/329.
- 2022 fold (2012,2016,2018,2020): 1823 raw rows to 1514 fixed-bin rows; 106 races with raw polls, 106 with retained bins; F0 training race intersection 106/357.
- 2024 fold (2012,2016,2018,2020): 1823 raw rows to 1514 fixed-bin rows; 106 races with raw polls, 106 with retained bins; F0 training race intersection 106/387.
