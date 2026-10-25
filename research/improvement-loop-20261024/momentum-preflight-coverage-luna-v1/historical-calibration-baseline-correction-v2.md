# Historical baseline ledger correction v2

The first-pass baseline summary used all earlier training cycles correctly, but its detailed selection ledger iterated only the last individual training-cycle frame. This version supersedes that ledger with every source row from every strictly earlier cycle. The historical calibration operator and its summary counts are unchanged.

The new ledger has 6,395 target-fold/source-row records (2018: 2012/2016; 2020: 2012/2016/2018; 2022 and 2024: 2012/2016/2018/2020). Every selected row is retained as a separate original observation ID; no source rows are combined.

The deterministic check reconstructs every field of all 16 existing fold/cycle summary rows from the complete source rows and selected IDs. It also confirms every earlier training cycle is represented and that raw 0–100 and fixed-bin selected row counts are recoverable from the ledger. No outcomes, model fit, or score are used.

Use historical-calibration-baseline-id-ledger-v2.csv as the complete detailed ledger and historical-calibration-baseline-summary-recheck-v2.csv as the reconstructed summary check. The first-pass files remain preserved.
