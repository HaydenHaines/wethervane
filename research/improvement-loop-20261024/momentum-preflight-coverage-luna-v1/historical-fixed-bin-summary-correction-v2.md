# Fixed-bin horizon summary correction v2

The earlier corrected table used the correct selected IDs and row/race counts, but reported repeat-group and span summaries computed from the larger full-series input even on the fixed-bin operator rows. This version recomputes every summary metric from that operator's retained IDs. The exact complete row-ID ledger is historical-fixed-bin-horizon-selection.csv.

The deterministic reconstruction checks all 96 cells (four folds × three horizons × four extra lags × two operators). It confirms row/sample/race/pollster/F0 support counts from each operator-specific ID set. Fixed-bin repeat groups and spans are now computed from selected fixed-bin rows only.

Use historical-fixed-bin-horizon-correction-v2.csv as the authoritative historical cutoff/truncation table. The earlier corrected table remains preserved with its repeat metrics marked superseded in this note; the unchanged 0–100 calibration baseline is a separate operator.
