# Review notes on the first forecast execution proposal

The Luna proposal `momentum-execution-design-luna-v1.md` is retained unchanged, SHA `604e9542ff6bbb847bd38b7834d548ae62f2a7065f702b33bd413cbcb5456fc3`. It is not an accepted execution contract. Root identified these corrections before any forecast or score:

- The 2024 calibration has only 2012/2016/2018/2020 source cycles. There is no 2022 calibration source. The sealed preparation proves exact 2022/2024 numerical identity, beyond equal row counts.
- The final comparator table row has a missing/misaligned column.
- A proposed polling reference that learns a pooled historical-outcome prior needs an exact source universe, weighting/variance formula and joint law. Calling it “F0 calibration races” is ambiguous. A fixed broad proper prior would be a different explicitly declared reference.
- The F0 likelihood update must integrate its target-cycle and Student-t race effects through the actual selected updater. It is not a direct Gaussian likelihood of coefficients alone. Posterior mixture sampling can repeat intact empirical rows; it must preserve identity and dependence rather than forbid all resampling.
- Latest-pollster versus full-series priority, numeric integration gates and expanded-event admission remain proposals until a concrete matched runner is selected.

Luna is revising the proposal in a separate v2 file. This does not block the independently defined five/six-scale earlier-only calibration runner. Calibration fitting is distinct from forecast sampling and score comparison.
