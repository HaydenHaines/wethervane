# Advertised field-interval overlap supplement

The native full-series target view contains repeated samples from the same named pollster in the same race. This supplement checks whether their reported field-start/end intervals intersect. It does not determine respondent-level timing, repeated respondents, shared samples, or covariance.

At zero additional lag, the repeated race/pollster groups and groups with at least one interval intersection by cycle/horizon are:
- 2018 H30: 15 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2018 H14: 25 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2018 H7: 37 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2020 H30: 19 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2020 H14: 32 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2020 H7: 45 repeated groups; 1 group(s) with overlapping advertised intervals; 1 intersecting pair(s).
- 2022 H30: 19 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2022 H14: 38 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2022 H7: 53 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2024 H30: 50 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2024 H14: 74 repeated groups; 0 group(s) with overlapping advertised intervals; 0 intersecting pair(s).
- 2024 H7: 91 repeated groups; 1 group(s) with overlapping advertised intervals; 1 intersecting pair(s).

Thus overlap is sparse in this archive: zero intersecting pairs at 2018 H30/H14/H7, 2020 H30/H14 and H7 has 1; zero at 2022 H30/H14/H7; 2024 H30/H14 zero and H7 has 1. The exact group/pair IDs and dates are preserved in the two CSVs.

The 2022 selected 538 archive includes two 91-day ActiVote records. The Florida series (observation 538:80838:2022:FL:senate:regular, field 2022-06-27 through 2022-09-25, availability proxy 2022-09-29) passes the field-end and proxy filters at H30/H14/H7 under all four extra-lag values. The North Carolina series (538:81483:2022:NC:senate:regular, field 2022-08-05 through 2022-11-03, proxy 2022-11-04) fails the requested H30/H14/H7 cutoffs because its reported field end is after them. These are archive metadata classifications only; the long Florida interval is not expanded to daily polls or treated as known respondent dates.

Distinct sample and observation IDs do not prove independent respondents. The counts are not a correction for overlap or an authenticated measurement covariance. A later likelihood decision must keep those limits visible.
