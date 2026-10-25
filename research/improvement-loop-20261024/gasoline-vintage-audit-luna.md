# EIA weekly gasoline vintage audit: H14 cutoffs

**Date:** 2026-10-24. **Scope:** official-source release audit only. No model fits, coefficients, or owner-repository edits. Raw archived EIA weekly summaries and extracted values are under [`gasoline-source-evidence-v1/`](./gasoline-source-evidence-v1/); extraction and hashes are recorded in [`release-vintage-values.json`](./gasoline-source-evidence-v1/release-vintage-values.json).

## Result

The archive identifies Monday gasoline observations for each requested Tuesday H14 date, and the historical EIA schedule indicates these values would normally have been posted the prior Monday. But the surviving archived records are Wednesday summaries, after the cutoff. They do **not** verify that the exact prices shown in those later summaries were posted on Monday, or that no intervening revision occurred. Treat Monday availability only as a historical-schedule proxy, conditional on on-time release and no revision before the Wednesday summary. The dates are not federal Monday holidays. EIA's prior schedule published the Gasoline and Diesel Fuel Update around 5 p.m. Eastern on Monday; beginning the week of April 7, 2025, the schedule moved to Tuesday around 10 a.m. Do not project the newer schedule back onto these cutoffs.

| Cutoff (Tuesday) | Latest Monday observation (availability inferred from schedule) | Price ($/gal) | 13-week-earlier Monday | Price ($/gal) | 13-week change ($/gal) | Log change |
|---|---|---:|---|---:|---:|---:|
| 2018-10-23 | 2018-10-22 | 2.841 | 2018-07-23 | 2.831 | +0.010 | +0.00353 |
| 2020-10-20 | 2020-10-19 | 2.150 | 2020-07-20 | 2.186 | -0.036 | -0.01661 |
| 2022-10-25 | 2022-10-24 | 3.769 | 2022-07-25 | 4.330 | -0.561 | -0.13876 |
| 2024-10-22 | 2024-10-21 | 3.144 | 2024-07-22 | 3.471 | -0.327 | -0.09895 |

Log change is `ln(P_cutoff / P_13-weeks-earlier)` using the printed source values. Prices are published to $0.001/gallon; changes inherit that rounding. The full calculation, observations, archive release dates, URLs, and SHA-256 hashes are in the JSON receipt.

## Source and timing evidence

The EIA TWIP archived summaries identify a release date and print an explicit gasoline observation date and U.S. price. For example, 2018's October 24 issue says the U.S. regular price was $2.841/gal on Oct. 22; its July 25 issue gives $2.831 on July 23. The other archive pairs give 2020: Oct. 19 $2.150 / July 20 $2.186; 2022: Oct. 24 $3.769 / July 25 $4.330; 2024: Oct. 21 $3.144 / July 22 $3.471. The reports are stored as downloaded HTML files, with exact official URLs and hashes in the receipt.

EIA's older product documentation described the weekly update as published Monday around 5 p.m. Eastern (federal Monday holidays moved release to Tuesday, still for Monday prices). EIA's 2022 Product Highlight also describes the Monday schedule. EIA's March 18, 2025 notice states the schedule changed **beginning the week of April 7, 2025** to Tuesday mornings around 10 a.m. Eastern (Wednesday after federal Monday holidays). This resolves the apparent conflict with today's methodology page, which reflects the new schedule. Primary references: [EIA 2022 product highlight](https://www.eia.gov/TodayinEnergy/detail.php?id=54439), [EIA 2025 schedule-change notice](https://www.eia.gov/petroleum/gasdiesel/notice.php?fs=e&s=cl), [EIA gasoline methodology](https://www.eia.gov/petroleum/gasdiesel/gas_proc-methods.php).

The old/new gasoline sampling methodology break began May 14, 2018. EIA says estimates from May 14 are not directly comparable with May 7; the change included an updated frame, sampling/estimation methodology, and city definitions. Both dates used for the 2018 H14 value and its 13-week change are after this break, so the **within-2018** 13-week contrast does not cross it. It remains a major comparability break for training examples that join pre-May 2018 to later years. EIA's archived gasoline-method page identifies May 7, 2018 as the last release on the old method; the new-method description and two-week alternative-sample comparison are on its [methodology page](https://www.eia.gov/petroleum/gasdiesel/gas_proc-methods.php).

## Availability classification and limits

- **2018, 2020, 2022, 2024 H14 features: historical-schedule proxy only; strict as-of qualification fails.** Under the assumption that EIA posted each Monday update as scheduled and did not revise the figure before the Wednesday archive summary, the shown Monday values would precede Tuesday's cutoff. The exact original Monday posting, value bytes, and no-intervening-revision condition have not been verified. Do not treat these four features as strictly as-of qualified.
- **Later archive is not the contemporaneous vintage.** EIA's archived TWIP HTML preserves exact Monday observation dates and prices in reports released later that week; it does not prove what was public by the earlier H14 cutoff. Raw weekly update captures/HTTP posting logs were not found. Keep the availability assumption explicit in any intake receipt.
- **13-week change is reproducible from archived releases.** The point at each end comes from separate date-stamped EIA archived issues; it is not calculated from today's historical table. It is a simple price change, not CPI-relative or inflation-adjusted change.
- **Earlier-cycle training is promising but not yet admitted.** EIA's public TWIP archive has annual archive groups back through 2002–2005, so archived weekly releases can potentially supply older training vintages from 2002 onward. I did not audit completeness or retrieve election-specific 13-week source pairs for every earlier cycle. The live full-history series alone does not prove cutoff availability. Before using earlier midterms, audit every cutoff/pair the same way and handle the May 2018 method break explicitly. Do not assume full-vintage coverage before the visible TWIP archive begins; pre-2002 archive evidence was not established in this pass.
- **No 2026 real-time claim.** The contemporary release schedule has changed; apply its 2026 publication time and a fully specified cutoff timestamp to any future 2026 join. This historical evidence does not certify today's unexamined live snapshot as an H14 input.

## Safest next intake

Use the EIA weekly national regular gasoline, all-formulations measure in dollars per gallon, with the observation date and publication date/time stored separately. For strict intake, retain a timestamped release artifact or other primary evidence proving that the exact source value was public before cutoff and not revised afterward; a normal publication schedule plus a later archive is only a historical proxy. Derive the 13-week contrast from the same dated source vintage. Pair each value with raw source bytes, exact retrieval URL, hash, observation date, release date/time, revision status, and cutoff comparison. Keep the numeric time-series table as a convenience only, never as the vintaging evidence. All four cases remain unqualified for strict as-of use until their original posting/revision provenance is verified; wider training coverage also needs archive-by-archive audit.
