# CPS replicate-weight availability and vintage follow-up

Research completed 2026-10-26. Official Census pages and small SAS-layout text were inspected through Census web sources. No microdata or replicate-weight data file was downloaded.

## Finding

There are public 2018 and 2020 person replicate-weight files outside the November Voting and Registration year pages, but the files verified here belong to the **Annual Social and Economic Supplement (ASEC)**. Their existence does not establish a Voting-supplement replicate file or a valid weight crosswalk for voting-registration responses.

The Census 2018 ASEC page lists a replicate-weight instruction, SAS input statements, and a data file. Its official SAS layout defines `PWWGT0` through `PWWGT160` and ends with `H_SEQ` and `PPPOS`. The 2020 ASEC page similarly lists SAS/CSV files that include replicate weights and separate replicate data/documentation; its SAS input layout also ends with `H_SEQ` and `PPPOS`. Thus the ASEC replicate records can be aligned to their associated ASEC person rows on the documented household-sequence and person-position keys. Census's public-use ASEC guidance describes merging person, household, and replicate-weight files. This is a within-ASEC linkage, not a link to the November Voting Supplement.

The ASEC uses 160 replicate estimates. Census CPS documentation describes successive-difference replication, with the direct variance formula `4/160 * sum((theta_r - theta_0)^2)` for `r=1..160` in official CPS ASEC source-and-accuracy documentation. The 2020 ASEC release has an additional material vintage distinction: Census exposes both 2010-Census-based and 2020-Census-based replicate weights for 2020 ASEC, with official guidance assigning the two to different comparison purposes. These are alternative ASEC population-control weight vintages; neither is a Voting Supplement replicate-weight file.

For November voting records, the year-specific 2018 and 2020 Census pages list the supplement data file but no separate replicate-weight section/file. Their November public-use directories likewise list `nov18pub` / `nov20pub` but no `nov18rep` / `nov20rep`. The 2018/2020 Voting technical documentation states that there is no supplement weight associated with the supplement and directs use of `PWSSWGT` for point estimates. This supports the limited conclusion that no Voting-specific replicate-weight file was observed in the official year pages/directories checked. It does not prove that no other unlinked file exists anywhere in Census archives, and it does not make ASEC weights valid for Voting respondents.

## Vintage and release limits

The official ASEC dataset pages currently expose the files and SAS layouts, but their page-revised dates (2018 ASEC: 2021-10-08; 2020 ASEC: 2022-10-27) are not first-release dates for each raw data file. The CPS FTP directory's `Last modified` timestamps are useful current archive metadata, not proof of first public availability or immutable bytes at the historical date. The 2020 alternate control-based replicate files especially make it necessary to identify which version was acquired. I did not find a contemporaneous source establishing exact first-public-release timestamps and exact then-current bytes for both ASEC replicate files. Therefore this work establishes current official availability and structure, not cutoff-valid historical vintage.

## Source record

All pages below are primary Census Bureau sources. Web access checked 2026-10-26.

| Evidence | Official source |
|---|---|
| 2018 ASEC release page; replicate instructions/file; revised 2021-10-08 | https://www.census.gov/data/datasets/2018/demo/cps/cps-asec-2018.html |
| 2018 ASEC replicate layout; `PWWGT0..160`, `H_SEQ`, `PPPOS` | https://www2.census.gov/programs-surveys/cps/datasets/2018/march/CPS_ASEC_ASCII_REPWGT_2018.SAS |
| 2020 ASEC release page; two control vintages, replicate instructions/files; revised 2022-10-27 | https://www.census.gov/data/datasets/2020/demo/cps/cps-asec-2020.html |
| 2020 ASEC replicate layout; `PWWGT0..160`, `H_SEQ`, `PPPOS` | https://www2.census.gov/programs-surveys/cps/datasets/2020/march/CPS_ASEC_ASCII_REPWGT_2020.SAS |
| CPS ASEC public-use guide; person/household/replicate merge and purpose | https://www.census.gov/topics/population/foreign-born/guidance/cps-guidance/using-cps-asec-microdata.html |
| 2020/2021 ASEC control-vintage use guidance | https://www2.census.gov/programs-surveys/cps/datasets/2021/march/Guidance_on_Using_Replicate_Weights_2020-2021.pdf |
| CPS design paper, successive-difference replication and 160 replicates | https://www2.census.gov/programs-surveys/cps/methodology/tp63rv.pdf |
| Current CPS November supplement/replicate directory and year links | https://www.census.gov/data/datasets/time-series/demo/cps/cps-supp_cps-repwgt/cps-voting.html |
| 2018 Voting page, file list (no Replicate Weights section) | https://www.census.gov/data/datasets/2018/demo/cps/cps-voting.html |
| 2020 Voting page, file list (no Replicate Weights section) | https://www.census.gov/data/datasets/2020/demo/cps/cps-voting.html |
| 2018 CPS November supplement directory; `nov18pub`, no `nov18rep` observed | https://www2.census.gov/programs-surveys/cps/datasets/2018/supp/ |
| 2020 CPS November supplement directory; `nov20pub`, no `nov20rep` observed | https://www2.census.gov/programs-surveys/cps/datasets/2020/supp/ |
| 2018 Voting technical documentation (`PWSSWGT`, no supplement weight) | https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov18.pdf |
| 2020 Voting technical documentation (`PWSSWGT`, no supplement weight) | https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov20.pdf |

The general CPS supplement-and-replicate site also has a generic person-replicate instruction link, but its availability alone does not identify a voting-specific file, weighting family, or valid year-specific merge. A Word document link was not text-renderable in the web view, so no year-specific voting join/variance instructions are asserted from that file.

## Prior work retained

This follow-up supplements but does not alter [`m01-cps-historical-instrument-luna-v1/README.md`](../m01-cps-historical-instrument-luna-v1/README.md) (SHA-256 `083adabb6b50a919754e503af6f3e039f7a2cef9f21170f188fdd44e5a57118f`) and its `evidence.json` (SHA-256 `2525d68485172721dd1636c59761952364e8dcf9cf6118a187ec4302617c792c`). The earlier caution against claiming all-year voting-replicate availability remains correct; the narrower fact now established is that separate ASEC person replicate files exist for 2018 and 2020.
