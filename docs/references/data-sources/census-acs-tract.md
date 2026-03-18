---
source: https://api.census.gov/data/
captured: 2026-03-18
version: ACS 5-year 2022 (covers 2018-2022); ACS 5-year 2019 (covers 2015-2019)
---

# Census ACS at Tract Level

## API Entry Point
`https://api.census.gov/data/{year}/acs/acs5`

Free. API key required (instant, free): https://api.census.gov/data/key_signup.html

## Stage 1 Query Pattern
```
GET /data/2022/acs/acs5
  ?get=B19013_001E,B19013_001M,C24010_001E,...
  &for=tract:*
  &in=state:01,12,13   ← Alabama=01, Florida=12, Georgia=13
  &key=YOUR_KEY
```

## Key Tables for This Project

| Table | Content | Notes |
|---|---|---|
| `B01001` | Sex by age | Population structure |
| `B03002` | Hispanic/Latino origin by race | Racial composition |
| `B08301` | Commute mode | Urban/rural proxy |
| `B19013` | Median household income | Economic character |
| `B23001` | Employment status by age/sex | Labor force participation |
| `B25001`/`B25003` | Housing units / tenure | Owner vs. renter |
| `C24010` | Occupation by sex | Industry/class proxy |
| `DP02` | Social characteristics | Education, household type |
| `DP03` | Economic characteristics | Income, poverty, commute |

Every estimate field (`_E` suffix) has a margin of error field (`_M` suffix). **Fetch both.**

## MOE Flagging Rule
Flag tracts where `MOE / estimate > 0.30` on any primary feature variable. Store the flag as a boolean column. Do not exclude or model the error — just surface it for post-Stage 2 review.

## Vintage Convention
- Use **2019 5-year** (2015–2019) as the training baseline
- Use **2022 5-year** (2018–2022) for current-cycle analysis
- Never mix vintages within a single model run

## Python Access
`cenpy` package wraps the API. Alternatively, use `requests` directly — the API is simple enough that direct calls are fine for batch pulls.

## Gotchas
*Populated as failures are encountered in practice. First entry goes here the first time the API surprises us.*
