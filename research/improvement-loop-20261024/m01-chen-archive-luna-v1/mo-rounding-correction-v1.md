# Correction: Missouri Emerson candidate-count comparison

This correction supplements, and does not overwrite, `README.md` or `evidence.json`.

The original note overstated the precision of the Chen-row-to-November Emerson comparison by calling the candidate support counts “compatible” with the release and 538 toplines. The exact arithmetic is:

- Chen historical table: N=750, Kander `numberSupport=341`, Blunt `numberSupport=335`. These are 45.4667% and 44.6667% of N. Rounded directly to nearest whole percentages, they are 45% and 45%, whereas Emerson's Nov. 7 release reports Kander 46% and Blunt 45%.
- The 538 Nov. 5 archive row gives 45.5% and 44.8%. Multiplying those displayed percentages by 750 gives 341.25 and 336, not the Chen counts 341 and 335. A legacy rounding, weighting, unrounded-source, or transformation difference could account for this, but the legacy 1992–2016 file has no documented derivation or poll ID that lets us test which explanation applies.
- The result supports only same pollster, state/cycle, and sample-size proximity, with approximate candidate values; it does **not** establish exact count agreement or an event/sample lineage.
- Tierney's row still exactly repeats Chen's 341/335 D/R values, but its end date is 2016-10-30 and its D/R sum is 676, not N=750. That value match alone cannot determine whether it refers to the Nov. 4–5 release, a misdated or transformed record, or another record.

## Revised status

The Chen 750-person Emerson record remains a plausible candidate for the later Nov. 4–5 Emerson poll because pollster, race, N, and approximate toplines align. It is not verified at exact-count, questionnaire, respondent-sample, or unique-poll-ID level. The Tierney row remains **ambiguous; do not join**. The earlier n=650 Oct. 28–31 Emerson release is a separate poll and does not match the Tierney/Chen counts.
