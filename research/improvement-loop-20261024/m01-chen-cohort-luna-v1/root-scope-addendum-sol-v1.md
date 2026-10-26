# Scope clarification for the Chen candidate inventory

The original README and evidence remain as issued. The independent review reproduces the pairable candidate-group counts and adds two definition clarifications; it does not admit a source or establish poll identity.

For the stated coarse key `(cycle, state, daysLeft, pollster, samplesize)` after excluding positive `daysLeft`, the reported ambiguous counts of **80** for the 2019 snapshot's 2018 rows and **7** for the 2021 snapshot's 2018 rows require **both Democratic and Republican rows to be present**, with multiple rows on at least one party side. Counting every group with multiple rows on either party side instead gives **105** and **25**. The differences are respectively 25 and 18 single-sided groups. The pairable-group counts of 609 and 562 are unaffected.

The index file “adds only `pollsteridx`” describes its **schema**. It does not mean every shared value string is literally identical to the unindexed combined table. The independent comparison found 2,052 outcome-label string differences at floating-point precision and one `numberSupport` representation of `0` versus `-0`. These do not create a separate cohort or restore missing survey/question IDs. Separately, the combined file's 1,613-row 2018 projection exactly matches the standalone 2019 2018 file in order across the ten compared fields; that distinct projection result remains valid.

These are descriptive metadata predicates, not thresholds for accepting a source, choosing a model or declaring adequate evidence. A uniquely paired group is not a verified unique poll, questionnaire, respondent sample or independent observation. Neither these counts nor Git identities establish historical public availability, exact Tierney lineage, outcome correctness or scientific qualification. Original-release crosswalks, date conventions, sample-size meaning and source overlap remain unresolved.

See `independent-review-sol-v1.json` for the pinned inputs, reproducible counts and limitations. No original author report, source snapshot, training file or owner artifact was changed by this addendum.
