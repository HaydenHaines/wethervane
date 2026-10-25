# Exact-midnight CPI adapter: root repair review

October 25, 2026 UTC. This accepts the corrected source-selection adapter, not an economic effect or fitted forecast.

The [independent source review](economic-cutoff-review-luna-v2.md) checked the archived August 2004 BLS release, all 28 selected values and both direct and inherited source pins. The new release supplies headline CPI-U inflation of 2.7% and core inflation of 1.7%, released September 16, 2004 at 12:30 UTC. It is available at the model's October 19 midnight cutoff; the subsequent September release is 12.5 hours too late. Exactly the two 2004 selections change from the earlier end-of-day packet. All four A/B election values remain unchanged.

That review identified a real rebuild gap: the first adapter imported the sibling parser before authenticating it and regenerated its manifest from current inputs. The repaired [adapter](economic-cutoff-adapter-v2/adapter.py) checks fixed, reviewed hashes for the v1 parser, rows, selections and receipt before import/use. It validates every one of the receipt's 20 bounded relative file entries, including raw PDFs, and rejects a changed v2 input manifest. V1 was not changed.

Root inspected the repair and ran all seven tests successfully. Corrupted parser, rows, receipt, nested raw PDF and manifest fixtures fail before output writes. A separate root fresh-process fixture prepended executable sentinel code to a copied parser: importing the copied adapter failed on the expected hash guard and never executed the sentinel. This additionally checks the pre-import boundary rather than only calling `build()` after an earlier safe import. All ten current v2 receipt entries match their byte counts and hashes.

Accepted v2 receipt SHA-256: `4646bc2ea152f441a1b34388ce4bd5693306a065947c3f0d17a09c1f781ac733`.

Unchanged selected-values SHA-256: `208c02593a8f39771ae23248ec2c5608aa493948cda2d90d64a53fdc3fbf0a0b`.

The original review remains as the record of the discovered gap. This note supersedes its pending rebuild-hardening finding only. Archived release copies and nominal release times retain their stated provenance limits; current retrieval does not independently authenticate original server bytes. These inputs support preparation of an earlier-information aggregate comparison, with national replication counted by election cycle. They do not qualify historical population/count inputs or justify a production inflation adjustment.
