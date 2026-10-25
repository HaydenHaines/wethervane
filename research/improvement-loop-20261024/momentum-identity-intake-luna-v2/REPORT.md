# 2022 excluded truth joins: primary-source identity review

**Review date:** 2026-10-25  
**Scope:** Identity/provenance only. This intake does not edit `momentum-audit-v1`, lower its poll-window or coverage rules, fit a model, recalculate momentum frequencies, or create new outcome scores.

## Result

The four joins in v1 are recoverable with explicit documented evidence. Three are verified person-name variations. The North Carolina row is not a missing official vote row: the pinned FEC workbook contains Ted Budd's complete official result, but the raw FEC candidate ID cell has a trailing space. A byte-preserving exact-ID comparison failed because it compared `S2NC00505 ` to `S2NC00505` without trimming. The defensible key normalization is to trim surrounding whitespace from FEC candidate-ID strings only; keep the unmodified source bytes and original cell value in provenance.

| Race | Poll source name | Official names / ID | Evidence-backed resolution |
| --- | --- | --- | --- |
| Indiana | Thomas McDermott | FEC list/result: McDermott, Thomas M., Jr.; ID `S2IN00208` | Same Senate race, state and party; middle initial and suffix are present in official records. Explicit alias is supported. |
| Oregon | Jo Rae Perkins | FEC list/result: Perkins, Joe Rae; ID `S4OR00156`; FEC candidate page: Jo Rae | Official sources themselves vary between “Jo” and “Joe.” State Secretary of State output and FEC candidate page use “Jo”; FEC list/results use “Joe.” Same FEC ID, office/state/party, exact surname and rare two-part given name support one identity. Preserve source spelling; record the alias. |
| Pennsylvania | John Fetterman | FEC list: Fetterman, John; results/manual: Fetterman, John K.; FEC profile: John Karl; ID `S6PA00274` | Same candidate ID and state/office/party; middle initial/name variants are directly documented. Explicit alias is supported. |
| North Carolina | Ted Budd | FEC list/result: Budd, Ted; ID `S2NC00505`; FEC raw workbook row 423: `S2NC00505 ` | Name and ID are exact after trimming ID-cell whitespace. Complete 1,905,786 general votes exist. This is a source formatting/join-key issue, not an absent outcome row or a person-name alias. |

`crosswalk-proposal.csv` records these mappings and the basis for each. `poll-roster-name-evidence.csv` records exact candidate names and archive candidate IDs from the same pinned polling archive used by v1. Those archive candidate IDs are not FEC IDs; the proposed crosswalk deliberately joins names to FEC IDs through official FEC/state election records.

## Primary records inspected

The FEC describes [Federal Elections 2022](https://www.fec.gov/introduction-campaign-finance/election-results-and-voting-information/federal-elections-2022/) as an official compilation of results reported by state/territorial election offices. I downloaded and retained its [FEC results PDF](https://www.fec.gov/resources/cms-content/documents/federalelections2022.pdf) and [Excel workbook](https://www.fec.gov/documents/5676/federalelections2022.xlsx). The workbook was independently byte-identical to the already pinned source workbook in the House intake (`SHA-256 cdb258ea23803e50d752bcea19faa39b3c201eb960094bcb556c3a38c2350897`). Its `7. US Senate Results by State`, row 423 contains North Carolina, Senate, raw candidate ID `S2NC00505 `, candidate name `Budd, Ted `, party R and general votes 1,905,786. This proves the votes and FEC identity are in the source workbook before the downstream CSV export. The derived official-outcomes CSV preserves the trailing space in `fec_candidate_id`.

The [FEC 2022 general-election candidate list](https://www.fec.gov/resources/cms-content/documents/2022congressgecands.pdf) explicitly labels the candidate ID/name/party for each state Senate ballot. It maps IN `S2IN00208` to McDermott, Thomas M., Jr.; OR `S4OR00156` to Perkins, Joe Rae; PA `S6PA00274` to Fetterman, John; and NC `S2NC00505` to Budd, Ted. Individual FEC candidate profile pages corroborate office/state/party and stable candidate ID: [Indiana](https://www.fec.gov/data/candidate/S2IN00208/), [Oregon](https://www.fec.gov/data/candidate/S4OR00156/), [Pennsylvania](https://www.fec.gov/data/candidate/S6PA00274/), and [North Carolina](https://www.fec.gov/data/candidate/S2NC00505/). Pennsylvania's state-published [Manual](https://www.pa.gov/content/dam/copapwp-pagov/en/dgs/documents/publications/documents/pa-manual-v-126.pdf) gives the official statewide Senate result heading `FETTERMAN, JOHN K`. NCSBE's [official 2022 contest pages](https://er.ncsbe.gov/?contest=0&county_id=71&election_dt=11%2F08%2F2022&office=ALL) report Ted Budd, Republican, in county-level rows; the exact statewide 1,905,786 total is independently in the retained FEC PDF and workbook. Oregon SOS's indexed official [abstract](https://sos.oregon.gov/elections/Documents/results/november-general-2022.pdf) and [Blue Book](https://sos.oregon.gov/blue-book/Documents/elections/2022-general.pdf) show “Jo Rae” and the matching 788,991 vote total.

The Indiana archived Secretary of State results URL currently returns a shell page without a candidate table in the captured HTML; the Angular “more information” link resolves to the literal template path and returns 404. The FEC result publication remains the accessible primary record for the complete Indiana candidate result. Oregon's official PDF URL returned 404 to direct retrieval during this pass; its indexed content remains available to search and the FEC results PDF independently carries the Oregon total under the “Joe Rae” spelling. These retrieval failures are preserved in `source-attempts.json`; they are not evidence that source election results are missing.

## What this does and does not change

This evidence supports a v2 join repair: three reviewed aliases, plus a whitespace-trimmed FEC ID key for North Carolina. It does not change the raw poll archive, outcome votes, source candidate IDs, poll-selection logic, trend windows, completeness threshold, or frequency calculations. The original v1 exclusion ledger remains intact. If root adopts these mappings in a separately reviewed v2 audit, it should add an explicit alias/ID-normalization ledger and prove the exact 90 poll rows lost to 2022 missing-truth join are the only rows restored, before reporting any changed frequency.

## Reproduction/provenance boundaries

- v1 report: SHA-256 `96081fdd70202026f153d94c1ed55ea72cf26e34fdbd3b95e461a2436e0f99f2`.
- v1 truth join file: SHA-256 `ad183e349ecd90caa169887ed728d77aac461da33f3669143a84dad340e0dbf1`.
- v1 receipt: SHA-256 `67d073ef97c024522af06697be075d7227208c603f040225081798534219d874`.
- Poll archive source SHA-256: `09ae303a94f76dd761532fce7a356fe3b5beb1e24d4e5ba343c424c372fdfabf`.
- Current v1 roster, candidate support, and derived official vote exports are separately pinned in `source-inputs.json`.

This is a focused intake note; it does not requalify the descriptive momentum results or validate the source archive's historical publication availability.
