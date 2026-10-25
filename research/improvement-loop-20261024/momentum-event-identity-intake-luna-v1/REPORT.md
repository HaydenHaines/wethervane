# Five Senate archive event-identity gaps — bounded intake v1

**Purpose.** Resolve only the five F0 bank-only regular-event joins previously flagged in `momentum-preflight-coverage-luna-v1`. This packet is a proposed event crosswalk, not an experiment admission. No polling rows are added to selection, and no fitting or scoring is performed.

## Decision

| Target F0 event | Proposed archive event | Status |
|---|---:|---|
| 2018 Minnesota Class I — Amy Klobuchar / Jim Newberger | 538 race 107; Class I, general | Supported match |
| 2018 Mississippi Class I — David Baria / Roger F. Wicker | 538 race 108; Class I, general | Supported match |
| 2022 California Class III — Alex Padilla / Mark P. Meuser, regular full term | 538 races 8921 and 9480 are both plausible | **Unresolved; keep both out of a regular-event join** |
| 2022 Oklahoma Class III — Madison Horn / James Lankford, full term | 538 race 8943; Class III, general | Supported match |
| 2024 California Class I — Adam B. Schiff / Steve Garvey, regular full term | 538 race 9506; special comparison race 10013 | Supported event identity for 9506 |

The 538 archive does not contain a `special` boolean. Its event fields are race ID, seat number/name, stage, and election date. Class plus the candidate pair resolves Minnesota, Mississippi and Oklahoma. For 2024 California, the named original publisher questions and their values distinguish the two same-pair source events. For 2022 California, the two source events use the same nominees, same Class III label and same date; the archive questions have distinct internal candidate IDs but no recorded event text, and the source poll question IDs are not linked to the archive question IDs. This is a true event-identity ambiguity, not a missing-polls conclusion.

## Evidence by event

### 2018 Minnesota

F0 names the regular Class I event `2018:MN:senate:regular`, seat `MN:senate:class_1`, Amy Klobuchar / Jim Newberger. The 538 archive event 107 is a November 6 general-election Class I event with candidate IDs 12373 Amy Klobuchar and 12374 Jim Newberger. The archive also has race 129, Class II, for Tina Smith / Karin Housley. Minnesota Secretary of State 2018 results separate “Votes for U.S. Senator” (Newberger/Klobuchar) from “Votes for U.S. Senator — Special Election” (Housley/Smith), corroborating the distinction. [MN Secretary of State 2018 results](https://www.sos.mn.gov/elections-voting/election-results/2018/2018-general-election-results/?searchTerm=RESULTS); [FEC Federal Elections 2018](https://www.fec.gov/documents/2705/federalelections2018.pdf).

### 2018 Mississippi

F0 names the regular Class I event `2018:MS:senate:regular`, David Baria / Roger F. Wicker. Archive race 108 is Class I, general, with candidate IDs 11416/11417. The FEC’s official 2018 results label this pair Mississippi **Full Term**. The simultaneous vacancy contest is explicitly **Unexpired Term**, with Hyde-Smith/Espy in the general election and a later runoff; the archive keeps it under Class II race 130 (jungle primary) and 6209 (runoff), with separate candidate IDs. [FEC Federal Elections 2018](https://www.fec.gov/documents/2705/federalelections2018.pdf); [FEC notice for the Mississippi 2018 special election](https://www.fec.gov/updates/mississippi-special-election-reporting-senate-2018/).

### 2022 California — unresolved

F0 target `2022:CA:senate:regular` is Class III, regular/full term, Alex Padilla / Mark P. Meuser. California’s official certified candidate list and Statement of Vote establish two separate Class III Senate contests on November 8: full term and partial/unexpired term, with the **same two candidates** in both. The archive contains race IDs 8921 and 9480; both say Class III, general, November 8. One uses candidate IDs 29277/29384; the other 30797/30798. Neither the archive records nor the checked crosswalk carry text saying which is regular and which is the partial-term race. [California SOS certified candidate list](https://elections.cdn.sos.ca.gov/statewide-elections/2022-general/cert-list.pdf); [Full Term SOV](https://elections.cdn.sos.ca.gov/sov/2022-general/sov/42-uss-ft.pdf); [Partial/Unexpired Term SOV](https://elections.cdn.sos.ca.gov/sov/2022-general/sov/45-uss-pt.pdf).

Publisher documents confirm two distinct questions existed, but they do not supply a trustworthy key from their question numbering to 538’s question IDs. SurveyUSA’s original poll page says voters were asked separately about the regular six-year term and the remainder term; its main page shows those as questions 4 and 5. The 538 archive has two adjacent question IDs, 163133 and 163134, with 56/34 for both, but its CSV does not expose the question text or publisher’s item number. Treating order as identity would be a guess. Research Co’s 2022 cross-tabs also print distinct special and regular items, but the archive’s associated q165129/q165130 rows (53/37) do not match the source table’s likely-voter or decided-voter figures, so those values cannot establish the event mapping. [SurveyUSA original report](https://results.surveyusa.com/client/PollReport.aspx?g=f36e5b4d-140b-4cc1-b061-603e980136b7); [Research Co original 2022 tables](https://researchco.ca/wp-content/uploads/2022/11/Tables_StateRaces_USA_07Nov2022.pdf).

**Disposition:** do not map either archive race to the F0 regular event yet, combine them, or count both as independent polls. A source-issued poll-question/event crosswalk or original poll-question field tied to the archive records would resolve this gap.

### 2022 Oklahoma

F0 target `2022:OK:senate:regular` names Class III / full-term Madison Horn and James Lankford. Archive race 8943 is Class III, general, with candidate IDs 30519/29287. The FEC’s official 2022 Senate compilation explicitly labels the pair Oklahoma **Full Term**. Archive race 9482 is Class II and contains Kendra Horn / Markwayne Mullin; the FEC labels that simultaneous vacancy contest **Unexpired Term**. [FEC Federal Elections 2022](https://www.fec.gov/documents/5675/federalelections2022.pdf); [Oklahoma Election Board 2022 results index](https://oklahoma.gov/elections/elections-results/election-results/2022-election-results/2022-november-general-election.html).

### 2024 California

F0 target `2024:CA:senate:regular` is Class I, regular/full term, Adam B. Schiff / Steve Garvey. The California Secretary of State’s Official Voter Guide states there were two separate contests and names both candidates in both. The archive has race 9506 and race 10013; both use Class I/general and the same candidate names, but separate IDs. Research Co’s primary November 2024 cross-tabs explicitly label two questions: special election likely voters Schiff 61% / Garvey 35%, regular election likely voters 60% / 37% (decided-voter rows are 64/36 and 62/38). The 538 archive values agree: question 216192 is 61/35 on race 10013 with candidate IDs 31184/31187; question 216193 is 60/37 on race 9506 with candidate IDs 30942/31183. This supports race 9506 as the regular event and race 10013 as the special event. It does not qualify historical publication timing, vintage, or sample admission. [California SOS Official Voter Guide](https://vigarchive.sos.ca.gov/2024/general/voter-info/info-about-candidate-statements.htm); [California SOS 2024 Statement of Vote](https://elections.cdn.sos.ca.gov/sov/2024-general/sov/complete-sov.pdf); [Research Co original 2024 cross-tabs](https://researchco.ca/wp-content/uploads/2024/11/Tables_USStates_04Nov2024.pdf).

## Exact IDs and provenance

`f0-target-events.csv` preserves the five named F0 rows and original source line numbers from the label-free feature file. `archive-event-question-candidates.csv` preserves raw source values and row numbers from the pinned FiveThirtyEight historical Senate archive for the eleven relevant source event IDs, including potential special events and every archived question/candidate row. The archive has no explicit special flag. Event IDs and question IDs are source IDs; they are not FEC IDs.

`crosswalk-proposal.csv` gives, for each target, the F0 target metadata, exact archive race/seat/stage/date IDs, all question IDs, candidate IDs, comparison event IDs, and the proposed disposition. The original full archive remains at its pinned source path; this packet does not rewrite it or the accepted coverage packet.

Raw official and publisher documents are saved in `raw/`; `source-inventory.json` records their direct URLs, byte counts and hashes. The exact local input files and hashes are in `receipt.json`. All sources here are used only to establish race identity; no outcome is copied from one event to another.

## Limits and next handoff

- This is an event-identity proposal only. It makes no new polling-coverage or forecast-performance claim.
- No polls were selected, admitted, added, reweighted or scored. No model was fitted.
- The historical archive and F0 roster are retrospective sources. This packet does not certify poll-publication dates or as-of vintages.
- 2022 California remains unresolved because the same candidate pair appears in two contests and there is no event key between the 538 question IDs and the source survey question numbers. Keep both question groups visible but excluded from the regular-event intake until a source mapping is supplied.
