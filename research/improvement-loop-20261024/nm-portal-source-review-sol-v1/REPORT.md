# New Mexico portal source review

Independent parsing of the original New Mexico Secretary of State homepage and its linked candidate portal confirms the root packet's extraction. The homepage link titled **2026 General Election – Candidate Filing Day** targets `CandidateList.aspx?eid=2917&cty=99`. The portal title is **2026 General Election Contest/Candidate List**. Its selected controls are county **State (99)**, all contests **(0)**, and all parties **(0)**.

The Telerik grid displays **524 candidate rows** with `AllowPaging=false`, `PageCount=1`, and `CurrentPageIndex=0`. `PageSize=10` and `VirtualItemCount=0` are inactive configuration values in this unpaged result; they do not describe the visible row count. The independent status totals are 503 Qualified, 11 Disqualified, and 10 Withdrawn.

The United States Senator contest has four rows: **BEN R LUJAN** and **LARRY E MARKER** are Qualified; **MIRA OCONNELL** and **BOB PERLS** are Disqualified. Six other contest rows explicitly include the `(write-in)` suffix. None of the four Senate names has that suffix. The check preserves the displayed contest and the underlying office as separate columns; 329 other rows differ between them because the displayed contest can include a district.

This state-selected portal materially improves on the earlier Santa Fe County-only evidence and demonstrates that the portal can display named write-ins. It shows the roster and statuses at the October 25 retrieval. It does not certify that no unrecorded or later qualified Senate write-in filing exists, and the present bytes do not establish what had been published by the October 20 forecast cutoff. The earlier Luna reconciliation's portal-unavailable condition is now historical. Source admission and cutoff qualification remain with the owner.

The review reads only the root packet's raw files. It extracts no contact or address fields and makes no model, owner, source-admission, runtime, or scorer change.
