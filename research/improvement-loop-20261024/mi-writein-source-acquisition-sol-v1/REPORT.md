# Michigan 2026 U.S. Senate write-in source pass

The Michigan Department of State's linked 2026 general-election candidate report lists six U.S. Senate candidates. The report has no `write` or `declaration` text or field anywhere in its HTML. Its six rows therefore describe the ordinary candidate report, not a complete set of timely write-in declarations.

The Bureau of Elections' *Election Officials Manual, Chapter 4: Write-In Candidates* (July 2024) says write-in candidates for federal office must file a Declaration of Intent with the appropriate election official by 4 p.m. on the second Friday before the election. It says the Bureau or county clerk will notify local clerks of filed candidates. It also states a waiver for an office if a candidate dies or is otherwise disqualified after the deadline. The existing U.S. Senator filing guide specifies the BOE and October 23, 2026, 4 p.m. for general-election declarations.

No official post-October 23 BOE or county U.S. Senate declaration roster was located in this focused pass. A Saginaw county list surfaced in search but its older indexing did not establish a post-deadline version; an ordinary HEAD request met an HTTP 403 Cloudflare challenge and was stopped. The state site initially returned 403 to a generic curl request; the same public GET succeeded with a normal browser User-Agent. No authentication, challenge solving, filing, or contact was attempted.

**Hold:** Preserve the six ordinary candidate rows separately from write-in coverage. Do not infer zero write-ins from this report or from the failed county request. Current retrieval on October 25 does not establish that any source was available at the October 20 forecast cutoff. The manual was published in July 2024; its server `Last-Modified` value is May 19, 2025.

Raw response bytes and headers, exact source URLs, and SHA-256 hashes are pinned in [receipt.json](receipt.json). [extraction.json](extraction.json) contains only office, name, party, and filing-method fields.
