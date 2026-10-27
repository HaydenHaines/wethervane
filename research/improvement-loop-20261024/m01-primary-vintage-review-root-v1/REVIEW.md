# Independent check of two historical page captures

I independently retrieved the two archived publisher pages and their Internet Archive CDX records identified in `../m01-primary-vintage-authentication-luna-v1/evidence.json`, verifying that the exact response bodies match both the earlier SHA-256 pins and the CDX SHA-1 digests. The script emits hashes, capture times and content-check booleans only; it does not export poll rows, respondent information or page bodies. Internet Archive remains the external authority for its recorded capture time and digest.

The October 28, 2012 Rasmussen capture contains the identified Arizona question-page content, a 500-likely-voter October 21 field date, and candidate-order rotation. It supports the existence of that captured page by the archive's recorded time. This is a **2012** poll question, not an A2018 observation; the check does not itself authenticate its poll-event crosswalk, original first-publication time, top-line result, or individual samples.

The May 30, 2018 FiveThirtyEight article capture has the checked page timestamp and links to the pollster-ratings dataset through a mutable `master` tree. The independently repeated CDX query for the commit-pinned raw CSV returned no capture. The article supports public offer of the dataset family, but neither proves the exact CSV bytes available at that time nor proves that no other archive exists. I did not independently verify the GitHub signature or all prior source linkages.

The evidence packet's exact-version limits and `strictAsOf` warning therefore remain appropriate. Nothing here authenticates all 2018 observations, sampling independence, compiled model inputs, a forecast, or a layer ranking.

Reproduce with network access from this directory using `python3 -B reproduce.py`. The script checks the earlier evidence-file hash first and fails on altered response bodies, missing captures, or a new result to the exact-CSV query. External content and network availability can change; a later failure requires investigation, not automatic scientific admission.
