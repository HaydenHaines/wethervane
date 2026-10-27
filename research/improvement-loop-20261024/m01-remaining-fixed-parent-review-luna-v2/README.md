# Independent review: remaining fixed-parent verifier V2

V2 closes the specific V1 prelaunch completion-marker gap within its declared cooperative-client trust model. Before fixed-selected, the controller now requires a retained caller tool-return record and exact successful response matching the current controller marker, plus the exact phase artifact roster, checked comparison schemas/pins and reviewed-source identity. The frozen tests reproduce V1's old acceptance and verify that V2 rejects marker-only, empty/missing/extra artifact, altered comparison, stale result, nonzero tool exit and malformed/mismatched return cases.

I independently checked every own-file pin in V2 evidence.json: 16 files, zero mismatches. The bounded suite passed all 65 tests in a fresh temporary directory tree containing V2, preserved V1 and its Luna reproducer as siblings (a single-directory copy lacks the relative fixtures expected by the suite). No graph or owner job ran.

The remaining boundary is accurately documented: retained caller output and local hashes provide cooperative workflow provenance, not cryptographic protection against a deliberately dishonest same-user actor. Invented publisher fixtures do not authenticate model calculations. Actual resource feasibility and all scientific normalization/inference claims remain unverified. No actual phase admission was created.

See review.json, source-hashes.txt, and test-run.md for pins and exact reproduction details.
