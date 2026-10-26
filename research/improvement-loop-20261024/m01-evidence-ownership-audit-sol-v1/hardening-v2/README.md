# Research validator input hardening

This successor preserves the parent audit's sources, inventory and verification
bytes. It changes malformed-input handling only: ledger/record/claim mappings,
list containers, string identities, evidence pins and explicit aware datetime
inputs are checked before iteration, hashing or comparison. Invalid inputs
return `unresolved_or_rejected` with errors rather than raising ordinary
container/type exceptions or silently dropping bad fields into a success.

Twenty-two test groups pass: the original eleven scientific-contract fixtures
and eleven malformed-input groups with multiple cases. Date tests reject naive
versus aware mismatches and compare different valid timezone offsets as actual
instants. Duplicate records, edges and list elements are rejected. Every
synthetic success still explicitly reports `authenticated=false` and
`fit_authorized=false`; no new source, scientific or admission claim follows.

Run from this directory with `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest
discover -v`. `verification.json` records test counts, source hashes and the
unchanged parent evidence check. No fit, sampling, registry or owner changes.
