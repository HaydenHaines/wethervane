# Explicit historical target requirement

This successor preserves hardening-v2 and the original audit evidence. A
historical replay now requires at least one explicit nonempty target event key
in `evaluation_event_keys`; an empty target inventory is rejected rather than
letting target-label intersection checks pass vacuously. Retrospective mode may
use an explicit empty list, but still cannot omit the inventory field.

Twenty-six test groups pass: the prior 22 plus four regressions covering empty
historical targets, explicit empty retrospective targets, missing retrospective
inventory, and detected label reuse against an explicit historical target.
No source authentication, completeness, scientific benefit or fit admission is
established. `authenticated` and `fit_authorized` remain false.

Run from this directory using `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest
discover -v`. Separate verification pins document preserved v2 evidence.
