# Validation-order correction to frozen ancestor optimization documentation

This addendum corrects ../queue-finite-ancestor-sol-v1/README.md and ../queue-ownership-audit-design-sol-v1/DESIGN.md. Their statements that resolved/canonical path validation runs before the overlap helper are too broad. Frozen files and execution code are unchanged. The independent review correctly identified this documentation error.

Actual ownership-audit order in the pinned successor execution.py:

1. ownership_audit acquires the ownership lock and calls ownership_intents (line 933).
2. ownership_intents calls _validate(spec, False) for each decoded intent (line 392). For output_directory this uses lexical_absolute: absolute string plus os.path.normpath(value) == value (lines 131–133, 165). This does not resolve symlinks or authenticate current filesystem aliases.
3. overlapping_outputs compares lexical pathlib Path components (line 394; helper line 367). It adds no resolve/stat calls. Root, duplicates and ancestor detection concern these lexical components. Generic helper/oracle equivalence does not independently prove absence of filesystem aliases.
4. After structural roster/claim/binding checks, ownership_audit calls validate(spec) for each intent (line 947). validate reruns _validate(spec, False), then _validate(spec, True) (lines 135–138). The live pass uses absolute(), whose Path.resolve equality check is at lines 121–124; it also checks current executable and input pins and live receipt containment. Live failures are collected as historical_input_drift, not universally raised as structural errors or treated as a global unrelated-job launch prohibition.

This describes a successful path through the earlier structural gates; errors can stop before later live checks. Separately, prepare calls live validate(spec) before ownership_intents (lines 427 and 438). Thus validation order is call-path-specific. The new helper preserves the old pairwise comparator's lexical semantics and does not move any live checks.

The symlink unit test directly exercises absolute() rejection and helper/oracle equivalence. It does not establish that ownership_intents resolves symlinks before comparing. Source inspection supplies the ordering evidence above; no tests/model jobs were run for this correction.

All source/evidence pins are in evidence.json. This addendum makes no scale or throughput benchmark claim, no freshness optimization claim, and no deployment/scientific qualification claim. Existing repeated history-wide reads and hashes remain unchanged.
