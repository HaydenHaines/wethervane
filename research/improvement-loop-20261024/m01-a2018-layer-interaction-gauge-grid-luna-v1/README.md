# Standalone A2018 v3 gauge-grid probe

This small invented-fixture probe reruns 200 utility-shift cases at each of `1e16` and `1e20`. It verifies that every utility difference survives the offset exactly, then checks that the frozen v3 ordinary-poll logit is invariant. It reads no raw poll/survey rows, model arrays, or owner outputs.

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B probe.py`. The script verifies the exact SHA-256 of the frozen v3 `check.py` before loading its pure arithmetic function. The test is a numeric regression check, not evidence of fitted/native parameter support or forecast qualification.
