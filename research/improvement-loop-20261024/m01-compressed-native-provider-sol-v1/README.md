# Prepared compressed M01 successor

Read FINDINGS.md, selection.json and result.json. This packet implements and tests a source-pinned measurement replacement only in tiny supplied-rate graphs. Actual full TX graph construction remains gated on independent review and a fresh resource check. No sampling or forecast.

provider.py pins selection.json externally in source; selection pins original provider/native mapping and frozen compression artifacts. Both original modes remain supported. check.py has hard process resource limits and requires the existing calibration-v1 environment with one-thread environment variables and no-C PyTensor. evidence.json pins final files. First-attempt evidence is preserved.
