# 2022 Senate parent-0 CPU parity: receipt consistency review

The completed corrected receipt reports five passing original parent-0 points. Each has 817 ordered checks: log density, full gradient and 815 named physical outputs, for 4,075 physical-output comparisons. All reported checks have zero failures and zero nonfinite values; the largest reported tolerance fraction is about 0.0071. The owner service terminated successfully. The receipt records no GPU use, no sampler transitions, no fit admission and no forecast qualification.

This small read-only checker pins the receipt, rechecks the six receipt-bound metadata/source files and the two source pins in the boundary proof, verifies the saved runtime-source copy, confirms the native reference's named-output roster and matching plan identity, and checks every saved pass record and point order. It does not open original points, reference arrays, posterior draws or execute a model. It does **not independently recompute numerical differences** or establish that every upstream reference and environment was scientifically valid; it checks the saved evidence's internal consistency and byte bindings.

The all-parent artifact checked here is explicitly `PASS_all_parent_structure_only`; it is not all-parent numerical parity. This result is one complete parent on CPU, not GPU parity, sampler speed, convergence, a fitted model, or forecast admission. The prior failed CPU receipt and ROCm device-argument failure remain separate preserved evidence.

Run `python3 -B check.py` from this directory to reproduce the aggregate receipt check. No model jobs or owner files were changed.
