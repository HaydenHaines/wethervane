# Independent numerical-refinement delta review

2026-10-25. **Accept the sealed v2 implementation delta for a separate Root-authorized calibration attempt. No discrepancy found.** This is not a claim that the new attempt will pass. The original v1 failure and all its gates remain unchanged. No historical optimization, sampler, target forecast or forecast score was executed by this reviewer.

V2 manifest SHA `583ca465e8fea568843486542ad8d85832585661919aae974bcfb5c103e67160` (21 artifacts); byte-delta record SHA `c44b928f27980610e2e9d74cd36d4c85c6b5e4743832dcf3c8e8340b927569dd`; Root v2 design SHA `dede87ca1843d8d98fd569effe3ddc81f84ccbb5111486ea3d1b19508b541f62`. The predecessor manifest SHA remains `fcaccb33a483045e16d179ce5266370520db853f4c997cb3e0e5afcb4c991ef2` with all18 original artifact hashes intact.

Independent `check_delta.py` verifies both manifests and the following byte/contract comparisons, rather than relying on the producer's delta declarations:

- `runner.py`, `test_runner.py`, `runtime.json` and supplied-real-point JSON are byte-identical across versions.
- Normalizing only `DESIGN_SHA` and `BINDINGS_SHA` makes `execution_sources.py` byte-identical; no logic changed.
- Source bindings differ only in the first design reference. The other seven references and their seven snapshots are identical; all current source/snapshot hashes verify.
- The optimizer dictionary differs only at `options.ftol`, **1e-11→1e-14**. All scientific references, objectives, medians/penalties, gates, dimensions, starts, aliases, gtol1e-6, numerical bounds and maxiter300 are unchanged. Additional design differences are version/time/output/contract/predecessor/refinement metadata, not hidden scientific changes.
- Every exact SHA recorded by the frozen v1 terminal review still matches, including the original113 run files and source/receipt pins. The original failure-review receipt `76e224ced1ccd4eec2a6a591a628db86312e8a7023f9e0f01b3aba9dbc7c4a19` and its report/code/results also remain unchanged.

I independently ran the same **11 bounded tests**, all passing in **2.249 seconds**, under the identical lab `.venv` and four single-thread settings with bytecode disabled. Actual-source checks block the historical solver; real solves are synthetic quadratics only. The six-fit orchestration test uses fit stubs. The preserved supplied-point JSON is identical, and the actual supplied-wrapper tests recompute the same accepted objective/gradients. No populated preparation or original attempt was regenerated.

The original failed six-scale job stopped under relative-objective reduction while exceeding the independent fixed best-gradient gate. Tightening the objective stopping tolerance is therefore a justified limited numerical change. Applying it to **all six fits and all eighteen original starts**, with no examined-point warmstarts and no threshold relaxation, preserves the matched computation design. It does not establish convergence or scientific identification in advance; actual v2 terminal outputs must satisfy the unchanged gates and receive separate review. It does not convert training likelihood changes into forecast improvement.

Root must supply a new externally pinned execution receipt, unique run_id, exclusive v2 output directory and actual handle record after launch. The original process/attempt is never restarted. Reviewer validation is implementation evidence, not automatic execution or promotion. Owner/producer and prior immutable files were not edited. `delta-results.json` retains exact independently inspected hashes and comparisons; the original numerical diagnosis remains in its separate immutable v1 result-review packet.
