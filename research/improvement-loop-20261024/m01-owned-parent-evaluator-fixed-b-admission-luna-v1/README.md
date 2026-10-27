# Fixed-B-only phase source admission

`admission-review.json` is a separate source admission for exactly the `fixed-B` phase. It reuses the V3 source review and adds the completed dynamic phase's caller-observed success acknowledgment. It does not include or create a fresh resource receipt; root must supply one for this phase, and the launch gate requires its phase, review digest, contract digest, freshness, headroom and conflict checks.

## Why this phase is adequately bounded

The preserved external `functions.write_stdin` return records exit 0 and `arm=dynamic,status=completed`, with controller-success SHA `649f1e4c…`. That digest matches the current 689-byte `attempt-v1/dynamic/controller-success.json`. The acknowledgment binds result, terminal and comparison artifacts, and the exact returned pins match the on-disk artifacts. The dynamic phase produced 27 calls in the frozen ordered 8-parent × 3-point roster plus parent0 revisit; the independent dynamic audit reconciles all 27 call files and reports parent0's saved baseline/revisit checks pass. This phase is completed but is still only pointwise, unnormalized evidence.

The contract selects original bank ordinal 1 (second record) as anchor before new density values; its parent identity is `c0481866…`. The dynamic phase contains the three matching B-parent calls at points 0, 1, and 2. The fixed-B branch compiles a separate native graph with that exact B record and fixed exposure (no dynamic exposure input), retains the 662-coordinate saved point layout, and checks the protected `population_cut_v1` contract with parent mass `.125` outside the log density. The comparator requires exactly `(1,0),(1,1),(1,2)`, reconstructs full points, verifies parent records and distinct compiler-parent identities, then compares full logp, poll logp, every factor array, rates, physical outputs, and the declared finite-difference calls.

If it passes, it supplies the missing native fixed baseline for parent 1 only. Parent 0 has its saved native baseline; parents 2–7 remain six missing native baselines. No outer mass is updated, and this is not a normalized conditional or posterior calculation.

## Caller acknowledgment and launch boundary

The source checks the caller-supplied hash against the saved dynamic `controller-success.json`, then checks result/terminal pins and clean exit before entering the fixed-B branch. The caller return file separately preserves the actual successful tool return; matching the on-disk marker alone would not prove caller acknowledgment. The fixed-B launch still requires a distinct external fresh resource receipt and source-review digest. No model run is launched by this packet.

## Pins and limits

The admission object pins the exact V3 contract and all nine required V3 sources/tests. The dynamic result/ack/terminal/comparison pins and external caller-return record hash are included in the object. Root handles per-job resources and current staging conflicts. No actual fixed-B calls, sampler, normalized mass, posterior or forecast are claimed. Factor comparisons are positional arrays from the same pinned model source/inventory; the output format does not attach human-readable factor names to each array, a documentation/diagnostic limitation rather than a reason to expand this phase's scope.
