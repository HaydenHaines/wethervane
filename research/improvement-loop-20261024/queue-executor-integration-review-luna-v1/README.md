# Independent review: queue/executor integration design V1

I reran the packet's source/hash/AST checker and independently confirmed its nine input pins. The two principal compatibility gaps are real source-level concerns:

1. The executor's global ownership scan validates every historical intent with the full runnable-spec validator, which rechecks mutable input, executable and working-directory state. Queue V2 has an explicit regression contract allowing an unrelated unchanged job to run after another prepared job's input drifts. A drop-in replacement would likely make that old drift block unrelated preparation. Separate immutable ownership-shape auditing from current runnable-input validation.
2. Queue V2's released-result path authenticates queue result/release/terminal/output artifacts but never consults the new executor's ownership intent or claim files. Final evidence likewise omits them. Add a read-only ownership audit on released results and before final commit, bound to the effective queue spec/reservation, and pin its findings.

The proposed lock order is plausible from these source paths: dispatcher D → ownership O → nonblocking job lock J during preparation; O/J close before queue launch; final audit would release O before SQLite S. It is not a liveness proof: blocking O while holding D can stall the queue, and direct clients in the managed execution root bypass queue capacity accounting. A dedicated root plus enforced launch entry boundary is required for that claim.

No implementation or runtime tests were performed. The full source pins, exact call-site evidence, compatibility requirements and limits are in `review.json`.
