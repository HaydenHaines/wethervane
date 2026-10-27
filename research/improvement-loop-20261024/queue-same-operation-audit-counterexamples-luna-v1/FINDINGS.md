# Independent review: queue same-operation audit counterexamples

I validated all 15 entries in the frozen source-pins manifest against their live source bytes, then copied the ancestor queue Python modules and frozen test/measurement scripts into a fresh `scratch/` layout. The four focused tests passed there in 2.297 seconds with four fresh tiny accepted histories. No model or owner job was run.

The mutations are meaningful counterexamples to lock-only audit reuse: with dispatcher `D` held, the test can alter external executable bytes while preserving size and mtime, replace a job-lock inode with same contents, or corrupt a claim; a new full audit notices the changed condition. A cooperating status client blocks on `D`. This does not prove every actor follows the lock; it shows the lock is not an input or filesystem immutability boundary.

I separately reran the one-tiny-job/two-audit instrumentation. It reproduced 46 `execution.decode` calls across 7 exact inputs (39 repeats), and 58 `execution.pin` calls. Byte totals differed from the frozen author result (48,124 vs 35,520 decode input bytes; 16,474,902 vs 16,464,820 pin-reported bytes). The evidence does not explain that variation, so I treat only the call-count pattern as reproduced. Neither result times decoding or demonstrates a speedup. A successful-decode memo remains a possible optimization experiment only if full reads, hashes, lock/inode checks, live validation, errors, and mutation behavior are retained and differential-tested.

The original unintended test-discovery timeout log remains pinned and unchanged. The corrected explicit selection was independently rerun; I did not rerun the inherited full suite or watchdog case. Fixture teardown ran during the four tests. The original packet’s process scan is only a sampled post-run observation, not proof of complete descendant accounting.

Machine-readable review and pins: `independent-review.json`. Rerun output: `reproduction-tests.log`, `reproduction-call-counts.json`, `reproduction-call-detail.json`. Scratch source layout is local to this packet. Frozen source packet remains unchanged.
