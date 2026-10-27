# Independent review: queue overlap microbenchmark

The benchmark imports the exact pinned old and new execution modules and calls only their overlap helper(s). Its inputs are disjoint absolute sibling paths at one fixed depth; the old nested pairwise path therefore performs all n(n−1)/2 comparisons, while the new ancestor-set helper processes each path's components. The assertion checks both return false.

I reran the exact harness. Its medians reproduced the same increasing function-level speedup pattern at 50, 150 and 300 paths. The original result records affinity CPU 2; my rerun ran with CPUs 0–11 allowed. These short, isolated timings do not establish queue throughput or a thousands-cycle service capacity.

The new path-set implementation's semantic equivalence is covered separately by five inherited differential/property tests, including prefix-order permutations, roots, duplicates, lexical path cases, symlink validation and randomized comparisons. The microbenchmark itself does not time conflicts or the rest of ownership audit.

One evidence limitation: original result.json records source module hashes but not the microbench.py hash, command, or run timestamp. This review pins the harness and preserves a separate rerun. Neither the benchmark nor this review ran a queue or model, changed owner files, or measured storage/dispatch/crash behavior.

See review.json and source-pins.json for exact scope and hashes.
