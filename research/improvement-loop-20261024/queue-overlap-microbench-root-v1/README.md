# Isolated path-overlap microbenchmark

This is a small timing of the exact old `overlap` function in `queue-finite-owned-sol-v1` against the new `overlapping_outputs` function in `queue-finite-ancestor-sol-v1`. It uses only nonoverlapping, invented path strings; it does not create jobs, open those paths, run the queue or evaluate a forecast. The independently reviewed property and regression tests cover correctness separately.

On this shared host, one warm Python process on CPU 2, with nice 19, compared 50, 150 and 300 paths three times each. At 300 paths, the old 44,850 pair checks had a median of 1.3214 seconds and the new ancestor-set check had a median of 0.00681 seconds. Full samples and exact source hashes are in `result.json`. Maximum process RSS was about 25.4 MiB. The old check always ran before the new one; these are not randomized or isolated measurements.

Command, from the improvement-lab root:

```sh
PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=2s 15s nice -n 19 taskset -c 2 python3 -B research/improvement-loop-20261024/queue-overlap-microbench-root-v1/microbench.py
```

The result establishes only the relative cost of this isolated no-overlap case under these conditions. It does not measure file verification, lock contention, queue admission, real processes, end-to-end throughput, or thousands of improvement cycles. The successor still repeatedly hashes historical inputs. A separate safeguarded real-queue pilot remains unadmitted and is not implied by these timings.
