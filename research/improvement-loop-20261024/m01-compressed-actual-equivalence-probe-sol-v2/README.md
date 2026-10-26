# Prepared actual equivalence probe V2 — unexecuted

V1 remains frozen and unsuitable for actual use because its direct child could bypass controller limits. V2 adds a required live inherited controller channel before model imports, repeated admission, child-side hard limits/wall alarm and parent-death handling. It is not security against hostile same-user code.

Read DESIGN.md, contract.json, tiny-result.json and evidence.json. Twenty-three invented checks passed; no actual graph imported/built/compiled. No approved review/resource receipts or actual attempt exist. Independent review and a fresh per-arm resource/concurrency check are still required.

Run only test_tiny.py for invented tests, under the one-thread nice15 calibration runtime command recorded in evidence. Do not run child.py or launch.py for an actual attempt before the independent gates.
