# Row-level membership file

The generated `factor-membership.csv` contains source-scoped survey record tokens and stays in the local workspace. It is excluded from the repository rather than publishing row-level identifiers. The aggregate ledger, builder, tests, source pins, and the generated file's SHA-256 receipt are retained; authorized local readers can regenerate the file from the pinned prepared inputs and verify it against `evidence.json`.

The tokens do not establish unique people or independent samples. Keeping the file local does not change the model, the original prepared inputs, or the interpretation of the aggregate ledger.
