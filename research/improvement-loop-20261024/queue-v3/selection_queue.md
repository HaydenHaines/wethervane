# Selection recipes for queue V3

`selection_queue.py` in this directory is a separate copy of the original
selection recipe compiler, adapted for new V3 stores. It pins this directory's
compiler and four queue dependencies, and the unchanged parent directory's
selection and helper modules. It invokes the parent
`scoped_selection.py` as the native command. The recipe format and scientific
admission boundaries are unchanged: preparation cannot select a model, and
application requires a separately reviewed decision and the selection module's
native and journal checks. Neither a queue result nor this adapter qualifies or
publishes a forecast.

Follow the recipe and retry contracts in `../selection_queue.md`, substituting
the absolute paths to this directory's `selection_queue.py` and
`experiment_queue.py` for the older scripts. Run tests from this directory with
`python3 -B test_selection_queue.py -v`. Only freshly initialized V3 stores
are supported. A V1 or V2 store cannot be migrated by changing source pins;
preserve the exact old source and records. Existing plans or running jobs are
not discovered or migrated by this adapter. Retry and resume recipes require
the V3 source pins and the queue's separate lineage checks; those pins alone do
not prove who created a supplied plan or provide hostile-history protection.

V3 validates evidence under its documented cooperating local-store contract;
it does not solve arbitrary hostile history changes, scientific model
selection, restartable autonomous research, or scaling to thousands of real
model comparisons. Synthetic test fixtures demonstrate workflow mechanics,
not empirical superiority.
