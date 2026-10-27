# Independent reconciliation of 2018 descriptive controls

The owner's saved control report links to the retained-divergence audit and
compares 76 flagged saved states with 76 distinct unflagged states in the same
chains. Without loading the large raw draw archives or importing the owner's
code, `check.py` reconciles the full roster, audit and owner-code pins, control choices,
reported leading-family counts and medians. It independently reconstructs the
owner's sequential nearest-available matching. `review.json` records the exact
source hashes and results. Two small tests compare the separate minimum-distance
calculation with all assignments in tiny invented lanes.
An independent Sol review at `../national2018-controls-independent-sol-v1/`
also checks the report's remaining plan, inventory and exploration-code pins.

Fifty-one controls are one retained draw from their flagged counterparts; 42
pairs share the same leading coordinate family. The flagged and control median
largest absolute standardized coordinates are about 4.255 and 4.200. These
summaries corroborate that the owner's *reported* diagnostic is descriptive;
they do not show a cause, establish an absence of bias, or validate sampling.
The accepted saved state need not be the trajectory that produced a flag, and
same-chain neighbors are dependent.

The owner's sequential greedy rule is accurately described as selecting the
nearest available state at each step, but need not minimize the total distance
of all pairs. Across the recorded flags its distance sum is 178 draws versus a
same-lane minimum of 156, with three affected chains. A tiny counterexample has
flags 2,3,4 and available states 0,1,5: sequential nearest with the stated
lower-draw tie break costs 7, while a jointly optimal assignment costs 5. This
is a property of the stated matching algorithm, not evidence of an error in
the owner's analysis. A minimum-total-distance assignment is not automatically
a better diagnostic and was not used to recalculate model-coordinate values.

This review authenticates and reconciles small JSON records, not raw retained
NPZ data or actual sampler execution. It does not calculate R-hat or ESS, replay
the divergent trajectories, test the proposed NUTS remedy, alter owner files,
or qualify a forecast. The original zero-divergence gate remains unmet.

Reproduce from this directory with `python3 -m unittest -v test_check` and
`python3 check.py`. The results depend on the pinned owner evidence at its
original paths; no historical EAVS CSV is involved.
