"""Exact-rational protected-parent-mass counterexample; invented toy only."""
from fractions import Fraction as F
import json

# Two parent conditionals begin with protected masses. The observed datum has
# different evidence under each parent, while the normalized Gamma nuisance
# block is independent and integrates to one in both conditionals.
MASS = (F(1, 8), F(7, 8))
EVIDENCE = (F(1, 4), F(3, 4))
NUISANCE_INTEGRAL = (F(1), F(1))

cut = tuple(w * z / z for w, z in zip(MASS, EVIDENCE))
assert cut == MASS
uncut_normalizer = sum(w * z for w, z in zip(MASS, EVIDENCE))
uncut = tuple(w * z / uncut_normalizer for w, z in zip(MASS, EVIDENCE))
assert uncut == (F(1, 22), F(21, 22))

result = {
    "schema": "LunaProtectedParentMassCounterexampleV1",
    "status": "passed",
    "parent_masses_before": [str(x) for x in MASS],
    "parent_specific_observation_evidence": [str(x) for x in EVIDENCE],
    "normalized_nuisance_block_integral_within_each_parent": [str(x) for x in NUISANCE_INTEGRAL],
    "protected_cut_masses_after_conditional_update": [str(x) for x in cut],
    "ordinary_bayesian_evidence_reweighted_masses_for_contrast": [str(x) for x in uncut],
    "interpretation": "The normalized nuisance integral leaves each parent-specific conditional likelihood unchanged. The cut keeps original masses even when parent evidence differs; ordinary model averaging would update them. This catches a behavior the source toy's equal-evidence parent fixture cannot distinguish.",
    "scope": "Exact rational invented counterexample; not actual model or data."
}
print(json.dumps(result, indent=2))
