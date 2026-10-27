"""Minimal invented reproduction of a support violation in the V2 candidate."""
import json, math
from fractions import Fraction
import sys
sys.path.insert(0, "/home/hayden/projects/wethervane-redesign-20261006/integration/research/coordination-frontend-20261026/participation-simplex-correction-candidate-v1")
from participation_extrapolation_prototype import Cell, Design, Event, Scenario, mixture_moments

weights = [9.555591182889165e-19, 0.2294746878335557,
           0.7556416181139466, 0.014883694052497535]
assert math.fsum(weights) == 0.9999999999999998
assert abs(math.fsum(weights) - 1.0) <= 1e-12
assert all(0.0 <= x <= 1.0 for x in weights)
design = Design(["X"], [2020], "X", 2020)
scenarios = [Scenario(w, b, {"X": 0}, {2020: 0})
             for w, b in zip(weights, [1000, -1000, -1000, -1000])]
result = {}
for n in [1_000_000, 2**53]:
    event = Event("e", "X", 2020, [Cell("c", n, 0)], False)
    actual = mixture_moments(design, [event], scenarios)
    exact_weights = [Fraction.from_float(w) for w in weights]
    mass = sum(exact_weights)
    expected_mean = Fraction(n) * exact_weights[0] / mass
    result[str(n)] = {
        "returned_mean": actual["means"][0],
        "exact_fraction_mean": float(expected_mean),
        "within_physical_support": 0 <= actual["means"][0] <= n,
        "returned_variance": actual["covariance"][0][0],
        "exact_fraction_variance": float(Fraction(n*n) * exact_weights[0] * (mass-exact_weights[0]) / (mass*mass)),
    }
print(json.dumps({"accepted_weights": weights,
                  "math_fsum": math.fsum(weights),
                  "normalized_float_fsum": math.fsum(w/math.fsum(weights) for w in weights),
                  "results": result}, indent=2))
