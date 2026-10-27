"""Counterexample for common-utility-offset cancellation in frozen v2."""
import math, sys, json
sys.path.insert(0, str(__import__("pathlib").Path.cwd()))
from check import ordinary_poll_logit

weights = (1.0, 9.0)
pairs = ((4.0, -4.0), (-4.0, 4.0))
other = 0.0
response, event_g = 0.72, 1.5
offset = 1e16
base = ordinary_poll_logit(weights, pairs, other, response, event_g)
shifted_pairs = tuple(tuple(value + offset for value in row) for row in pairs)
shifted_other = other + offset
shifted = ordinary_poll_logit(weights, shifted_pairs, shifted_other, response, event_g)

def independently_gauge_centered(ws, ps, o):
    log_d = []
    log_r = []
    for weight, (d, r) in zip(ws, ps, strict=True):
        center = max(d, r, o)
        # Compute differences before adding/logging a large common gauge.
        normalizer = math.log(sum(math.exp(v - center) for v in (d, r, o)))
        log_d.append(math.log(weight) + (d - center) - normalizer)
        log_r.append(math.log(weight) + (r - center) - normalizer)
    def lse(values):
        m = max(values)
        return m + math.log(sum(math.exp(x - m) for x in values))
    return lse(log_d) - lse(log_r)

reference_base = independently_gauge_centered(weights, pairs, other)
reference_shifted = independently_gauge_centered(weights, shifted_pairs, shifted_other)
print(json.dumps({
    "weights": weights,
    "base_pairs": pairs,
    "base_other": other,
    "common_offset": offset,
    "shifted_pairs": shifted_pairs,
    "shifted_other": shifted_other,
    "v2_base": base,
    "v2_shifted": shifted,
    "v2_shift_error": shifted - base,
    "independent_centered_base": reference_base,
    "independent_centered_shifted": reference_shifted,
    "reference_invariance_error": reference_shifted - reference_base,
}, indent=2))
