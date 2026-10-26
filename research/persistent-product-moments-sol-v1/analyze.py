"""Analytical prior sensitivity only; standard library, no inference or owner imports."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OWNER = Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/turnout-refinement-20261026/nevada-persistent-event-provider-root-v1')

def moments(s, k):
    a = math.exp(s*s)
    variance = math.expm1(s*s) + a/k
    covariance = math.expm1(s*s)
    return {'mean': 1.0, 'variance': variance, 'cross_event_covariance': covariance,
            'cross_event_correlation': covariance/variance if variance else None}

def matching_shape(s, target_variance):
    """None denotes no finite positive gamma shape; equality is deterministic U."""
    d = target_variance - math.expm1(s*s)
    if d < 0:
        return {'status': 'impossible', 'shape': None}
    if d == 0:
        return {'status': 'deterministic_U_boundary', 'shape': None}
    return {'status': 'finite_gamma', 'shape': math.exp(s*s)/d}

checks = []
def check(name, condition):
    assert condition, name
    checks.append(name)

check('s_zero_reduces_to_gamma_variance', math.isclose(moments(0, 4)['variance'], .25))
check('s_zero_cross_event_covariance_zero', moments(0, 4)['cross_event_covariance'] == 0)
check('s_zero_matching_shape_recovers_incumbent', math.isclose(matching_shape(0, .25)['shape'], 4))
check('deterministic_U_leaves_B_variance', math.isclose(moments(.35, math.inf)['variance'], math.expm1(.35**2)))
check('deterministic_U_perfect_event_correlation', moments(.35, math.inf)['cross_event_correlation'] == 1)
check('zero_s_deterministic_U_zero_variance', moments(0, math.inf)['variance'] == 0)
check('equality_boundary', matching_shape(.35, math.expm1(.35**2))['status'] == 'deterministic_U_boundary')
check('infeasible_fixed_example', matching_shape(.35, 1/32.2575)['status'] == 'impossible')
km = matching_shape(.1, 1/32.2575)['shape']
check('feasible_example_recovers_target_variance', math.isclose(moments(.1, km)['variance'], 1/32.2575))
# Non-normal persistent factor B=0 or 2 with equal probability, U=0 or 2 independently.
# Direct exact enumeration checks the general mean-one product and event covariance identities.
rows = [(b*u, b*v) for b in (0, 2) for u in (0, 2) for v in (0, 2)]
mean = sum(x for x,y in rows)/8
var = sum((x-mean)**2 for x,y in rows)/8
cov = sum((x-mean)*(y-mean) for x,y in rows)/8
check('enumerated_general_product_variance', var == (1+1)*(1+1)-1)
check('enumerated_general_cross_event_covariance', cov == 1)

k = 32.2575
result = {'status': 'analytical_prior_sensitivity_not_a_fit_or_recommendation',
          'illustrative_fixed_k': k, 'incumbent_variance': 1/k,
          'max_feasible_log_sd': math.sqrt(math.log1p(1/k)),
          'sensitivities': [{'log_sd':s, **moments(s,k), 'matched_variance_gamma': matching_shape(s,1/k)} for s in (0,.05,.1,.15,.175,.2,.35)],
          'checks_passed': checks}
(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
inputs = [OWNER/'provider.py', OWNER/'README.md', OWNER/'manifest.json']
artifacts = [ROOT/'analyze.py', ROOT/'README.md', ROOT/'results.json']
manifest = {'owner_sources_read_only':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
            'lab_outputs':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts]}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'checks_passed':len(checks),'fixed_k':k,'max_log_sd':result['max_feasible_log_sd'],'s_035':moments(.35,k)},indent=2))
