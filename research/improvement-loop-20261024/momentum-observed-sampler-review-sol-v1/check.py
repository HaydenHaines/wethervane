"""Independent sealed-byte and synthetic dense-law review; no target inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
P = HERE.parent / 'momentum-observed-sampler-sol-v1'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'MANIFEST.json') == '5e8608c4d1de978e8119a8cd1b4dded46ffe6253c4ff973f53e20d2bf98402c2'
m = json.loads((P/'MANIFEST.json').read_text())
for f in m['files']:
    assert sha(P/f['path']) == f['sha256']
    assert (P/f['path']).stat().st_size == f['bytes']
# Reconstruct private instrumentation without permitting any producer write.
built = []
scope = {'__file__': str(P/'build_observed.py')}
with patch.object(Path, 'write_text', lambda self, text: built.append((self, text))):
    exec(compile((P/'build_observed.py').read_text(), str(P/'build_observed.py'), 'exec'), scope)
assert len(built) == 1 and built[0][1].encode() == (P/'observed_sampler.py').read_bytes()
spec = importlib.util.spec_from_file_location('review_fixture', P/'test_observed.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
import numpy as np
from scipy.special import logsumexp
prior, polls = t.fixture(t.observed, rows=8, q=2, n=3)
records = []
args = dict(chains=2, warmup=4, draws=13, seed=927345)
original_prior, _ = t.fixture(t.original, rows=8, q=2, n=3)
a, ac, ast = t.tracked_run(t.original, original_prior, polls, **args)
b, bc, bst = t.tracked_run(t.observed, prior, polls, observer=records.append, **args)
t.assert_equal(a, b); t.assert_equal(ac, bc); t.assert_equal(ast, bst)
obs = np.unique(polls.race_indices)
H = (polls.race_indices[:, None] == obs).astype(float)
P0 = H.T @ np.linalg.solve(polls.covariance, H)
h = H.T @ np.linalg.solve(polls.covariance, polls.values)
max_mean = max_var = max_weight = 0.0
for r in records:
    if r['phase'] == 'gaussian_conditional':
        k = r['conditioning_row_index']; lam = r['conditioning_lambda']
        # Build a joint Gaussian by explicit linear measurement design.
        A = np.column_stack((np.eye(len(obs)), -np.ones(len(obs))))
        v = prior.tau_race[k]**2 / lam
        J = A.T @ np.diag(1/v) @ A
        J[:-1, :-1] += P0
        J[-1, -1] += 1/prior.tau_cycle[k]**2
        mu = prior.offset[obs] + prior.design[obs] @ prior.beta[k]
        rhs = A.T @ (mu/v); rhs[:-1] += h
        max_mean = max(max_mean, float(np.max(abs(np.linalg.solve(J, rhs)-r['mean_theta_cycle']))))
        max_var = max(max_var, float(np.max(abs(np.diag(np.linalg.inv(J))-r['marginal_variances']))))
    elif r['phase'] == 'row_weights':
        # Product of normalized joint normals, including row-varying scales.
        from scipy.stats import norm
        means = prior.offset[obs][None,:] + prior.beta @ prior.design[obs].T + r['cycle']
        sd = prior.tau_race[:,None]/np.sqrt(r['conditioning_lambda'])
        lw = norm.logpdf(r['cycle'], scale=prior.tau_cycle) + norm.logpdf(r['theta'], means, sd).sum(axis=1)
        max_weight = max(max_weight, float(np.max(abs(np.exp(lw-logsumexp(lw))-r['normalized_row_weights']))))
assert max_mean < 1e-13 and max_var < 1e-13 and max_weight < 1e-13
result = dict(schema='ObservedSamplerIndependentSyntheticCheckV1', manifest_files=len(m['files']), reconstructed_private_source_exact=True, synthetic_only=True, actual_target_inputs=False, chains=2, warmup=4, draws=13, rows=8, basis=2, max_mean_error=max_mean, max_variance_error=max_var, max_probability_error=max_weight, exact_outputs_rng_calls_and_final_states=True)
(HERE/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
