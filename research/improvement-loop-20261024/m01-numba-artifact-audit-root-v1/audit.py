"""Cross-check existing receipts; no recompile, sampling or scientific qualification."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
NEW = BASE / 'm01-native-numba-feasibility-sol-v1'
OLD = BASE / 'm01-native-successor-provider-sol-v1'

def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

previous, previous_sha = read(OLD / 'result.json')
current, current_sha = read(NEW / 'full-result.json')
terminal, terminal_sha = read(NEW / 'terminal.json')
launch, launch_sha = read(NEW / 'launch.json')
evidence, evidence_sha = read(OLD / 'evidence.json')
assert previous_sha == evidence['files']['result.json']['sha256']
for name, digest in launch['source_sha256'].items():
    assert hashlib.sha256((NEW / name).read_bytes()).hexdigest() == digest
assert terminal['returncode'] == 0 and terminal['termination_reason'] is None
assert terminal['pid'] == launch['pid'] == current['pid']
assert terminal['full_result_exists'] and terminal['no_retries']
assert current['gradient_shape'] == [662] and current['finite_coordinates'] == 662
assert current['fastmath'] is False and current['cxx'] == ''
assert abs(current['logp'] - previous['full_joint_fixed_state']['multi_logp']) < 1e-6
old_checks = {entry['variable']: entry for entry in previous['full_joint_fixed_state']['finite_difference_checks']}
new_checks = {entry['value_variable']: entry for entry in current['finite_difference_checks']}
comparisons = {}
for name in ('choice_intercept', 'turnout_intercept'):
    prior_fd = old_checks[name]['central_differences'][0]
    this_fd = new_checks[name]['numeric']
    analytic = new_checks[name]['analytic']
    assert abs(this_fd - prior_fd) < .01
    assert abs(analytic - prior_fd) < .01
    comparisons[name] = {'prior_central_difference': prior_fd, 'new_central_difference': this_fd,
                         'new_analytic_gradient': analytic, 'analytic_vs_prior_absolute_difference': abs(analytic-prior_fd)}
assert len(new_checks) == 6
assert all(entry['absolute_error'] < entry['tolerance'] for entry in new_checks.values())
output = {
    'schema': 'M01NumbaArtifactAuditV1',
    'status': 'recorded_receipts_consistent',
    'source_sha256': {'previous_result': previous_sha, 'previous_evidence': evidence_sha, 'new_result': current_sha, 'launch': launch_sha, 'terminal': terminal_sha},
    'prior_vs_new_logp_absolute_difference': abs(current['logp'] - previous['full_joint_fixed_state']['multi_logp']),
    'independently_saved_nonzero_coordinate_comparisons': comparisons,
    'peak_observed_rss_bytes': terminal['peak_observed_rss_bytes'],
    'limits': [
        'Audit cross-checks existing recorded runs; it does not replay a full graph or independently produce a gradient.',
        'The exact 662-vector and initial point were not persisted; only six derivative values are recorded, three approximately zero.',
        'This is one initial state and first protected parent only; normalized conditional densities, other parents, sampler behavior and scientific performance remain unverified.',
        'Three object-mode fallback warnings and failed fusion warnings mean no native-only or throughput claim follows.'
    ]
}
(HERE / 'result.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
