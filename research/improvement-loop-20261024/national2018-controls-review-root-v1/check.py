"""Independently reconcile the owner's saved descriptive matched-control report.

This checks the retained JSONs, not the underlying NPZ values or model inference.
It intentionally does not import or run the owner's implementation.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import statistics

ROOT = Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1')
AUDIT = ROOT / 'gpu-production-v1/retained-divergence-audit-v1/result.json'
CONTROLS = ROOT / 'gpu-production-v1/divergence-matched-controls-v1/result.json'
SOURCE = ROOT / 'compare_divergent_controls.py'


def read(path):
    data = path.read_bytes()
    return sha256(data).hexdigest(), json.loads(data)


def minimum_total_distance(flags, candidates):
    """Small exact monotone assignment DP for equal-count distinct controls."""
    flags = sorted(flags)
    costs = [0] + [float('inf')] * len(flags)
    for candidate in sorted(candidates):
        for j in range(len(flags), 0, -1):
            costs[j] = min(costs[j], costs[j - 1] + abs(flags[j - 1] - candidate))
    return costs[-1]


def main():
    audit_hash, audit = read(AUDIT)
    control_hash, report = read(CONTROLS)
    assert report['source_pins']['audit'] == audit_hash
    assert report['source_pins']['code'] == sha256(SOURCE.read_bytes()).hexdigest()
    flags = {(row['parent'], row['chain'], row['draw']) for row in audit['flagged_transitions']}
    pairs = report['pairs']
    assert len(flags) == audit['retained_divergences'] == len(pairs) == 76
    assert report['flagged_rows'] == report['matched_control_rows'] == 76
    assert {(p['parent'], p['chain'], p['draw']) for p in pairs} == flags
    assert len({(p['parent'], p['chain'], p['control_draw']) for p in pairs}) == 76

    # Reconstruct the exact deterministic sequential matching policy from flags.
    available = {}
    for parent, chain, draw in flags:
        available.setdefault((parent, chain), set()).add(draw)
    expected = []
    for (parent, chain), lane_flags in sorted(available.items()):
        unused = set(range(1000)) - lane_flags
        for draw in sorted(lane_flags):
            # Explicit distance expansion rather than the owner's min(key=...).
            for distance in range(1, 1001):
                candidates = [x for x in (draw - distance, draw + distance) if x in unused]
                if candidates:
                    selected = min(candidates)
                    break
            else:
                raise AssertionError('no unflagged control')
            unused.remove(selected)
            expected.append((parent, chain, draw, selected))
    assert expected == [(p['parent'], p['chain'], p['draw'], p['control_draw']) for p in pairs]

    # For one-dimensional absolute distance, an optimum has sorted (noncrossing)
    # assignments. Dynamic programming computes its total without changing the
    # owner's valid, preexisting greedy matches or accessing underlying draws.
    comparison = []
    for (parent, chain), lane_flags in sorted(available.items()):
        greedy = sum(abs(p['draw'] - p['control_draw']) for p in pairs
                     if (p['parent'], p['chain']) == (parent, chain))
        optimum = minimum_total_distance(lane_flags, set(range(1000)) - lane_flags)
        assert greedy >= optimum
        if greedy > optimum:
            comparison.append({'parent': parent, 'chain': chain,
                               'greedy_distance_sum': greedy, 'minimum_distance_sum': optimum})
    for key in ('flagged', 'control'):
        families = Counter(p[key]['leading_coordinate']['name'] for p in pairs)
        assert dict(families) == report[f'{key}_leading_coordinate_families']
        median = statistics.median(p[key]['maximum_absolute_z'] for p in pairs)
        assert abs(median - report[f'median_{key}_max_abs_standardized']) < 1e-12

    distances = Counter(abs(p['draw'] - p['control_draw']) for p in pairs)
    output = {
        'schema': 'National2018ControlsIndependentReconciliationV1',
        'audit_sha256': audit_hash, 'control_report_sha256': control_hash,
        'owner_code_sha256': sha256(SOURCE.read_bytes()).hexdigest(),
        'flags_reconciled': len(flags), 'unique_controls': len(pairs),
        'distance_counts': dict(sorted(distances.items())),
        'greedy_distance_sum': sum(k * v for k, v in distances.items()),
        'same_lane_minimum_distance_sum': sum(k * v for k, v in distances.items())
            - sum(x['greedy_distance_sum'] - x['minimum_distance_sum'] for x in comparison),
        'lanes_with_longer_greedy_total': comparison,
        'same_leading_family_pairs': sum(p['flagged']['leading_coordinate']['name'] == p['control']['leading_coordinate']['name'] for p in pairs),
        'flagged_median': report['median_flagged_max_abs_standardized'],
        'control_median': report['median_control_max_abs_standardized'],
        'verified': ['audit linkage and count', 'complete exact matching roster', 'unique same-chain controls', 'greedy nearest-control policy', 'reported family counts and medians'],
        'not_verified': ['underlying NPZ values or hashes', 'posterior mixing', 'divergence mechanism or bias', 'forecast qualification'],
        'limitation': 'Adjacent same-chain states are dependent; the accepted state is not the failed trajectory. A minimum total-distance match is not automatically more scientifically informative. These exploratory summaries do not establish cause or sampler validity.'
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
