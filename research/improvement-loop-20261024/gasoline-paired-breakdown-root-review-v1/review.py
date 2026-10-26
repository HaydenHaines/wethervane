"""Independent arithmetic check of an exposed, previously scored comparison."""
import hashlib
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPORT = BASE / 'gasoline-paired-breakdown-luna-v1/result.json'
SOURCE = BASE / 'gasoline-paired-breakdown-luna-v1/analyze.py'
result = json.loads(REPORT.read_bytes())
assert result['qualification'] is False
checks = []
for cycle in result['cycles']:
    year = cycle['year']
    source = BASE / 'gasoline-execution-v1/attempt-v1' / str(year)
    arms = [json.loads((source / arm / 'scores.json').read_bytes()) for arm in ('F0', 'G01')]
    maps = [{r['race_id']: r for r in a['rows']} for a in arms]
    assert all(len(m) == len(a['rows']) for m, a in zip(maps, arms))
    assert maps[0].keys() == maps[1].keys()
    delta = [maps[1][k]['share_crps'] - maps[0][k]['share_crps'] for k in sorted(maps[0])]
    n = len(delta)
    assert n == cycle['n_matched_all_races']
    mean = statistics.fmean(delta)
    close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)
    assert close(mean, cycle['mean_delta'])
    assert close(statistics.median(delta), cycle['median_delta'])
    assert sum(x < 0 for x in delta) == cycle['improved_race_count']
    assert sum(x > 0 for x in delta) == cycle['worsened_race_count']
    assert sum(x == 0 for x in delta) == cycle['exact_tie_count']
    magnitude = sorted(map(abs, delta), reverse=True)
    assert close(sum(magnitude[:3]) / sum(magnitude), cycle['top_three_absolute_deltas_share_of_absolute_total'])
    ranked = sorted(delta, key=abs, reverse=True)
    assert close(sum(ranked[:3]), cycle['top_three_largest_absolute_races_signed_delta_sum'])
    assert close(sum(ranked[3:]), cycle['remaining_races_signed_delta_sum'])
    assert close(statistics.fmean(ranked[3:]), cycle['remaining_races_mean_delta'])
    leave_one = [statistics.fmean(delta[:i] + delta[i+1:]) for i in range(n)]
    assert all(close(a,b) for a,b in zip((min(leave_one),max(leave_one)),cycle['leave_one_race_out_mean_range']))
    checks.append({'year': year, 'n': n, 'mean': mean})
for filename, expected in result['matched_inputs_sha256'].items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected
assert math.isclose(statistics.fmean(c['mean'] for c in checks),
                    result['cycle_direction_count']['equal_cycle_mean_delta'], abs_tol=1e-14)
review = {
    'schema': 'GasolinePairedBreakdownRootIndependentReviewV1',
    'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'result_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
    'status': 'passed_descriptive_arithmetic_only',
    'checks': checks,
    'limits': 'No raw forecast bank, model fit, causal identification, prospective accuracy or eight-layer ranking was evaluated. Races within cycles share errors.',
}
(Path(__file__).resolve().parent / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review, indent=2))
