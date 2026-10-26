import hashlib, json
from pathlib import Path

LAB = Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')
EXEC = LAB / 'gasoline-execution-v1/attempt-v1'
REVIEW = LAB / 'gasoline-results-review-luna-v1'
OUT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(path.read_text())

cycles = []
pins = {}
for year in (2018, 2020, 2022, 2024):
    f0_path = EXEC / str(year) / 'F0/scores.json'
    g_path = EXEC / str(year) / 'G01/scores.json'
    review_name = 'independent-score-review-2018-r2.json' if year == 2018 else f'independent-score-review-{year}.json'
    review_path = REVIEW / review_name
    f0, g, report = load(f0_path), load(g_path), load(review_path)
    pins.update({str(p): digest(p) for p in (f0_path, g_path, review_path)})
    assert f0['numerical_passed'] and g['numerical_passed']
    assert not f0['forecast_qualified'] and not g['forecast_qualified']
    frows = {r['race_id']: r for r in f0['rows']}
    grows = {r['race_id']: r for r in g['rows']}
    assert len(frows) == len(f0['rows']) and len(grows) == len(g['rows'])
    assert set(frows) == set(grows), (year, set(frows) ^ set(grows))
    deltas = [grows[rid]['share_crps'] - frows[rid]['share_crps'] for rid in sorted(frows)]
    n = len(deltas)
    mean = sum(deltas) / n
    sorted_d = sorted(deltas)
    median = (sorted_d[n//2] if n % 2 else (sorted_d[n//2-1] + sorted_d[n//2]) / 2)
    ranked = sorted(range(n), key=lambda i: (-abs(deltas[i]), sorted(frows)[i]))
    top_three = ranked[:3]
    remaining = [deltas[i] for i in ranked[3:]]
    top_three_signed_sum = sum(deltas[i] for i in top_three)
    remaining_signed_sum = sum(remaining)
    abs_sorted = sorted((abs(d) for d in deltas), reverse=True)
    loo = [(sum(deltas) - d)/(n-1) for d in deltas]
    reviewed = next(x for x in report['per_cycle'] if x['target'] == year)
    expected = reviewed['arms']['G01']['strata']['all']['share_crps'] - reviewed['arms']['F0']['strata']['all']['share_crps']
    assert abs(mean - expected) <= 2e-10, (year, mean, expected)
    cycles.append({
        'year': year, 'tier': reviewed['tier'], 'n_matched_all_races': n,
        'paired_race_delta_definition': 'G01 share_crps minus F0 share_crps; negative favors lower G01 loss',
        'improved_race_count': sum(d < 0 for d in deltas),
        'worsened_race_count': sum(d > 0 for d in deltas),
        'exact_tie_count': sum(d == 0 for d in deltas),
        'mean_delta': mean, 'reviewed_cycle_delta': expected,
        'median_delta': median, 'minimum_delta': min(deltas), 'maximum_delta': max(deltas),
        'top_three_absolute_deltas_share_of_absolute_total': sum(abs_sorted[:3]) / sum(abs(d) for d in deltas),
        'top_three_largest_absolute_races_signed_delta_sum': top_three_signed_sum,
        'remaining_races_signed_delta_sum': remaining_signed_sum,
        'remaining_races_mean_delta': remaining_signed_sum / len(remaining),
        'leave_one_race_out_mean_range': [min(loo), max(loo)],
    })

result = {
    'schema': 'MatchedGasolinePerRaceCRPSBreakdownV2',
    'status': 'bounded_descriptive_reanalysis_of_exposed_retrospective_folds',
    'primary_metric': 'share_crps',
    'pair': 'original G01 gasoline challenger minus F0 incumbent, no arm reselection',
    'race_weights': 'one equal weight per originally scored race within each cycle; no row dropped',
    'cycles': cycles,
    'cycle_direction_count': {
        'cycles_with_lower_G01_mean_crps': sum(x['mean_delta'] < 0 for x in cycles),
        'cycles_with_higher_G01_mean_crps': sum(x['mean_delta'] > 0 for x in cycles),
        'equal_cycle_mean_delta': sum(x['mean_delta'] for x in cycles) / len(cycles),
    },
    'matched_inputs_sha256': pins,
    'independent_audit_seal_sha256': digest(REVIEW / 'final-review-seal.json'),
    'qualification': False,
    'interpretation_limits': [
        'The four folds are exposed retrospective development comparisons, not prospective validation.',
        'Races within an election share national context and model/source errors; race counts are not independent experiments.',
        'No fit, sampler, raw prediction bank, resampling, significance test, or forecast-qualified result was created.',
        'Per-race concentration and leave-one-race-out ranges are descriptive influence checks only; they do not isolate causality or solve cycle dependence.',
        'This breakdown does not rank the eight model roles or show which layer is empirically weakest.'
    ]
}
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
