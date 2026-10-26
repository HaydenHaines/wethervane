#!/usr/bin/env python3
"""Publish separately reviewed fixed32 F/FP expectations, without deriving draws or using results."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SOURCE = Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
BASE = Path('research/senate-polling-refresh-20261026')
HANDOFF = BASE / 'fixed32-scorecard-frontend-handoff-root-v1.json'
HANDOFF_SHA = 'e571037c39a1203b79c2027ec72a4e2e7ff3b3340c1648eba39f75b6a7c41fbb'
REVIEW_SHA = 'b82bc61967c3af40274178d79f65ba088e3f5313d32233ec4fbc0c9ac3648c8d'
FREEZE_SHA = 'a90c5c30a2db502a71ff56b98c5c2fc7966f9b1675539fc30b479d9bca18f3d3'
POLICY_SHA = 'efe1bf193e2959284b73ea833af0a136f6c592001a2faf50505e3258e25cf0cd'
MAPPING_SHA = '9024069ac0f397c2d379663d653d384e32532548ef26772035eda61ce55265e0'
COHORT_SHA = 'a11a92c18ad01867b5e68a9ac5f56a3398012abddacd980207098f09323e4ed1'
FAILED_V1_SHA = 'd73ed9b406ba9c1184bca648eaf830a9f8c96ab7f6809e5165d52591554d2550'
EXCLUDED = ['2026:AK:senate:regular', '2026:GA:senate:regular', '2026:ME:senate:regular']
ARMS = {
 'F': {'runId':'current-fundamentals','referenceSha256':'e8945138896bad678ee4d4dee5ab182e92ca150766deb47595cee6fc4825b120',
       'resultSha256':'37af627c934f519f32b2cf77a326f5658758a24153ee574fdc1a0ba46bfd51c1','forecastSha256':'c02bdc63a0df27824578c07d6e2b752b831ae4bbd858cabc432ee00fbbf94136',
       'jointDrawsSha256':'3244418fd97e377e72dd013e6e3bad3c25948bd80228b3a97953e4599b88a6a8','shape':[100000],'rowCount':100000,
       'issuedAt':'2026-10-20T15:25:53.623020+00:00','cutoffAt':'2026-10-20T15:21:23Z','targetId':'national_senate_aggregate_baseline_v1'},
 'FP': {'runId':'current-polling','referenceSha256':'545ff5c25cc74e681afd10305b7c5a00c0653368cf3a6adda76184b9dd142568',
        'resultSha256':'905ce512c44104d3e48900745cec0f430c9e3954736614468e1a8b728b6da5f0','forecastSha256':'a1ffba60d937bfc81d06df51af6ed917a7ddc71fd9860c1d702de7c7c5db67cb',
        'jointDrawsSha256':'a7240d654861078c7850e3d0c10c1f287604035de531abf14d82821aa4d5cc31','shape':[4,10000],'rowCount':40000,
        'issuedAt':'2026-10-26T15:05:11.698781+00:00','cutoffAt':'2026-10-26T15:03:57.212405+00:00','targetId':'national_senate_aggregate_poll_v1'},
}
class CoreReferenceExportError(ValueError): pass

def require(value: Any, message: str) -> None:
    if not value: raise CoreReferenceExportError(message)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pinned(root: Path, relative: str | Path, expected: str, length: int | None = None) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source path')
    full = (root / path).resolve()
    require(full.is_relative_to(root.resolve()) and full.is_file(), 'Missing or escaped source path')
    require(sha(full) == expected and (length is None or full.stat().st_size == length), f'Source bytes changed: {relative}')
    return full

def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'Expected JSON object')
    return value

def near(a: float, b: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-10

def build(source_root: Path, repo_root: Path) -> dict[str, Any]:
    root = source_root.resolve()
    require(root == SOURCE.resolve(), 'Unreviewed source root')
    handoff = read(pinned(root, HANDOFF, HANDOFF_SHA))
    require(handoff.get('schema') == 'Fixed32ScorecardReferencesFrontendHandoffV1' and handoff.get('status') == 'independently_reviewed_ready_for_owner_fixed32_integration'
            and handoff.get('evaluation_lock_or_publication_issued') is False and handoff.get('new_forecast_or_outcomes') is False, 'Unqualified handoff')
    for key, rel, digest in [('review', BASE/'fixed32-scorecard-review-sol-v1/review-v1.json', REVIEW_SHA), ('review_freeze', BASE/'fixed32-scorecard-review-sol-v1/freeze-v1.json', FREEZE_SHA)]:
        require(handoff[key] == {'path':str(root/rel),'sha256':digest}, 'Changed review binding')
        pinned(root, rel, digest)
    review = read(root/BASE/'fixed32-scorecard-review-sol-v1/review-v1.json')
    freeze = read(root/BASE/'fixed32-scorecard-review-sol-v1/freeze-v1.json')
    require(review.get('status') == 'passed_bounded_independent_recount' and review.get('blocking_issues') == [] and review.get('checks_passed') == 285
            and freeze.get('status') == 'passed', 'Review does not qualify these references')
    for item in review['source_pins'] + freeze['files']:
        pinned(root,item['path'],item['sha256'],item.get('bytes'))
    require(any(row['sha256'] == FAILED_V1_SHA for row in review['source_pins']), 'Failed v1 evidence must remain pinned')
    policy = read(pinned(root, BASE/'result-policy-special-rules-sol-v1/policy-v1.json', POLICY_SHA))
    mapping = read(pinned(root, BASE/'scorecard-baseline-compatibility-sol/candidate-pair-event-mapping-v1.json', MAPPING_SHA))
    core = policy['frontend_binding']['fixed_cohorts']['plurality_core_32']
    full = policy['frontend_binding']['fixed_cohorts']['full_35_locked_rows']
    require(len(core) == len(set(core)) == 32 and [r for r in full if r not in core] == EXCLUDED and len(full) == 35, 'Changed fixed32 scope')
    indices = [i for i,race in enumerate(full) if race in core]
    entries = []
    for arm, expected in ARMS.items():
        relative = BASE/f'fixed32-scorecard-reference-root-v2/{arm}/reference.json'
        require(handoff['references'][arm] == {'path':str(root/relative),'sha256':expected['referenceSha256']}, 'Only qualified v2 references are supported')
        ref = read(pinned(root,relative,expected['referenceSha256']))
        require(ref['schema'] == 'SenateFixedCohortScorecardReferenceV1' and ref['arm'] == arm and ref['native_target_id'] == expected['targetId']
                and ref['edition'] == expected['cutoffAt'][:10] and ref['input_cutoff_utc'] == expected['cutoffAt'] and ref['cohort_size'] == 32
                and ref['common_target_cohort_sha256'] == COHORT_SHA and ref['excluded_from_this_reference'] == EXCLUDED, 'Reference edition/cohort changed')
        require(hashlib.sha256(json.dumps(ref['cohort'],sort_keys=True,separators=(',',':')).encode()).hexdigest() == COHORT_SHA, 'Cohort content changed')
        require([r['race_id'] for r in ref['cohort']] == core == [r['race_id'] for r in ref['per_race']], 'Ordered cohort changed')
        for item in ref['sources'] + [ref['joint_rows']]: pinned(root,item['path'],item['sha256'])
        sources = {Path(row['path']).name:row['sha256'] for row in ref['sources']}
        require(sources['forecast.json'] == expected['forecastSha256'] and sources['joint-draws.npz'] == expected['jointDrawsSha256'], 'Native bank changed')
        draw = ref['draw_identity']
        require(draw == {'shape':expected['shape'],'weight_per_row':1/expected['rowCount'],
                'order':'unchanged original row order and native chain axes, selected fixed race columns only','original_race_axis_indices':indices,
                'shared_across_selected_races':True,'resampling':False,'cross_bank_pairing':False}, 'Native draw identity changed')
        require(ref['distributions'] == review['arms'][arm]['summaries'], 'Reviewed summaries changed')
        public_path = repo_root/f'public/data/explorer/{expected["runId"]}.json'
        require(sha(public_path) == expected['resultSha256'], 'Public forecast bytes changed')
        public = read(public_path); run = public['run']
        require(run['id'] == expected['runId'] and run['targetId'] == expected['targetId'] and run['lawId'] == 'senate_principal_pair_aggregate_v1'
                and run['issuedAt'] == expected['issuedAt'] and run['cutoffAt'] == expected['cutoffAt'] and run['historical'] is False
                and public['provenance']['sources']['forecastSha256'] == expected['forecastSha256'], 'Public/native edition mismatch')
        misses = 0.; mae = 0.
        for identity, row, index in zip(ref['cohort'],ref['per_race'],indices,strict=True):
            race = public['races'][index]; first,second = race['outcomes']; share = first['share']; old = mapping['races'][index]
            require(identity['race_id'] == race['id'] == old['race_id'] and identity['seat_id'] == race['seatId'] and identity['state'] == race['state']
                    and identity['first_candidate']['id'] == first['id'] == old['first_candidate']['id']
                    and identity['second_candidate']['id'] == second['id'] == old['second_candidate']['id']
                    and identity['denominator'] == 'sum_of_named_principal_candidate_votes_under_declared_rule_proxy', 'Exact candidate pair identity changed')
            require(near(row['point_margin_mean_pp'],200*share['mean']-100) and all(near(a,b) for a,b in zip(row['interval90_margin_pp'],[200*share['quantiles']['p05']-100,200*share['quantiles']['p95']-100],strict=True))
                    and row['first_pair_win_probability'] == first['winProbability'] and row['modal_pair_winner_id'] == (first['id'] if first['winProbability'] >= .5 else second['id']), 'Saved marginal mismatch')
            misses += 1-max(first['winProbability'],second['winProbability']); mae += row['expected_absolute_margin_error_pp']/32
        require(near(misses,ref['distributions']['modal_pair_winner_miss_count']['mean']) and near(mae,ref['distributions']['mean_absolute_margin_error_pp']['mean']), 'Marginal expectation identity failed')
        entries.append({'chamber':'senate','runId':expected['runId'],'resultSha256':expected['resultSha256'],'targetId':expected['targetId'],
            'lawId':run['lawId'],'forecastIssuedAt':run['issuedAt'],'forecastCutoffAt':run['cutoffAt'],'evaluationLock':False,'observedResultsConsumed':False,
            'sourcePins':{'handoffSha256':HANDOFF_SHA,'referenceSha256':expected['referenceSha256'],'reviewSha256':REVIEW_SHA,'reviewFreezeSha256':FREEZE_SHA,
                'policySha256':POLICY_SHA,'mappingSha256':MAPPING_SHA,'forecastSha256':expected['forecastSha256'],'jointDrawsSha256':expected['jointDrawsSha256'],
                'jointRowsSha256':ref['joint_rows']['sha256'],'receiptSha256':sources['receipt.json'],'failedV1EvidenceSha256':FAILED_V1_SHA},'reference':ref})
    require(entries[0]['reference']['cohort'] == entries[1]['reference']['cohort'], 'F/FP target cohorts differ')
    return {'schemaVersion':1,'entries':entries}

def export(source_root: Path, repo_root: Path, output: Path, publication: Path) -> dict[str, Any]:
    catalog = build(source_root,repo_root)
    data = (json.dumps(catalog,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    manifest = {'schemaVersion':1,'href':'/data/election-night/core-references.json','sha256':hashlib.sha256(data).hexdigest(),
                'editions':[{k:e[k] for k in ['chamber','runId','resultSha256']} for e in catalog['entries']]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_bytes(data)
    publication.parent.mkdir(parents=True,exist_ok=True); publication.write_text(json.dumps(manifest,indent=2)+'\n')
    return catalog

if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,default=SOURCE);p.add_argument('--repo-root',type=Path,default=repo)
    p.add_argument('--output',type=Path,default=repo/'public/data/election-night/core-references.json');p.add_argument('--publication',type=Path,default=repo/'data/election-night-core-reference-publication.json')
    a=p.parse_args();export(a.source_root,a.repo_root,a.output,a.publication)
