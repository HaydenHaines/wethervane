"""Read current pilot provenance and retain only schemas/counts/identity joins."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
STACK=OWNER/'research/senate-stack-integration-20261020'
PILOT=STACK/'actual-pilot-source-sol-v1'
LAB=HERE.parent

def main():
    paths={
        'training':STACK/'tx-disjoint-calibration-sol-v1/attempt-v1/selected-disjoint-training.csv',
        'target':LAB/'calibration-consumer-v1/inputs/target-coordinates.json',
        'bank_plan':LAB/'calibration-consumer-v1/inputs/plan.json',
        'polls':PILOT/'observation-maps-v2/TX2026-six-LV-poll-original-observations.json',
        'surveys':PILOT/'observation-maps-v2/CES2020-joint-survey-components.json',
        'survey_receipt':PILOT/'observation-maps-v2/CES2020-mapping-receipt.json',
        'survey_source_receipt':STACK/'source-panel-sol-v1/ces2020-behavior-training-v1/receipt.json',
        'official':PILOT/'physical-sources-v2/TX2020-2024-official-candidate-count-observations.json',
        'official_model':PILOT/'observation-maps-v2/official-count-components.json',
        'poll_receipt':PILOT/'observation-maps-v2/TX2026-poll-mapping-receipt.json',
        'calibration_route':STACK/'actual-source-review-astra-v1/disjoint-calibration-route.json',
        'outcomes':OWNER/'research/polling-layer-implementation-20261020/historical-intake-sol/derived/outcomes_allowed_cycles_SEPARATE.csv',
        'owner_coordination':OWNER/'research/coordination-frontend-20261026/m01-count-binding-followup-v2.md',
    }
    data={k:(list(csv.DictReader(p.open())) if p.suffix=='.csv' else json.loads(p.read_text()))
          for k,p in paths.items() if p.suffix in ('.csv','.json')}
    train=data['training']; polls=data['polls']; surveys=data['surveys']; official=data['official']['rows']
    survey_ids=[r['record_id'] for b in surveys for r in b['rows']]
    target=data['target']
    training_events=set(r['race_id'] for r in train)
    official_events=set(r['race_id'] for r in official)
    results=dict(schema='M01CurrentPilotProvenanceInventoryV1',
        scope='Current TX2026 supplied-bank/native-interface candidate only. No claim this is a selected full Problem or historical replay.',
        pins={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},
        calibration=dict(rows=len(train),fields=list(train[0]),event_count=len(training_events),
            metadata_counts={k:dict(collections.Counter(r[k] for r in train)) for k in ('cycle','source_id','sample_overlap_status','publication_date_verified','historical_version_verified','strict_asof_qualified')},
            label_definition=data['bank_plan']['residual'],strict_asof=data['bank_plan']['source_qualifications'],
            outcome_source_fields=list(data['outcomes'][0])),
        current_polls=dict(rows=len(polls),fields=sorted(set().union(*(r.keys() for r in polls))),
            model_map_rows=len(target['source_map']),
            mapping_bijection=(len({r['model_observation_id'] for r in target['source_map']})==6
                and {r['original_observation_id'] for r in target['source_map']}=={r['sample_id'] for r in polls}),
            exact_publication_instants_known=sum(r.get('publication_time_utc') is not None for r in polls),
            cross_survey_overlap_claims=sorted({r['known_overlap'] for r in polls}),
            unknown_metadata_counts=dict(collections.Counter(k for r in polls for k in r['unknown_metadata_fields']))),
        surveys=dict(blocks=len(surveys),records=len(survey_ids),unique_source_record_ids=len(set(survey_ids)),
            block_fields=list(surveys[0]),row_fields=list(surveys[0]['rows'][0]),
            raw_join_key=['year','case_id'], source_id=surveys[0]['source_id'],
            source_receipt_fields=list(data['survey_source_receipt']),
            coverage=data['survey_receipt']['coverage'],
            weights=data['survey_receipt']['weights'],observation_units=data['survey_receipt']['units']),
        official=dict(rows=len(official),fields=sorted(set().union(*(r.keys() for r in official))),
            events=sorted(official_events),role=data['official']['role']),
        joins=dict(calibration_event_vs_current_official=sorted(training_events & official_events),
            calibration_cycle_vs_current_official=sorted({r['cycle'] for r in train}&{str(r['cycle']) for r in official}),
            interpretation='Empty event intersection is a recomputed event-key check only; shared cycle and respondent/source dependence are not removed.'),
        unresolved=['All 1483 calibration rows have unknown author sample identity/field start and unverified publication/vintage.',
            'Current poll sample/release/overlap IDs identify repeated releases, not cross-survey respondent disjointness.',
            'CES year/case_id is unique internally; no cross-source person crosswalk to polls/calibration is established here.',
            'Outcome race keys permit event exclusion; row-level original official result byte/vintage chain is not fully verified by this audit.',
            'Calibration and downstream 2020 events share cycle; a joint shared-error conditional model or separately justified route is required.',
            'No complete selected M01 Problem, fundamentals training-label ledger, or full vector manifest is yet bound.'],
        model_fit=False, owner_tree_mutated=False, empirical_admission=False)
    (HERE/'inventory.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(dict(calibrationRows=len(train),currentPolls=len(polls),surveyRecords=len(survey_ids),
                         calibrationOfficialEventIntersection=sorted(training_events&official_events))))

if __name__=='__main__':
    main()
