"""Aggregate-only A2018 ledger with fail-closed eventG likelihood closure.

Survey row tokens are counted/validated in local memory and never serialized.
The caller must supply an output directory; tests use TemporaryDirectory.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT=Path('/home/hayden/projects/wethervane-redesign-20261006')
LAB=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')
ROOT=PROJECT/'integration/research/senate-stack-integration-20261020'
CONTEXT=ROOT/'national-multicycle-execution-astra-v1/A2018/context.json'
PLAN=ROOT/'national-multicycle-execution-astra-v1/A2018/UNADMITTED-plan.json'
PROBLEM=ROOT/'historical-national-2018-assembly-astra-v1/problem.json'
COMPONENTS=ROOT/'national-2018-gpu-preparation-root-v1/gpu-production-v1/attempt1/source-snapshots/117-components.py'
RANK=ROOT/'mixed-count-graph-astra-v1/rank_contribution.py'
OPEN=ROOT/'mixed-count-graph-astra-v1/open_contribution.py'
SCORE=ROOT/'national-2018-gpu-preparation-root-v1/arizona-writein-score-review-v1.json'
V1=LAB/'m01-a2018-observation-ownership-luna-v1'
V1_LEDGER=V1/'ledger.json'
V1_PINS=V1/'source-pins.json'
SOL=LAB/'m01-a2018-observation-ownership-review-sol-v1'
SOL_CLOSURE_PINS=SOL/'closure-source-pins.json'
SOL_REVIEW=SOL/'REVIEW.md'
EXPECTED={
 str(CONTEXT):'4d1f9fa71a67e83405d1f769dd4f4f238ada3046f7396947af29cd0a8b77bcf4',
 str(PLAN):'3e33169c4e1b0f2811d43d3afbb8d8a8f74132b132db866d65ca119567e27003',
 str(PROBLEM):'abcaf347e682422303780c5ff12bb3e6179e4b98362686c01338e29df06a385a',
 str(COMPONENTS):'1574e7831344009f08470b9287b257556a9c031fb7e33e29938a70d51b1a854d',
 str(RANK):'7fbc42874bb6fde95195fe0506db51bb86d2fb2ef418b2fc12592578f1bf2cf7',
 str(OPEN):'444d3b82883e1d9c921046c3bd643096a04b4b0ff950a0e42788f3e124cefdb3',
 str(SCORE):'4c6e66225b3c37c2977c0773ba43cc9009b83822035cc6a371ab6409a660f15b',
 str(V1_LEDGER):'dfce1d1641606a41b26001d9a9ec6db083e69e5aa67a706d00513c0c4905dbf1',
 str(V1_PINS):'5604a63e36eeb2a19f7e32fdb76e9402665b1c662da36c4b6f746fb7b526a848',
 str(SOL_CLOSURE_PINS):'ba970697ef9a9161fe0b56c63c31ace807a391da4ae0d9aaff3a598bcd71983c',
 str(SOL_REVIEW):'8e8a7a264eeb4e2bc9eb0a2e16683c144cd656ef8c8245af24846c9a609da1b1',
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate_model_source(problem, component_text, rank_text, open_text):
    # Conservative exact source-shape checks bound to the frozen selected implementation.
    for needle in (
      'response[name] = pm.Deterministic(f"response_{name}", pm.math.sigmoid(logits[name]))',
      'category = pt.concatenate(',
      'response * v["candidate_probability"][name]',
      'rates[name] = pt.sum(',
      'v["eventG"][c.event_index[contest["event_id"]]] * v["base_category_rates"][name]',
      'means.append(pt.log(rates[first]) - pt.log(rates[second]))',
      'pm.Gamma(\n                "eventG_unobserved", alpha=v["event_gamma_shape"], beta=v["event_gamma_shape"], shape=len(unobserved)',
      'pm.Poisson(', 'pm.NegativeBinomial(', 'pm.MvNormal('):
        if needle not in component_text: raise ValueError('Selected component source no longer matches closure contract: '+needle[:40])
    start=component_text.index('for survey in c.data["surveys"]:')
    end=component_text.index('reports = c.data["participation_reports"]',start)
    survey_loop=component_text[start:end]
    if 'eventG' in survey_loop or 'event_intensity' in survey_loop:
        raise ValueError('Survey likelihood now depends on eventG/event_intensity; prior closure withdrawn')
    if 'for o in ranked["observations"]:' not in rank_text or 'for o in runoff["observations"]:' not in rank_text:
        raise ValueError('Rank/runoff source control flow changed; closure must be re-reviewed')
    if 'for ci, item in enumerate(v["contests"]):' not in open_text or 'for obs in item["observations"]:' not in open_text:
        raise ValueError('Open-person source control flow changed; closure must be re-reviewed')
    if problem['ranked_measurement'].get('poll_channels') != {}:
        raise ValueError('Ranked poll channel exists; ordinary-poll cancellation proof cannot be reused')
    return {'survey_loop_eventG_free':True,'ordinary_poll_formula_verified':True,'contest_response_is_scalar':True}

def validate_closure(context,plan,problem,component_text,rank_text,open_text):
    if context['problem'] != problem: raise ValueError('Context/prepared problem mismatch')
    if context['identity'] != plan['context_identity'] or context['descriptor']['problem_identity'] != plan['problem_identity']:
        raise ValueError('Plan/context identity mismatch')
    ext=context['extension']
    if ext.get('open_person') is not None: raise ValueError('open_person is present; eventG closure not established')
    rr=ext.get('ranked_runoff')
    if not isinstance(rr,dict): raise ValueError('ranked_runoff extension missing')
    ranked=rr.get('ranked'); runoffs=rr.get('runoffs')
    if not isinstance(ranked,list) or not isinstance(runoffs,list): raise ValueError('ranked/runoff observation lists missing')
    if any(not isinstance(x.get('observations'),list) or x['observations'] for x in ranked):
        raise ValueError('ranked observation list is nonempty or malformed')
    if any(not isinstance(x.get('observations'),list) or x['observations'] for x in runoffs):
        raise ValueError('runoff observation list is nonempty or malformed')
    core_ranked=problem['ranked_measurement'].get('ranked')
    channels=problem['ranked_measurement'].get('poll_channels')
    if not isinstance(core_ranked,list) or any(not isinstance(x.get('observations'),list) or x['observations'] for x in core_ranked):
        raise ValueError('core ranked measurement observations nonempty or malformed')
    if channels != {}: raise ValueError('ranked poll channels nonempty or malformed')
    if problem['official_counts'] or problem['participation_reports']:
        raise ValueError('official counts/reports present; eventG prior-only closure withdrawn')
    poll_ids=[r['observation_id'] for b in problem['polls'] for r in b['rows']]
    if len(poll_ids)!=len(set(poll_ids)): raise ValueError('poll row IDs duplicate')
    if any(pid in channels for pid in poll_ids): raise ValueError('ranked poll channel assigned to a poll row')
    source_checks=validate_model_source(problem,component_text,rank_text,open_text)
    return {'open_person_absent':True,'ranked_config_count':len(ranked),'ranked_observation_count':0,
            'runoff_config_count':len(runoffs),'runoff_observation_count':0,
            'core_ranked_config_count':len(core_ranked),'core_ranked_observation_count':0,
            'ranked_poll_channel_count':0,'ordinary_poll_row_count':len(poll_ids),
            'official_count_observation_count':0,'participation_report_count':0,**source_checks}

def load_actual():
    paths=[CONTEXT,PLAN,PROBLEM,COMPONENTS,RANK,OPEN,SCORE,V1_LEDGER,V1_PINS,SOL_CLOSURE_PINS,SOL_REVIEW]
    hashes={str(p):sha(p) for p in paths}
    if hashes!=EXPECTED: raise ValueError('Frozen source/receipt drift; refusing closure claim')
    context=json.loads(CONTEXT.read_text()); plan=json.loads(PLAN.read_text()); problem=json.loads(PROBLEM.read_text())
    # Verify contribution implementations are exactly those declared by pinned A2018 source inventory.
    inventory=context['descriptor']['source_inventory']
    source_rel={COMPONENTS:'research/senate-stack-integration-20261020/national-problem-assembly-astra-v1/components.py',
                RANK:'research/senate-stack-integration-20261020/mixed-count-graph-astra-v1/rank_contribution.py',
                OPEN:'research/senate-stack-integration-20261020/mixed-count-graph-astra-v1/open_contribution.py'}
    for path,rel in source_rel.items():
        if inventory.get(rel)!=hashes[str(path)]: raise ValueError('Context source inventory mismatch for '+rel)
    if plan['source_pins'].get(source_rel[COMPONENTS])!=hashes[str(COMPONENTS)]:
        raise ValueError('Plan does not pin the selected component bytes')
    closure_pins=json.loads(SOL_CLOSURE_PINS.read_text())
    for rel in ('mixed-count-graph-astra-v1/rank_contribution.py','mixed-count-graph-astra-v1/open_contribution.py','national-multicycle-execution-astra-v1/A2018/context.json','national-2018-gpu-preparation-root-v1/gpu-production-v1/attempt1/source-snapshots/117-components.py'):
        if closure_pins.get(rel,{}).get('sha256') != hashes.get(str(PROJECT/'integration/research/senate-stack-integration-20261020'/rel)):
            raise ValueError('Independent review closure source pin mismatch for '+rel)
    return context,plan,problem,hashes

def aggregate_projection(ledger):
    count_keys=['contests','events','factor_membership_count_by_family','factor_membership_rows','official_count_observations','participation_report_observations','poll_blocks','poll_rows','ranked_poll_rows','survey_blocks','survey_record_rows']
    return {k:ledger[k] for k in ('context_identity','problem_admission_id','cutoff_utc','plan_status','selected_composition','parent_masses','sources','events','survey_blocks','poll_blocks','target','factor_families')} | {
        'counts':{k:ledger['counts'][k] for k in count_keys},
        'source_file_sha256':ledger['source_file_sha256']}

def build(output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    context,plan,problem,hashes=load_actual()
    component_text=COMPONENTS.read_text(); rank_text=RANK.read_text(); open_text=OPEN.read_text()
    closure=validate_closure(context,plan,problem,component_text,rank_text,open_text)
    events={e['id']:e for e in problem['events']}; contests={c['id']:c for c in problem['contests']}
    sources={s['id']:s for s in problem['sources']}
    survey_blocks=[]; ids_by_source=defaultdict(list)
    for s in problem['surveys']:
        # Tokens stay local: uniqueness is checked, and no token is serialized or printed.
        local=[r.get('record_id') for r in s['rows']]
        if any(not isinstance(x,str) or not x for x in local): raise ValueError('Survey record token missing')
        ids_by_source[s['source_id']].extend(local)
        survey_blocks.append({'factor_id':f"survey_likelihood_{s['id']}",'survey_block_id':s['id'],'source_id':s['source_id'],'event_id':s['event_id'],'contest_id':s['contest_id'],'record_count':len(local)})
    for source,ids in ids_by_source.items():
        if len(ids)!=len(set(ids)): raise ValueError('Source-scoped survey token repeated; factor ownership needs review')
    for s in survey_blocks: s['record_id_unique_within_source']=True
    poll_rows=[]; poll_blocks=[]
    for block in problem['polls']:
        for row in block['rows']:
            c=contests[row['contest_id']]
            poll_rows.append({'observation_id':row['observation_id'],'source_id':'A2018_polls','event_id':c['event_id'],'contest_id':c['id'],'first_candidate_id':row['first_candidate_id'],'second_candidate_id':row['second_candidate_id']})
        poll_blocks.append({'factor_id':f"poll_likelihood_{block['id']}",'poll_block_id':block['id'],'source_id':'A2018_polls','row_count':len(block['rows']),'covariance_shape':[len(block['covariance']),len(block['covariance'][0]) if block['covariance'] else 0]})
    factor_counts=Counter({'survey_potential':sum(s['record_count'] for s in survey_blocks),'correlated_poll_mvn':len(poll_rows)})
    survey_by_source=defaultdict(lambda:{'blocks':0,'records':0,'contest_specific_blocks':0,'no_contest_blocks':0,'events':set()})
    for s in problem['surveys']:
        a=survey_by_source[s['source_id']];a['blocks']+=1;a['records']+=len(s['rows']);a['events'].add(s['event_id']);a['contest_specific_blocks']+=int(s['contest_id'] is not None);a['no_contest_blocks']+=int(s['contest_id'] is None)
    survey_by_source={k:{**v,'events':sorted(v['events'])} for k,v in survey_by_source.items()}
    poll_by_event=Counter(r['event_id'] for r in poll_rows)
    source_summary=[{'source_id':s['id'],'role':s['role'],'available_utc':s['available_utc'],'source_sha256':s['sha256']} for s in problem['sources']]
    target_id='contest_az2018_regular';target_event='event_az2018_general';target=contests[target_id]
    target_polls=[r for r in poll_rows if r['contest_id']==target_id]
    old=json.loads(V1_LEDGER.read_text())
    base={
      'schema':'A2018ObservationOwnershipLedgerV1AggregateSuccessor',
      'context_identity':context['identity'],'problem_admission_id':problem['admission_id'],'cutoff_utc':problem['cutoff_utc'],
      'plan_status':{k:plan.get(k) for k in ('scope','purpose','national_complete','publication_authorized','forecast_qualified')},
      'selected_composition':plan['provider_contract']['composition']['config'],
      'parent_masses':[{'parent_id':p['parent_id'],'parent_identity':p['parent_identity'],'mass':p['mass']} for p in plan['parents']],
      'counts':{'events':len(events),'contests':len(contests),'official_count_observations':len(problem['official_counts']),'participation_report_observations':len(problem['participation_reports']),'survey_blocks':len(survey_blocks),'survey_record_rows':sum(s['record_count'] for s in survey_blocks),'poll_blocks':len(poll_blocks),'poll_rows':len(poll_rows),'ranked_poll_rows':0,'factor_membership_rows':sum(s['record_count'] for s in survey_blocks)+len(poll_rows),'factor_membership_count_by_family':dict(factor_counts)},
      'factor_families':[
        {'family':'official_count_vector_poisson','factor_count':0,'observation_ids':[],'unit':'candidate/noncandidate ballot-mark count vector; no A2018 rows'},
        {'family':'participation_report_negative_binomial','factor_count':0,'observation_ids':[],'unit':'reported participant count; no A2018 rows'},
        {'family':'survey_potential','factor_count':len(survey_blocks),'record_count':sum(s['record_count'] for s in survey_blocks),'factor_ids':[s['factor_id'] for s in survey_blocks],'source_ids':sorted({s['source_id'] for s in survey_blocks}),'unit':'masked survey record; respondent uniqueness/independence unknown'},
        {'family':'correlated_poll_mvn','factor_count':len(poll_blocks),'row_count':len(poll_rows),'factor_ids':[b['factor_id'] for b in poll_blocks],'source_ids':['A2018_polls'],'unit':'paired-candidate poll logit row within shared covariance block'}],
      'sources':source_summary,
      'events':[{'event_id':e['id'],'state_code':e['state'],'cycle':e['cycle'],'role':e['role']} for e in problem['events']],
      'survey_blocks':survey_blocks,'survey_block_counts_by_source':survey_by_source,
      'poll_blocks':poll_blocks,'poll_row_counts_by_event':dict(sorted(poll_by_event.items())),
      'target':{'contest_id':target_id,'event_id':target_event,'state_code':events[target_event]['state'],'cycle':events[target_event]['cycle'],'candidate_count':len(target['candidates']),'candidate_ids':[c['id'] for c in target['candidates']],'candidate_record_fields':sorted(target['candidates'][0].keys()),'noncandidate_category_ids':[c['id'] for c in target['noncandidate_categories']],'poll_row_count':len(target_polls),'poll_observation_ids':sorted(r['observation_id'] for r in target_polls),'survey_block_count':sum(s['event_id']==target_event for s in problem['surveys']),'official_count_observation_count':sum(c['contest_id']==target_id for c in problem['official_counts']),'participation_report_count':sum(r['event_id']==target_event for r in problem['participation_reports'])},
      'model_prior_families':[
        {'id':'turnout propensity','source_lines':'117-components.py:173-190','summary':'Gaussian intercept/midterm/main and pooled interaction/community/state terms feed a sigmoid cell propensity.'},
        {'id':'event scale','source_lines':'117-components.py:192-255','summary':'kappa ~ LogNormal(log(100),0.7); no official count factors means all eventG values take Gamma(kappa,kappa) prior branch.'},
        {'id':'candidate response','source_lines':'117-components.py:193-206','summary':'one scalar per contest: sigmoid(Normal(logit(.98),.5) baseline plus pooled race effect with scale .3.'},
        {'id':'candidate choice','source_lines':'117-components.py:280-383','summary':'selected IID per-candidate utility and profile softmax; nonmajor candidate effect Normal(-3,1), except sole-other Normal(-.3,1); no role hierarchy selected.'},
        {'id':'noncandidate allocation','source_lines':'117-components.py:392-425','summary':'per-contest Dirichlet(ones) over noncandidate categories when width>1.'},
        {'id':'outer parent mixture','source':'A2018 plan','summary':'eight fixed parent masses .125; not likelihood-updated by this ledger.'}],
      'external_scoring_not_model_factor':{'path':str(SCORE),'sha256':hashes[str(SCORE)],'role':'separate candidate-mark scoring receipt; not included in frozen A2018 problem likelihood'},
      'unknowns_and_overlap':['No official count vectors or participation-report records are present.','Survey record tokens remain local and are not exported; uniqueness is source-scoped only, not proof of unique respondents/independence/cross-source disjointness.','Poll IDs identify rows, not respondent/sample identity; pollster, dates, population, mode and n absent in problem schema.','Poll rows are in one covariance block; no independence claim.','Source role and available time are supplied metadata, not verified vintage or independent lineage.','Candidate metadata source exists, while frozen target candidate fields are id/bloc/incumbent, not ballot_role.'],
      'source_file_sha256':{p:hashes[p] for p in (str(CONTEXT),str(PLAN),str(PROBLEM),str(COMPONENTS),str(SCORE))},
      'ledger_policy':'Aggregate and block identifiers only. Survey row tokens, observed values, masks, poll logits/covariance values, and predictions/draws are not serialized.'}
    old_projection=aggregate_projection(old); new_projection=aggregate_projection(base)
    if old_projection != new_projection: raise ValueError('Frozen aggregate differs from V1; do not claim output unchanged')
    base['predecessor_aggregate_comparison']={'matches_v1_aggregate':True,'predecessor_ledger_sha256':hashes[str(V1_LEDGER)],'projection_sha256':canonical_hash(new_projection),'row_level_survey_tokens_compared':'local count/uniqueness only; no values exported'}
    base['eventG_likelihood_closure']={
      'status':'verified_source_algebra_conditional_on_shared_kappa',
      'claim':'For the exact pinned A2018 problem/source, each of 54 eventG coordinates retains its Gamma(kappa,kappa) conditional prior given kappa and the other parameters; no empirical posterior arrays were read.',
      'closure_checks':closure,
      'event_ids':sorted(events),
      'reasoning':['zero official counts and zero participation reports remove direct total/count factors','open-person absent and ranked/runoff observation arrays empty remove extension count factors','surveys do not reference eventG/event_intensity in the pinned survey likelihood path','all poll rows are non-ranked; ordinary log category-rate differences algebraically cancel common eventG and contest scalar response','shared kappa can induce marginal dependence; this is conditional prior status, not unconditional independence or a sampler result'],
      'invalidated_if':['any pinned source drift','any official count or participation report appears','open_person becomes present','ranked/runoff observations become nonempty','ranked poll channels appear','survey likelihood begins using eventG/event_intensity','ordinary poll formula or scalar-response semantics change']}
    base['closure_source_sha256']={str(p):hashes[str(p)] for p in (COMPONENTS,RANK,OPEN)}
    out_file=out/'aggregate-ledger.json'; out_file.write_text(json.dumps(base,indent=2,sort_keys=True)+'\n')
    return base

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    result=build(args.output_dir)
    print(json.dumps({'status':'PASS','aggregate_output':str(Path(args.output_dir)/'aggregate-ledger.json'),'counts':result['counts'],'closure':result['eventG_likelihood_closure']['status'],'predecessor_aggregate_match':result['predecessor_aggregate_comparison']['matches_v1_aggregate'],'survey_row_tokens_exported':False},indent=2))
if __name__=='__main__': main()
