"""Build an identifier-only A2018 observation/factor ownership ledger."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
BASE=Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020')
CONTEXT=BASE/'national-multicycle-execution-astra-v1/A2018/context.json'
PLAN=BASE/'national-multicycle-execution-astra-v1/A2018/UNADMITTED-plan.json'
PROBLEM=BASE/'historical-national-2018-assembly-astra-v1/problem.json'
COMPONENTS=BASE/'national-2018-gpu-preparation-root-v1/gpu-production-v1/attempt1/source-snapshots/117-components.py'
SCORE=BASE/'national-2018-gpu-preparation-root-v1/arizona-writein-score-review-v1.json'
OUT=Path(__file__).resolve().parent
EXPECTED={str(CONTEXT):'4d1f9fa71a67e83405d1f769dd4f4f238ada3046f7396947af29cd0a8b77bcf4',str(PLAN):'3e33169c4e1b0f2811d43d3afbb8d8a8f74132b132db866d65ca119567e27003',str(PROBLEM):'abcaf347e682422303780c5ff12bb3e6179e4b98362686c01338e29df06a385a',str(COMPONENTS):'1574e7831344009f08470b9287b257556a9c031fb7e33e29938a70d51b1a854d',str(SCORE):'4c6e66225b3c37c2977c0773ba43cc9009b83822035cc6a371ab6409a660f15b'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load_inputs():
    paths=[CONTEXT,PLAN,PROBLEM,COMPONENTS,SCORE]; hashes={str(p):sha(p) for p in paths}
    if hashes!=EXPECTED: raise ValueError('Pinned source drift: '+repr(hashes))
    context=json.loads(CONTEXT.read_text()); plan=json.loads(PLAN.read_text()); problem=json.loads(PROBLEM.read_text())
    if context['problem']!=problem: raise ValueError('Context and prepared problem differ')
    if context['identity']!=plan['context_identity'] or context['descriptor']['problem_identity']!=plan['problem_identity']:
        raise ValueError('Plan/context/problem identity mismatch')
    return context,plan,problem,hashes
def build():
    context,plan,problem,hashes=load_inputs()
    events={e['id']:e for e in problem['events']}; contests={c['id']:c for c in problem['contests']}
    membership=[]; survey_summary=[]; record_ids=defaultdict(list)
    for s in problem['surveys']:
        fid=f"survey_likelihood_{s['id']}"; src=s['source_id']
        for row in s['rows']:
            token=row['record_id']; record_ids[src].append(token)
            membership.append({'factor_id':fid,'factor_family':'survey_potential','observation_token':token,'source_id':src,'event_id':s['event_id'],'contest_id':s['contest_id'] or '', 'unit_as_encoded':'survey_record_id; person-level meaning/independence not proven','quantity_role':'community/turnout/answer masked survey contribution','joint_group':s['id']})
        survey_summary.append({'factor_id':fid,'survey_block_id':s['id'],'source_id':src,'event_id':s['event_id'],'contest_id':s['contest_id'],'record_count':len(s['rows'])})
    for src,ids in record_ids.items():
        if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate source-scoped survey record token in {src}')
    for s in survey_summary: s['record_id_unique_within_source']=True
    poll_summary=[]; poll_blocks=[]
    for block in problem['polls']:
        for row in block['rows']:
            c=contests[row['contest_id']]
            membership.append({'factor_id':f"poll_likelihood_{block['id']}",'factor_family':'correlated_poll_mvn','observation_token':row['observation_id'],'source_id':'A2018_polls','event_id':c['event_id'],'contest_id':c['id'],'unit_as_encoded':'paired-candidate poll logit row; respondent/sample unit unknown','quantity_role':'paired-candidate logit; shared covariance block','joint_group':block['id'],'first_candidate_id':row['first_candidate_id'],'second_candidate_id':row['second_candidate_id']})
            poll_summary.append({'observation_id':row['observation_id'],'source_id':'A2018_polls','event_id':c['event_id'],'contest_id':c['id'],'first_candidate_id':row['first_candidate_id'],'second_candidate_id':row['second_candidate_id']})
        poll_blocks.append({'factor_id':f"poll_likelihood_{block['id']}",'poll_block_id':block['id'],'source_id':'A2018_polls','row_count':len(block['rows']),'covariance_shape':[len(block['covariance']),len(block['covariance'][0]) if block['covariance'] else 0]})
    factor_counts=Counter(x['factor_family'] for x in membership)
    survey_by_source=defaultdict(lambda:{'blocks':0,'records':0,'contest_specific_blocks':0,'no_contest_blocks':0,'events':set()})
    for s in problem['surveys']:
        a=survey_by_source[s['source_id']]; a['blocks']+=1; a['records']+=len(s['rows']); a['events'].add(s['event_id'])
        a['contest_specific_blocks']+=int(s['contest_id'] is not None); a['no_contest_blocks']+=int(s['contest_id'] is None)
    survey_source_summary={k:{**v,'events':sorted(v['events'])} for k,v in survey_by_source.items()}
    poll_by_event=Counter(r['event_id'] for r in poll_summary)
    ranked_poll_ids=set(problem['ranked_measurement'].get('poll_channels',{}))
    sources=[{'source_id':s['id'],'role':s['role'],'available_utc':s['available_utc'],'source_sha256':s['sha256']} for s in problem['sources']]
    target_id='contest_az2018_regular'; target_event='event_az2018_general'; target=contests[target_id]
    target_polls=[r for r in poll_summary if r['contest_id']==target_id]
    ledger={
      'schema':'A2018ObservationOwnershipLedgerV1','context_identity':context['identity'],'problem_admission_id':problem['admission_id'],'cutoff_utc':problem['cutoff_utc'],
      'plan_status':{k:plan.get(k) for k in ('scope','purpose','national_complete','publication_authorized','forecast_qualified')},
      'selected_composition':plan['provider_contract']['composition']['config'],
      'parent_masses':[{'parent_id':p['parent_id'],'parent_identity':p['parent_identity'],'mass':p['mass']} for p in plan['parents']],
      'counts':{'events':len(events),'contests':len(contests),'official_count_observations':len(problem['official_counts']),'participation_report_observations':len(problem['participation_reports']),'survey_blocks':len(problem['surveys']),'survey_record_rows':sum(len(s['rows']) for s in problem['surveys']),'poll_blocks':len(problem['polls']),'poll_rows':sum(len(b['rows']) for b in problem['polls']),'ranked_poll_rows':sum(r['observation_id'] in ranked_poll_ids for b in problem['polls'] for r in b['rows']),'factor_membership_rows':len(membership),'factor_membership_count_by_family':dict(factor_counts)},
      'factor_families':[
        {'family':'official_count_vector_poisson','factor_count':0,'observation_ids':[],'unit':'candidate/noncandidate ballot-mark count vector; no A2018 rows'},
        {'family':'participation_report_negative_binomial','factor_count':0,'observation_ids':[],'unit':'reported participant count; no A2018 rows'},
        {'family':'survey_potential','factor_count':len(survey_summary),'record_count':sum(s['record_count'] for s in survey_summary),'factor_ids':[s['factor_id'] for s in survey_summary],'source_ids':sorted({s['source_id'] for s in survey_summary}),'unit':'masked survey record; respondent uniqueness/independence unknown'},
        {'family':'correlated_poll_mvn','factor_count':len(poll_blocks),'row_count':sum(b['row_count'] for b in poll_blocks),'factor_ids':[b['factor_id'] for b in poll_blocks],'source_ids':['A2018_polls'],'unit':'paired-candidate poll logit row within shared covariance block'}],
      'sources':sources,'events':[{'event_id':e['id'],'state_code':e['state'],'cycle':e['cycle'],'role':e['role']} for e in problem['events']],
      'survey_blocks':survey_summary,'survey_block_counts_by_source':survey_source_summary,'poll_blocks':poll_blocks,'poll_row_counts_by_event':dict(sorted(poll_by_event.items())),
      'poll_likelihood_semantics':{'ranked_poll_channel_count':len(ranked_poll_ids),'ranked_rows_in_A2018_block':sum(r['observation_id'] in ranked_poll_ids for b in problem['polls'] for r in b['rows']),'ordinary_paired_logit_rows':sum(r['observation_id'] not in ranked_poll_ids for b in problem['polls'] for r in b['rows']),'ordinary_branch_model_mean':'log(category_rate[first])-log(category_rate[second]); common eventG and scalar response cancel algebraically before covariance likelihood'},
      'target':{'contest_id':target_id,'event_id':target_event,'state_code':events[target_event]['state'],'cycle':events[target_event]['cycle'],'candidate_count':len(target['candidates']),'candidate_ids':[c['id'] for c in target['candidates']],'candidate_record_fields':sorted(target['candidates'][0].keys()),'noncandidate_category_ids':[c['id'] for c in target['noncandidate_categories']],'poll_row_count':len(target_polls),'poll_observation_ids':sorted(r['observation_id'] for r in target_polls),'survey_block_count':sum(s['event_id']==target_event for s in problem['surveys']),'official_count_observation_count':sum(c['contest_id']==target_id for c in problem['official_counts']),'participation_report_count':sum(r['event_id']==target_event for r in problem['participation_reports'])},
      'model_prior_families':[
        {'id':'turnout propensity','source_lines':'117-components.py:173-190','summary':'Gaussian intercept/midterm/main and pooled interaction/community/state terms feed a sigmoid cell propensity.'},
        {'id':'event scale','source_lines':'117-components.py:192-255','summary':'kappa ~ LogNormal(log(100),0.7); with zero official-count observations all 54 eventG values use Gamma(kappa,kappa) priors.'},
        {'id':'candidate response','source_lines':'117-components.py:193-206','summary':'one scalar per contest: sigmoid(Normal(logit(.98),.5) baseline + pooled race effect with scale .3).'},
        {'id':'candidate choice','source_lines':'117-components.py:280-383','summary':'selected IID candidate utilities and per-profile softmax; nonmajor candidate effect Normal(-3,1), except sole-other Normal(-.3,1); no role hierarchy selected.'},
        {'id':'noncandidate allocation','source_lines':'117-components.py:392-425','summary':'per-contest Dirichlet(ones) over noncandidate categories when width>1.'},
        {'id':'outer parent mixture','source':'A2018 plan','summary':'eight fixed parent masses .125; not a likelihood factor and not reweighted by observation evidence in this ledger.'}],
      'cancellation_and_identification':{'ordinary_poll_model_mean':'log(category_rates[first])-log(category_rates[second])','event_scale_cancels':True,'contest_scalar_response_cancels':True,'why_turnout_not_identified_by_ratio':'common event scale and scalar response cancel; cell-profile intensity still changes aggregation weights, so indirect compositional effects remain possible','no_target_participation_observation':True},
      'external_scoring_not_model_factor':{'path':str(SCORE),'sha256':hashes[str(SCORE)],'role':'separate candidate-mark scoring receipt; not included in frozen A2018 problem likelihood'},
      'unknowns_and_overlap':['No official count vectors or participation-report records are present.','Survey record_id is an opaque source-scoped key; uniqueness does not establish unique persons, independence, or cross-source disjointness.','Poll observation IDs identify rows, not respondents/samples; pollster, field dates, sample IDs, population, mode and sample size are absent from this problem schema.','Poll rows form a single 274x274 covariance block; do not call them independent. Exact overlap with survey respondents is unknown.','Source role/hash metadata alone does not establish observation independence or lineage.','Candidate metadata source exists, but the frozen candidate records expose id/bloc/incumbent, not modeled ballot_role.'],
      'source_file_sha256':hashes,'ledger_policy':'Identifier, factor-membership, source-role and aggregate metadata only; no observed values, masks, logits, covariance entries or predictions/draws copied.'}
    (OUT/'ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
    fields=['factor_id','factor_family','observation_token','source_id','event_id','contest_id','unit_as_encoded','quantity_role','joint_group','first_candidate_id','second_candidate_id']
    with (OUT/'factor-membership.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(membership)
    return ledger
if __name__=='__main__':
    x=build();print(json.dumps({'status':'PASS','counts':x['counts'],'target':x['target'],'outputs':['ledger.json','factor-membership.csv']},indent=2))
