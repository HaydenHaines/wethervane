"""All-six owned call verification; every parent is required before batch success."""
import json
from pathlib import Path
from common import *
import evaluator as ev
import baseline_compare as base
from selection import authenticate_predecessors
ROSTER=list(range(2,8))

def exact_roster(values):need(values==ROSTER and all(type(i)is int for i in values),'exact ordered six-parent roster required')
def compare_outputs(a,b):
 need(a.keys()==b.keys(),'output roster differs');errors={k:base.close(a[k],b[k])for k in ['logp','poll_logp']};need(len(a['factor_values'])==len(b['factor_values']),'factor roster differs');errors['factors']=max(base.close(x,y)for x,y in zip(a['factor_values'],b['factor_values']));need(a['rates'].keys()==b['rates'].keys()and a['physical'].keys()==b['physical'].keys(),'rate/physical roster differs');errors['rates']=max(base.close(a['rates'][k],b['rates'][k])for k in a['rates']);errors['physical']=max(base.close(a['physical'][k],b['physical'][k])for k in a['physical']);return errors

def verify_all(out,c,load_call,saved,expected_points,records,kernel):
 """Shared verifier used by actual receipt loader and invented test loader."""
 need(out['schema']=='SixParentDynamicWitnessResultV1'and out['status']=='all_six_owned_calls_persisted_not_yet_compared'and out['phase']=='dynamic-six'and out['contract_sha256']==EXPECTED_CONTRACT,'batch result identity differs');exact_roster(out['parent_roster']);exact_roster([row['ordinal']for row in out['parents']]);reports=[]
 need(kernel['phase']=='dynamic-six'and kernel['parent_roster']==ROSTER and kernel['contract_sha256']==EXPECTED_CONTRACT,'batch kernel identity differs')
 target=kernel['compiler_parent_contract'];record=records[0];need(target['composition_identity']==c['reference_composition_identity']and target['problem_identity']==c['problem_identity']and target['parent_identity']==digest(record)and target['parent_id']==record['parent_id']and target['upstream_identity']==record['upstream_identity']and target['parent_mass']==.125 and target['parent_mass_used_in_logp']is False and target['inference_target']=='population_cut_v1'and target['scientific_target_id']==digest({'composition_id':c['reference_composition_identity'],'problem':c['problem_identity'],'parent':digest(record),'target':'population_cut_v1'}),'full batch compiler-parent target differs')
 for row in out['parents']:
  ordinal=row['ordinal'];parent=load_call(row);need(parent['schema']=='PartialBatchParentCallsV1'and parent['status']=='numerical_calls_persisted_not_batch_success'and parent['ordinal']==ordinal and parent['parent_identity']==digest(records[ordinal])and parent['conditional_normalization']=='unverified','parent index identity differs');need(all(type(x['parent_index'])is int and type(x['point_index'])is int for x in parent['calls']),'integer base roster required');need([(x['parent_index'],x['point_index'])for x in parent['calls']]==[(ordinal,i)for i in range(3)],'exact selected base point roster required');errors=[]
  def check(item,point):
   call=load_call(item);need(call['parent_record']==records[ordinal]and call['parent_envelope']['parent_identity']==digest(records[ordinal])and call['parent_envelope']['mass']==.125 and call['outer_mass']==.125,'wrong whole parent or mass');need(call['provenance']==kernel and call['conditional_normalization']=='unverified','stale source or normalized claim');need(canonical(call['point'])==canonical(point)and call['point_sha256']==digest(point),'wrong complete point/FD input');return call
  for i,item in enumerate(parent['calls']):
   call=check(item,expected_points[i]);reference=saved[(ordinal,i)];need(reference['parent_record']==call['parent_record']and reference['point']==call['point'],'saved/new parent/point pairing differs');errors.append(compare_outputs(reference['output'],call['output']))
  fd=parent['finite_differences'];need([x['variable']for x in fd]==c['finite_difference_variables'],'exact FD variable roster required')
  for entry in fd:
   need(entry['coordinate']==0 and entry['step']==c['fd_step']and [x['sign']for x in entry['calls']]==[-1,1],'exact signed FD roster/step required')
   for item in entry['calls']:
    call=check(item,base.declared_fd_point(expected_points,c,entry['variable'],item['sign']));need(call['output']['logp']==item['value'],'FD value differs from owned call')
   need(entry['derivative']==(entry['calls'][1]['value']-entry['calls'][0]['value'])/(2*c['fd_step']),'FD arithmetic mismatch')
  reports.append({'ordinal':ordinal,'parent_identity':digest(records[ordinal]),'errors':errors,'FD_variables':c['finite_difference_variables'],'calls':7,'conditional_normalization':'unverified'})
 return {'schema':'AllSixDynamicWitnessComparisonV1','status':'all_six_three_saved_points_and_exact_owned_FD_calls_verified','parent_roster':ROSTER,'parents':reports,'call_count':42,'outer_masses':[.125]*8,'normalizers_posteriors':'unverified','scope':'dynamic witness only; no remaining fixed-native comparison inferred'}

def compare_batch(root,c):
 root=Path(root);authenticate_predecessors(c);out=root/'dynamic-six';r=json.loads((out/'result.json').read_text())
 need(set(r['artifacts'])=={'entry.json','kernel.json','model-envelope.json','parent-index.json','replacement-structure.json'},'exact completed batch artifact roster required')
 for name,expected in r['artifacts'].items():need(pin(out/name)==expected,'batch artifact changed')
 need(json.loads((out/'parent-index.json').read_text())==r['parents'],'parent index ledger differs')
 bindings=base.binding(c);records=[row['record']for row in bindings.load()['parents']];kernel=json.loads((out/'kernel.json').read_text());saved_calls,_=base.read_phase(c['baseline_attempt_root'],'dynamic',c);saved={(row['parent_index'],row['point_index']):call for row,call in saved_calls};points=base.expected_points(c)
 def load(item):
  path=out/item['file'];need(pin(path)==item['pin'],'batch parent/call bytes changed');raw=path.read_bytes()
  return json.loads(raw)if item['file'].endswith('-index.json')else ev.verify(raw,item['pin']['sha256'],bindings)
 result=verify_all(r,c,load,saved,points,records,kernel);atomic_json(root/'all-six-comparison.json',result);return result
