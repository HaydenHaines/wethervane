"""One remaining parent; saved dynamic baseline plus new explicit FD witness."""
import json
from pathlib import Path
from common import *
import evaluator as ev
import baseline_compare as base
from selection import selected,authenticate_predecessors

def target_identity(record,c):
 return digest({'composition_id':c['reference_composition_identity'],'problem':c['problem_identity'],'parent':digest(record),'target':'population_cut_v1'})
def validate_fixed_target(contract,record,c):
 need(contract['composition_identity']==c['reference_composition_identity']and contract['problem_identity']==c['problem_identity'],'fixed target composition/problem differs');need(contract['parent_identity']==digest(record)and contract['parent_id']==record['parent_id']and contract['upstream_identity']==record['upstream_identity'],'fixed complete parent identity differs');need(contract['parent_mass']==record['mass']==.125 and contract['parent_mass_used_in_logp']is False and contract['inference_target']=='population_cut_v1','fixed protected cut differs');need(contract['scientific_target_id']==target_identity(record,c),'full native target identity mismatch');return True

def compare_outputs(a,b):
 import numpy as np
 need(a.keys()==b.keys(),'output roster differs');e={k:base.close(a[k],b[k])for k in ['logp','poll_logp']};need(len(a['factor_values'])==len(b['factor_values']),'factor roster differs');e['factors']=max(base.close(x,y)for x,y in zip(a['factor_values'],b['factor_values']));need(a['rates'].keys()==b['rates'].keys()and a['physical'].keys()==b['physical'].keys(),'rate/physical roster differs');e['rates']=max(base.close(a['rates'][k],b['rates'][k])for k in a['rates']);e['physical']=max(base.close(a['physical'][k],b['physical'][k])for k in a['physical']);return e

def validate_roster(rows,ordinal):
 need(type(ordinal)is int and ordinal in range(2,8),'invalid selected fixed parent')
 need(all(type(row['parent_index'])is int and type(row['point_index'])is int for row in rows),'integer parent/point roster required')
 need([(row['parent_index'],row['point_index'])for row in rows]==[(ordinal,i)for i in range(3)],'wrong/repeated parent or point roster');return True

def read_new(root,phase,c,ordinal):
 out=Path(root)/phase;r=json.loads((out/'result.json').read_text());need(r['phase']==phase and r['selected_parent_ordinal']==ordinal and r['contract_sha256']==EXPECTED_CONTRACT and r['status']=='completed_calls_not_normalized_conditionals','selected phase/parent/result mismatch');rows=r['calls'];validate_roster(rows,ordinal)
 for name,expected in r['artifacts'].items():need(pin(out/name)==expected,'new phase artifact drift')
 bindings=base.binding(c);records=[row['record']for row in bindings.load()['parents']];record=records[ordinal];need(digest(record)==selected(c,ordinal)['parent_identity'],'contract selected record differs');kernel=json.loads((out/'kernel.json').read_text());need(kernel['phase']==phase and kernel['selected_parent_ordinal']==ordinal and kernel['contract_sha256']==EXPECTED_CONTRACT,'kernel selected parent mismatch')
 validate_fixed_target(kernel['compiler_parent_contract'],records[0]if phase=='dynamic-witness'else record,c)
 points=base.expected_points(c);calls=[]
 for row in rows:
  need(pin(out/row['file'])==row['pin'],'call bytes changed');call=ev.verify((out/row['file']).read_bytes(),row['pin']['sha256'],bindings);need(call['parent_record']==record and call['provenance']==kernel,'parent/source attribution mismatch');need(canonical(call['point'])==canonical(points[row['point_index']])and call['point_sha256']==row['point_sha256']==digest(points[row['point_index']]),'complete fixed point mismatch');calls.append(call)
 fd=json.loads((out/'finite-differences.json').read_text());need([row['variable']for row in fd]==c['finite_difference_variables'],'FD variable roster differs')
 for row in fd:
  need(row['coordinate']==0 and row['step']==c['fd_step']and len(row['calls'])==2,'FD coordinate/step/roster differs')
  for sign,item in zip([-1,1],row['calls']):
   need(pin(out/item['file'])==item['pin'],'FD call bytes changed');call=ev.verify((out/item['file']).read_bytes(),item['pin']['sha256'],bindings);expected=base.declared_fd_point(points,c,row['variable'],sign);need(call['parent_record']==record and call['provenance']==kernel and canonical(call['point'])==canonical(expected)and call['point_sha256']==digest(expected),'FD full parent/point/source mismatch');need(call['output']['logp']==item['value'],'FD value differs')
  need(row['derivative']==(row['calls'][1]['value']-row['calls'][0]['value'])/(2*c['fd_step']),'FD arithmetic mismatch')
 return calls,fd

def compare_witness(root,c,ordinal):
 authenticate_predecessors(c);saved,_=base.read_phase(c['baseline_attempt_root'],'dynamic',c);saved=[call for row,call in saved if row['parent_index']==ordinal];witness,fd=read_new(root,'dynamic-witness',c,ordinal);need(len(saved)==len(witness)==3,'saved selected dynamic roster missing');errors=[]
 for a,b in zip(saved,witness):
  need(a['parent_record']==b['parent_record']and a['point']==b['point'],'old/new dynamic pairing differs');errors.append(compare_outputs(a['output'],b['output']))
 result={'schema':'RemainingParentDynamicFDWitnessV1','ordinal':ordinal,'status':'three_saved_dynamic_points_match_new_owned_FD_witness','errors':errors,'why_new_witness':'original dynamic FD calls existed only for parent1; no derivative fabricated for2..7','FD_variables':[row['variable']for row in fd],'conditional_normalization':'unverified'};atomic_json(Path(root)/'dynamic-witness-comparison.json',result);return result

def compare_fixed(root,c,ordinal):
 compare_witness(root,c,ordinal);witness,wfd=read_new(root,'dynamic-witness',c,ordinal);fixed,ffd=read_new(root,'fixed-selected',c,ordinal);errors=[]
 for a,b in zip(witness,fixed):
  need(a['parent_record']==b['parent_record']and a['point']==b['point'],'independent fixed parent/point pairing differs');need(a['provenance']['compiler_parent_contract']['parent_identity']!=b['provenance']['compiler_parent_contract']['parent_identity'],'fixed parent graph not independent');errors.append(compare_outputs(a['output'],b['output']))
 fderr=[{'variable':a['variable'],'error':base.close(a['derivative'],b['derivative'],.01,1e-5)}for a,b in zip(wfd,ffd)]
 result={'schema':'RemainingIndependentFixedParentV1','ordinal':ordinal,'parent_identity':c['remaining_parents'][str(ordinal)]['parent_identity'],'status':'three_point_factor_rate_physical_and_exact_FD_comparison_passed','errors':errors,'FD_errors':fderr,'outer_masses':[.125]*8,'other_remaining_parents_not_inferred':[i for i in range(2,8)if i!=ordinal],'conditional_normalization':'unverified','scope':'one independently selected fixed parent, pointwise only; no sampler/posterior or outer evidence reweight'};atomic_json(Path(root)/'fixed-selected-comparison.json',result);return result
