"""One remaining parent; saved dynamic baseline plus new explicit FD witness."""
import json
from pathlib import Path
from common import *
import evaluator as ev
import baseline_compare as base
from selection import selected,authenticate_predecessors
from batch_adapter import select_witness

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


def compare_fixed_from_batch(root,c,ordinal):
    root=Path(root);auth=json.loads((root/'fixed-from-batch/authorization.json').read_text());raw=select_witness(c,ordinal,auth['batch_tool_return'],auth['batch_tool_return_sha256']);need(hashlib.sha256(raw).hexdigest()==auth['batch_selection']['sha256']and pin(root/'fixed-from-batch/batch-selection.json')==auth['batch_selection'],'selected batch witness changed');witness=json.loads(raw);parent=witness['parent'];fixed,ffd=read_new(root,'fixed-from-batch',c,ordinal);errors=[]
    for item,call in zip(parent['calls'],fixed):
        reference=witness['calls'][item['file']];need(reference['parent_record']==call['parent_record']and reference['point']==call['point'],'batch/fixed complete parent or point differs');need(reference['provenance']['compiler_parent_contract']['parent_identity']!=call['provenance']['compiler_parent_contract']['parent_identity'],'fixed graph is not independently formed');errors.append(compare_outputs(reference['output'],call['output']))
    wfd=parent['finite_differences'];need([row['variable']for row in wfd]==[row['variable']for row in ffd]==c['finite_difference_variables'],'batch/fixed FD roster differs');fd_errors=[]
    for a,b in zip(wfd,ffd):
        output_errors=[]
        for source,item in zip(a['calls'],b['calls']):
            reference=witness['calls'][source['file']];actual=json.loads((root/'fixed-from-batch'/item['file']).read_text());need(reference['parent_record']==actual['parent_record']and reference['point']==actual['point'],'batch/fixed complete FD pairing differs');output_errors.append(compare_outputs(reference['output'],actual['output']))
        fd_errors.append({'variable':a['variable'],'error':base.close(a['derivative'],b['derivative'],.01,1e-5),'output_errors':output_errors})
    result={'schema':'FixedFromCompletedBatchComparisonV1','status':'selected_fixed_parent_matches_completed_batch_points_and_FD','ordinal':ordinal,'parent_identity':c['remaining_parents'][str(ordinal)]['parent_identity'],'errors':errors,'FD_errors':fd_errors,'outer_masses':[.125]*8,'batch_success':witness['batch_success'],'batch_tool_return':witness['batch_tool_return'],'other_remaining_parents_not_inferred':[i for i in range(2,8)if i!=ordinal],'conditional_normalization':'unverified','scope':'one independently built fixed parent; no normalizer/posterior'};atomic_json(root/'fixed-from-batch-comparison.json',result);return result
