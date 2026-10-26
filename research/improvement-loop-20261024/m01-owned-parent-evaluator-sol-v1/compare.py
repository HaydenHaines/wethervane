"""Small numerical receipt comparisons only; never builds a model."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from common import *
import evaluator as ev

def close(a,b,atol=1e-6,rtol=1e-9):
 a=np.asarray(a,float);b=np.asarray(b,float);need(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'shape/nonfinite mismatch');error=np.abs(a-b);need(np.all(error<=atol+rtol*np.maximum(np.abs(a),np.abs(b))),'numerical difference exceeds tolerance');return float(error.max(initial=0))
def binding(c):
 spec=importlib.util.spec_from_file_location('compare_parent_binding',c['parent_binding']);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def read_phase(root,phase,c):
 out=Path(root)/phase;r=json.loads((out/'result.json').read_text());need(r['phase']==phase and r['contract_sha256']==EXPECTED_CONTRACT and r['status']=='completed_calls_not_normalized_conditionals','phase not completed')
 for n,expected in r['artifacts'].items():need(pin(out/n)==expected,'phase artifact changed')
 b=binding(c);records=[row['record']for row in b.load()['parents']];kernel=json.loads((out/'kernel.json').read_text());need(kernel['contract_sha256']==EXPECTED_CONTRACT and kernel['phase']==phase,'kernel provenance mismatch');calls=[]
 for row in r['calls']:
  raw=(out/row['file']).read_bytes();need(pin(out/row['file'])==row['pin'],'owned call file changed');call=ev.verify(raw,row['pin']['sha256'],b);need(call['parent_record']==records[row['parent_index']]and call['parent_envelope']['parent_identity']==row['parent_identity'],'call parent attribution mismatch');need(call['point_sha256']==row['point_sha256']and call['provenance']==kernel,'point/kernel provenance mismatch');calls.append((row,call))
 fd=json.loads((out/'finite-differences.json').read_text())
 for row in fd:
  for item in row['calls']:
   need(pin(out/item['file'])==item['pin'],'FD call changed');call=ev.verify((out/item['file']).read_bytes(),item['pin']['sha256'],b);need(call['parent_record']==records[c['anchor_index']]and call['provenance']==kernel,'FD parent/kernel provenance mismatch');need(call['output']['logp']==item['value'],'FD scalar differs from owned result')
 return calls,fd

def compare_dynamic(root,c):
 calls,fd=read_phase(root,'dynamic',c);need(len(calls)==27,'all8x3 plus A revisit required');first=[call for row,call in calls[:3]];last=[call for row,call in calls[-3:]];need([canonical(x['output'])for x in first]==[canonical(x['output'])for x in last],'A to all8 to A changed numerical outputs')
 errors=[]
 with np.load(c['saved_numerical'],allow_pickle=False)as saved:
  for i in range(2):
   output=first[i]['output'];errors.append({'point':i,'logp':close(output['logp'],saved['logp'][i]),'poll_logp':close(output['poll_logp'],saved['poll_logp'][i]),'factors':close([np.sum(v)for v in output['factor_values']],saved['factor_logp'][i]),'selected_contest_rate':close(output['rates'][c['target_contest']],saved['rates'][i])})
 result={'schema':'OwnedDynamicParent0BaselineV1','status':'saved_parent0_baseline_and_A_return_passed','errors':errors,'all_eight_scope':'pointwise owned calls at3coordinates only; not native fixed-parent parity or normalized conditionals','missing_fixed_baselines':list(range(1,8)),'conditional_normalization':'unverified'};atomic_json(Path(root)/'dynamic-baseline.json',result);return result

def compare(root,c):
 compare_dynamic(root,c);dynamic,dfd=read_phase(root,'dynamic',c);fixed,ffd=read_phase(root,'fixed-B',c);dyn=[call for row,call in dynamic if row['parent_index']==c['anchor_index']];fix=[call for row,call in fixed];need(len(dyn)==len(fix)==3,'B needs three common points');errors=[]
 for a,b in zip(dyn,fix):
  need(a['parent_record']==b['parent_record']and a['point']==b['point'],'B exact parent/point mismatch');need(a['provenance']['compiler_parent_contract']['parent_identity']!=b['provenance']['compiler_parent_contract']['parent_identity'],'B not independently formed fixed graph');oa,ob=a['output'],b['output'];e={'logp':close(oa['logp'],ob['logp']),'poll_logp':close(oa['poll_logp'],ob['poll_logp'])};need(len(oa['factor_values'])==len(ob['factor_values']),'factor roster differs');e['factors']=max(close(x,y)for x,y in zip(oa['factor_values'],ob['factor_values']));need(set(oa['rates'])==set(ob['rates'])and set(oa['physical'])==set(ob['physical']),'rate/physical roster differs');e['rates']=max(close(oa['rates'][k],ob['rates'][k])for k in oa['rates']);e['physical']=max(close(oa['physical'][k],ob['physical'][k])for k in oa['physical']);errors.append(e)
 need([row['variable']for row in dfd]==[row['variable']for row in ffd],'FD coordinate order differs');fd_errors=[{'variable':a['variable'],'error':close(a['derivative'],b['derivative'],.01,1e-5)}for a,b in zip(dfd,ffd)]
 result={'schema':'OwnedParentIndependentBAnchorV1','status':'three_point_all_factors_rates_B_anchor_passed','anchor_index':c['anchor_index'],'anchor_selection':c['anchor_selection'],'errors':errors,'finite_difference_errors':fd_errors,'missing_fixed_parent_baselines':[i for i in range(8)if i not in [0,c['anchor_index']]],'conditional_normalization':'unverified','outer_masses':[.125]*8,'scope':'pointwise kernel reuse checks and one independent native B anchor; no fits/sampling/normalized conditionals'};atomic_json(Path(root)/'comparison.json',result);return result
