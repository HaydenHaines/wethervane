"""Small numerical receipt comparisons only; never builds a model."""
import importlib.util,json,sys,copy
from pathlib import Path
import numpy as np
from common import *
import evaluator as ev

def close(a,b,atol=1e-6,rtol=1e-9):
 a=np.asarray(a,float);b=np.asarray(b,float);need(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'shape/nonfinite mismatch');error=np.abs(a-b);need(np.all(error<=atol+rtol*np.maximum(np.abs(a),np.abs(b))),'numerical difference exceeds tolerance');return float(error.max(initial=0))
def binding(c):
 spec=importlib.util.spec_from_file_location('compare_parent_binding',c['parent_binding']);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def expected_points(c):
 """Reconstruct every exact declared point from pinned complete source bytes."""
 layout=json.loads(Path(c['saved_layout']).read_text());names=[row['name']for row in layout]
 with np.load(c['saved_points'],allow_pickle=False)as data:
  need(set(data.files)=={str(i)+'__'+name for i in range(2)for name in names},'saved point field roster mismatch')
  values=[{name:data[str(i)+'__'+name].copy()for name in names}for i in range(2)]
 values.append({name:value.copy()for name,value in values[1].items()})
 for name,delta in c['third_point_deltas'].items():
  need(name in values[2],'unknown declared perturbation');values[2][name].flat[0]+=delta
 return [{'layout':json.loads(point.layout_bytes),'values':json.loads(point.values_bytes)}for point in [ev.point(layout,value)for value in values]]

def check_roster_and_points(entries,points,phase,anchor):
 need(phase in ['dynamic','fixed-B']and len(points)==3,'unsupported phase/point roster')
 indices=list(range(8))+[0]if phase=='dynamic'else[anchor]
 expected=[(i,pi)for i in indices for pi in range(3)]
 need(all(type(row['parent_index'])is int and type(row['point_index'])is int for row,call in entries),'integer roster indices required')
 actual=[(row['parent_index'],row['point_index'])for row,call in entries]
 need(actual==expected,'exact ordered parent/point roster mismatch')
 for row,call in entries:
  point=points[row['point_index']];need(canonical(call['point'])==canonical(point),'call is not the complete declared point')
  need(call['point_sha256']==row['point_sha256']==digest(point),'declared point digest mismatch')
 return True

def declared_fd_point(points,c,variable,sign):
 need(variable in c['finite_difference_variables']and sign in [-1,1],'undeclared finite difference')
 point=copy.deepcopy(points[1]);names=[row['name']for row in point['layout']];need(variable in names,'FD variable absent');point['values'][names.index(variable)][0]+=sign*c['fd_step'];return point

def read_phase(root,phase,c):
 out=Path(root)/phase;r=json.loads((out/'result.json').read_text());need(r['phase']==phase and r['contract_sha256']==EXPECTED_CONTRACT and r['status']=='completed_calls_not_normalized_conditionals','phase not completed')
 for n,expected in r['artifacts'].items():need(pin(out/n)==expected,'phase artifact changed')
 b=binding(c);records=[row['record']for row in b.load()['parents']];kernel=json.loads((out/'kernel.json').read_text());need(kernel['contract_sha256']==EXPECTED_CONTRACT and kernel['phase']==phase,'kernel provenance mismatch');calls=[]
 for row in r['calls']:
  raw=(out/row['file']).read_bytes();need(pin(out/row['file'])==row['pin'],'owned call file changed');call=ev.verify(raw,row['pin']['sha256'],b);need(call['parent_record']==records[row['parent_index']]and call['parent_envelope']['parent_identity']==row['parent_identity'],'call parent attribution mismatch');need(call['point_sha256']==row['point_sha256']and call['provenance']==kernel,'point/kernel provenance mismatch');calls.append((row,call))
 points=expected_points(c);check_roster_and_points(calls,points,phase,c['anchor_index'])
 fd=json.loads((out/'finite-differences.json').read_text())
 need([row['variable']for row in fd]==c['finite_difference_variables'],'exact ordered FD variable roster mismatch')
 for row in fd:
  need(row['coordinate']==0 and row['step']==c['fd_step']and len(row['calls'])==2,'FD coordinate/step/call roster mismatch')
  for sign,item in zip([-1,1],row['calls']):
   need(pin(out/item['file'])==item['pin'],'FD call changed');call=ev.verify((out/item['file']).read_bytes(),item['pin']['sha256'],b);need(call['parent_record']==records[c['anchor_index']]and call['provenance']==kernel,'FD parent/kernel provenance mismatch');need(call['output']['logp']==item['value'],'FD scalar differs from owned result');expected=declared_fd_point(points,c,row['variable'],sign);need(canonical(call['point'])==canonical(expected)and call['point_sha256']==digest(expected),'FD complete point differs from declared signed perturbation')
  need(row['derivative']==(row['calls'][1]['value']-row['calls'][0]['value'])/(2*c['fd_step']),'stored FD derivative mismatch')
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
