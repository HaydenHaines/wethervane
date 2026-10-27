"""Invented 42-call batch protocol fixture; copied base evidence + invented FD values.

Never model execution or authority for a real attempt. Writes only the given temp root.
"""
import copy,json
from pathlib import Path
from common import *
from batch_adapter import frozen_batch

def build(root,c):
 root=Path(root);out=root/'dynamic-six';out.mkdir(parents=True);cfixture=dict(c,batch_attempt_root=str(root))
 with frozen_batch(c)as b:
  bc=b['common'];cfg=bc.read_contract();saved,_=b['baseline_compare'].read_phase(cfg['baseline_attempt_root'],'dynamic',cfg);saved={(r['parent_index'],r['point_index']):v for r,v in saved};points=b['baseline_compare'].expected_points(cfg);kernel=copy.deepcopy(next(iter(saved.values()))['provenance']);kernel.update(phase='dynamic-six',parent_roster=list(range(2,8)),contract_sha256=bc.EXPECTED_CONTRACT,execution_scope='INVENTED adapter protocol fixture; no graph, copied saved base outputs and invented FD outputs');kernel.pop('selected_parent_ordinal',None);parents=[];sequence=0
  for ordinal in range(2,8):
   calls=[];fd=[]
   def store(name,call):
    nonlocal sequence
    sequence+=1;call['sequence']=sequence;call['provenance']=kernel;call['kernel_receipt_sha256']=bc.digest(kernel);call['execution_scope']=kernel['execution_scope'];raw=bc.canonical(call);(out/name).write_bytes(raw);return {'file':name,'pin':bc.pin(out/name)}
   for i in range(3):
    call=copy.deepcopy(saved[(ordinal,i)]);item=store('p'+str(ordinal)+'-point'+str(i)+'.json',call);item.update(parent_index=ordinal,point_index=i);calls.append(item)
   for variable in cfg['finite_difference_variables']:
    pair=[]
    for sign in [-1,1]:
     call=copy.deepcopy(saved[(ordinal,1)]);call['point']=b['baseline_compare'].declared_fd_point(points,cfg,variable,sign);call['point_sha256']=bc.digest(call['point']);call['output']['logp']+=sign*cfg['fd_step'];factor=b['baseline_compare'].np.asarray(call['output']['factor_values'][0]).copy();factor.flat[0]+=sign*cfg['fd_step'];call['output']['factor_values'][0]=factor.tolist() # invented FD, not scientific
     item=store('p'+str(ordinal)+'-fd-'+variable+str(sign)+'.json',call);item.update(sign=sign,value=call['output']['logp']);pair.append(item)
    fd.append({'variable':variable,'coordinate':0,'step':cfg['fd_step'],'calls':pair,'derivative':(pair[1]['value']-pair[0]['value'])/(2*cfg['fd_step'])})
   parent={'schema':'PartialBatchParentCallsV1','status':'numerical_calls_persisted_not_batch_success','ordinal':ordinal,'parent_identity':bc.digest(saved[(ordinal,0)]['parent_record']),'calls':calls,'finite_differences':fd,'conditional_normalization':'unverified'};name='parent-'+str(ordinal)+'-index.json';bc.atomic_json(out/name,parent);parents.append({'ordinal':ordinal,'file':name,'pin':bc.pin(out/name)})
  bc.atomic_json(out/'kernel.json',kernel);bc.atomic_json(out/'parent-index.json',parents)
  for name in ['entry.json','model-envelope.json','replacement-structure.json']:bc.atomic_json(out/name,{'fixture':True})
  r={'schema':'SixParentDynamicWitnessResultV1','status':'all_six_owned_calls_persisted_not_yet_compared','phase':'dynamic-six','contract_sha256':bc.EXPECTED_CONTRACT,'parent_roster':list(range(2,8)),'parents':parents,'artifacts':{n:bc.pin(out/n)for n in ['entry.json','model-envelope.json','replacement-structure.json','kernel.json','parent-index.json']}};bc.atomic_json(out/'result.json',r)
  review=json.loads(Path(c['batch_admission_review']).read_text());permit={'phase':'dynamic-six','parent_roster':list(range(2,8)),'contract_sha256':bc.EXPECTED_CONTRACT};auth={'parent_roster':list(range(2,8)),'contract_sha256':bc.EXPECTED_CONTRACT,'review':review,'review_digest':bc.digest(review),'resource_receipt':permit,'resources_digest':bc.digest(permit),'fixture':'invented fixture only, not review/resource authorization'};bc.atomic_json(out/'authorization.json',auth);bc.atomic_json(out/'terminal.json',{'arm':'dynamic-six','returncode':0,'reason':None,'fixture':True});_,execution=b['comparison_gate'].execute(root,cfg,review,bc.digest(review));bc.atomic_json(out/'comparison-execution.json',execution);ack=b['completion'].publish_success(out,{'comparison-execution.json':bc.pin(out/'comparison-execution.json'),'../all-six-comparison.json':bc.pin(root/'all-six-comparison.json')});tool=root/'fixture-caller-return.json';bc.atomic_json(tool,{'schema':'RetainedSelectedStageToolReturnV1','write_stdin_return':{'exit_code':0,'output':json.dumps(ack)},'scope':'INVENTED locally constructed adapter fixture; no actual batch controller/model run'})
 return cfixture,tool,sha(tool)

def fixed_fixture(root,c,ordinal,tool,toolsha,witness_bytes):
 """Invented fixed-stage receipts to exercise adapter/comparator wiring, never graph."""
 import hashlib
 root=Path(root);out=root/'fixed-from-batch';out.mkdir(parents=True);w=json.loads(witness_bytes);parent=w['parent'];first=w['calls'][parent['calls'][0]['file']];record=first['parent_record'];kernel=copy.deepcopy(first['provenance']);native=kernel['compiler_parent_contract'];native.update(parent_identity=digest(record),parent_id=record['parent_id'],upstream_identity=record['upstream_identity'],parent_mass=.125,scientific_target_id=digest({'composition_id':c['reference_composition_identity'],'problem':c['problem_identity'],'parent':digest(record),'target':'population_cut_v1'}));kernel.update(phase='fixed-from-batch',selected_parent_ordinal=ordinal,contract_sha256=EXPECTED_CONTRACT,batch_selection_sha256=hashlib.sha256(witness_bytes).hexdigest(),execution_scope='INVENTED fixed protocol fixture, not native graph');kernel.pop('parent_roster',None);(out/'batch-selection.json').write_bytes(witness_bytes);auth={'batch_tool_return':str(tool),'batch_tool_return_sha256':toolsha,'batch_selection':pin(out/'batch-selection.json')};atomic_json(out/'authorization.json',auth);rows=[];fd=[]
 def clone(item):
  call=copy.deepcopy(w['calls'][item['file']]);call['provenance']=kernel;call['kernel_receipt_sha256']=digest(kernel);call['execution_scope']=kernel['execution_scope'];name='fixed-'+item['file'];(out/name).write_bytes(canonical(call));return dict(item,file=name,pin=pin(out/name))
 for item in parent['calls']:
  row=clone(item);row['point_sha256']=w['calls'][item['file']]['point_sha256'];row['parent_identity']=digest(record);rows.append(row)
 for row in parent['finite_differences']:fd.append(dict(row,calls=[clone(item)for item in row['calls']]))
 atomic_json(out/'finite-differences.json',fd);atomic_json(out/'kernel.json',kernel);atomic_json(out/'entry.json',{'fixture':True});atomic_json(out/'model-envelope.json',{'fixture':True});atomic_json(out/'call-index.json',rows);r={'status':'completed_calls_not_normalized_conditionals','phase':'fixed-from-batch','selected_parent_ordinal':ordinal,'contract_sha256':EXPECTED_CONTRACT,'calls':rows,'artifacts':{n:pin(out/n)for n in ['finite-differences.json','kernel.json','entry.json','model-envelope.json','call-index.json']}};atomic_json(out/'result.json',r);return out
