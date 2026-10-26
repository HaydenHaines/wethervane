"""Complete numerical arrays and nonmeasurement ownership comparison."""
import json
from pathlib import Path
import numpy as np
from common import need,pin,atomic_json

def arrays_equal(a,b,atol=1e-6,rtol=1e-9):
 need(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'shape/nonfinite mismatch');delta=np.abs(a-b);need(np.all(delta<=atol+rtol*np.maximum(np.abs(a),np.abs(b))),'numerical equivalence tolerance exceeded');return float(delta.max(initial=0))
def envelope_equal(a,b):
 # Different implementation/composition/target IDs are retained, not certified aliases.
 ac=dict(a['joint_contract']);bc=dict(b['joint_contract'])
 for key in ['composition_identity','scientific_target_id']:ac.pop(key);bc.pop(key)
 need(ac==bc and ac['parent_mass_used_in_logp']is False,'protected/target contract mismatch')
 need(a['parent']==b['parent']and a['model_ids']==b['model_ids'],'parent/evidence changed')
 def normalized(inv):
  out={}
  for key,value in inv.items():
   row=dict(value)
   if key.startswith('measurement:'):key='measurement';row.pop('implementation')
   out[key]=row
  return out
 need(normalized(a['component_inventory'])==normalized(b['component_inventory']),'factor ownership/nonmeasurement inventory changed')
 # Complete composition records bind every remaining provider spec and compiler dependency.
 ar=a['composition_record'];br=b['composition_record'];need(ar['compiler_and_baseline_dependencies']==br['compiler_and_baseline_dependencies'],'compiler dependencies changed')
 ai=ar['implementations'];bi=br['implementations'];need(len(ai)==len(bi),'provider roster changed')
 changes=[(x,y)for x,y in zip(ai,bi)if x!=y];need(len(changes)==1,'more than one provider spec changed')
 return {'distinct_composition_ids':a['joint_contract']['composition_identity']!=b['joint_contract']['composition_identity'],'distinct_target_ids':a['joint_contract']['scientific_target_id']!=b['joint_contract']['scientific_target_id']}
def compare(root):
 root=Path(root);receipts=[];data=[];envelopes=[]
 for arm in ['original','compressed']:
  p=root/arm;r=json.loads((p/'result.json').read_text());need(r['status']=='completed'and r['arm']==arm,'arm incomplete')
  for name,expected in r['artifacts'].items():need(pin(p/name)==expected,'arm artifact changed')
  receipts.append(r);envelopes.append(json.loads((p/'model-envelope.json').read_text()))
  with np.load(p/'numerical.npz',allow_pickle=False)as f:data.append({k:f[k].copy()for k in f.files})
 need(receipts[0]['contract_sha256']==receipts[1]['contract_sha256']and receipts[0]['parent_digest']==receipts[1]['parent_digest'],'arm provenance changed')
 need((root/'original/layout.json').read_bytes()==(root/'compressed/layout.json').read_bytes(),'coordinate order mismatch')
 with np.load(root/'original/points.npz',allow_pickle=False)as a,np.load(root/'compressed/points.npz',allow_pickle=False)as b:
  need(a.files==b.files and all(np.array_equal(a[k],b[k])for k in a.files),'shared full point mismatch')
 need(data[0]['gradient'].shape==data[1]['gradient'].shape==(2,662),'incomplete full vector')
 errors={k:arrays_equal(data[0][k],data[1][k])for k in ['logp','gradient','poll_logp','rates','factor_logp']};ids=envelope_equal(*envelopes)
 np.savez(root/'differences.npz',**{k:data[1][k]-data[0][k]for k in data[0]})
 result={'schema':'ActualCompressedEquivalenceComparisonV1','status':'fixed_state_equivalence_passed','max_errors':errors,'identity_distinction':ids,'artifacts':{'differences.npz':pin(root/'differences.npz')},'arms':{a:pin(root/a/'result.json')for a in ['original','compressed']},'scope':'Two exact shared points, first parent, complete ordered 662-vectors; selected full-joint finite differences only. No other-parent normalization, predictive qualification or target-ID alias.'};atomic_json(root/'comparison.json',result);return result
