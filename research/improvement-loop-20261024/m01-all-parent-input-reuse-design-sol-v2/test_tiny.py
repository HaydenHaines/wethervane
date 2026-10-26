"""Invented output receipts only, authentic parent metadata; no model imports."""
import copy,json,resource
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(10,15));resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,)*2)
import binding as b
H=Path(__file__).resolve().parent;m=b.load();tests=[]
def reject(name,fn):
 try:fn()
 except (ValueError,TypeError,FrozenInstanceError):tests.append(name)
 else:raise AssertionError('accepted '+name)
a=b.bind(m['parents'][0]['record']);other=b.bind(m['parents'][1]['record'])
for row in m['parents']:
 bound=b.bind(row['record']);raw=b.seal_output(bound,exposure_used=bound.exposure,envelope_claim=bound.envelope_bytes,kernel_id='invented-kernel',point_id='invented-point',result={'invented_value':1});assert b.verify_output(raw)==bound
 tests.append('parent_output_roundtrip_'+str(row['record']['row_id']))
reject('frozen_envelope_assignment',lambda:setattr(a,'envelope_bytes',other.envelope_bytes))
reject('frozen_exposure_assignment',lambda:setattr(a,'exposure',other.exposure))
for field,value in [('parent_id',json.loads(other.envelope_bytes)['parent_id']),('row_id',json.loads(other.envelope_bytes)['row_id']),('mass',.5),('reference_fixed_parent_target_id','wrong-target'),('numerical_input_sha256','0'*64),('parent_identity',json.loads(other.envelope_bytes)['parent_identity'])]:
 claim=json.loads(a.envelope_bytes);claim[field]=value;reject('changed_claim_'+field,lambda:b.seal_output(a,exposure_used=a.exposure,envelope_claim=claim,kernel_id='invented',point_id='invented',result=1))
reject('wrong_exposure_same_envelope',lambda:b.seal_output(a,exposure_used=other.exposure,envelope_claim=a.envelope_bytes,kernel_id='invented',point_id='invented',result=1))
bad=list(map(list,a.exposure));bad[0][0]*=1.00001;reject('exposure_value_digest_tamper',lambda:b.seal_output(a,exposure_used=bad,envelope_claim=a.envelope_bytes,kernel_id='invented',point_id='invented',result=1))
# Even a deliberate bypass of Python frozen assignment is detected at the final check.
corrupt=b.bind(m['parents'][0]['record']);object.__setattr__(corrupt,'envelope_bytes',other.envelope_bytes);reject('bypassed_frozen_envelope_revalidated',lambda:b.seal_output(corrupt,exposure_used=corrupt.exposure,envelope_claim=corrupt.envelope_bytes,kernel_id='invented',point_id='invented',result=1))
corrupt=b.bind(m['parents'][0]['record']);object.__setattr__(corrupt,'exposure',other.exposure);reject('bypassed_frozen_exposure_revalidated',lambda:b.revalidate(corrupt))
claim=json.loads(a.envelope_bytes);raw=b.seal_output(a,exposure_used=a.exposure,envelope_claim=claim,kernel_id='invented',point_id='invented',result=1);claim['mass']=.5;assert json.loads(raw)['parent_envelope']['mass']==.125;tests.append('mutation_after_seal_cannot_change_bytes')
changed=json.loads(raw);changed['parent_envelope']['mass']=.5;reject('serialized_mass_tamper',lambda:b.verify_output(b.canonical(changed)));changed=json.loads(raw);changed['input_exposure_sha256']='0'*64;reject('serialized_exposure_digest_tamper',lambda:b.verify_output(b.canonical(changed)))
changed=json.loads(raw);changed['parent_record']['parent_id']=json.loads(other.record_bytes)['parent_id'];reject('serialized_record_identity_swap',lambda:b.verify_output(b.canonical(changed)))
assert b.bind(m['parents'][0]['record'])==a and b.bind(m['parents'][1]['record'])==other;tests.append('A_B_A_immutable_binding')
# Invented finite kernels only; actual conditional normalizers remain unknown.
kernels=[[F(1),F(1)]]*7+[[F(2),F(2)]];weights=[F(1,8)]*8;conditionals=[[v/sum(row)for v in row]for row in kernels];joint=[[w*v for v in row]for w,row in zip(weights,conditionals)];assert all(sum(row)==F(1,8)for row in joint);tests.append('cut_preserves_fixed_masses_under_Z_imbalance')
result={'schema':'ParentOutputAttributionTinyTestsV2','status':'passed','tests':tests,'actual_graph_imported':False,'actual_kernel_evaluated':False,'actual_conditional_normalization':'unverified','example_output':json.loads(raw),'source_pins':{n:b.sha(H/n)for n in ['binding.py','test_tiny.py','parents-manifest.json']}}
(H/'tiny-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(tests)}))
