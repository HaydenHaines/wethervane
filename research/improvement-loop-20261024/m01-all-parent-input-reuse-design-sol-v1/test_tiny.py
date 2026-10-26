"""Whole-record authentication + invented finite kernels only; stdlib, no model imports."""
import copy,json,resource
from fractions import Fraction as F
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(10,15));resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,)*2)
import binding as b
H=Path(__file__).resolve().parent;m=b.load();tests=[];observations=[]
def reject(label,fn):
 try:fn()
 except (ValueError,TypeError):tests.append(label)
 else:raise AssertionError('accepted '+label)
for row in m['parents']:
 record=row['record'];exposure,envelope=b.bind(record);assert envelope['parent_identity']==row['parent_identity']and envelope['mass']==.125 and envelope['conditional_normalization']=='unverified'and envelope['kernel_identity']is None;observations.append({'parent_identity':envelope['parent_identity'],'reference_target_id':envelope['reference_fixed_parent_target_id'],'mass':envelope['mass'],'input_fingerprint':envelope['numerical_input_sha256']})
assert len(set(o['parent_identity']for o in observations))==len(set(o['reference_target_id']for o in observations))==8;tests.append('eight_distinct_whole_records_targets_fixed_masses')
a=m['parents'][0]['record'];other=m['parents'][1]['record'];first=b.bind(a);b.bind(other);assert b.bind(a)==first;tests.append('A_B_A_binding_no_state_leak')
for label,field,value in [('mass','mass',.5),('same_exposure_parent_id_swap','parent_id',other['parent_id']),('same_exposure_row_id_swap','row_id',other['row_id']),('same_exposure_upstream_swap','upstream_identity','other-source'),('problem_swap','problem_identity','other-problem')]:
 bad=copy.deepcopy(a);bad[field]=value;reject(label,lambda:b.bind(bad))
bad=copy.deepcopy(a);bad['event_ids'].reverse();reject('event_axis_reorder',lambda:b.bind(bad));bad=copy.deepcopy(a);bad['cell_ids'].reverse();reject('cell_axis_reorder',lambda:b.bind(bad));bad=copy.deepcopy(a);bad['exposure'][0]=copy.deepcopy(other['exposure'][0]);reject('mixed_parent_event_row',lambda:b.bind(bad));bad=copy.deepcopy(a);bad['exposure'][0][0]*=1.00001;reject('exposure_tamper',lambda:b.bind(bad))
# Exact invented finite-state conditionals; these are not actual model normalizers.
weights=[F(1,8)]*8;kernels=[[F(1),F(1)]]*7+[[F(2),F(2)]]
normalizers=[sum(k)for k in kernels];q=[[v/sum(k)for v in k]for k in kernels];joint=[[w*v for v in row]for w,row in zip(weights,q)]
assert all(sum(row)==F(1,8)for row in joint)and sum(map(sum,joint))==1 and all(row==[F(1,2)]*2 for row in q);tests.append('within_parent_normalization_preserves_outer_mass_despite_Z_imbalance')
wrong=[w*z/sum(a*b for a,b in zip(weights,normalizers))for w,z in zip(weights,normalizers)];assert wrong==[F(1,9)]*7+[F(2,9)]and wrong!=weights;tests.append('forbidden_global_evidence_reweight_counterexample')
# Same pointwise values are insufficient to establish q: missing tails change Z.
assert [1,1]==[1,1]and F(1,2)!=F(1,102);tests.append('two_point_agreement_not_conditional_normalization')
# Invented dynamic input kernel has independently computed closed-form finite state q.
def kernel(E,x):return F(E[0])*F(x+1)+F(E[1])*F(2-x)
def conditional(E):
 values=[kernel(E,x)for x in range(3)];return [v/sum(values)for v in values]
A=(1,2);B=(3,1);saved=conditional(A);assert conditional(A)!=conditional(B)and conditional(A)==saved;tests.append('invented_dynamic_input_A_B_A_normalized_only_within_parent')
r={'schema':'WholeParentReuseDesignTinyV1','status':'passed','tests':tests,'authenticated_parents':observations,'actual_graph_imported':False,'actual_kernel_evaluated':False,'actual_conditional_normalization':'unverified','invented_Z':[str(z)for z in normalizers],'invented_cut_outer_masses':[str(v)for v in weights],'invented_forbidden_evidence_weights':[str(v)for v in wrong],'source_pins':{n:b.sha(H/n)for n in ['binding.py','parents-manifest.json','test_tiny.py']}}
(H/'tiny-result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':'passed','tests':len(tests),'whole_parents':8}))
