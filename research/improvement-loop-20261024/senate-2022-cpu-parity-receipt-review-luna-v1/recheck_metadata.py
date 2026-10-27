"""Independent metadata-only cross-check of the saved 2022 CPU parent-0 receipt."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
ROOT=Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
BASE=ROOT/'research/senate-stack-integration-20261020/national-2022-gpu-preparation-root-v1'
CAND=BASE/'gpu-kernel-argument-remediation-v1'
RECEIPT=CAND/'cpu-jax-parent0-parity-v2/result.json'
RECEIPT_SHA='109feacd03133852444239e9bc173481b41a92086208722230537a0a38f72d39'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok: raise ValueError(msg)

raw=RECEIPT.read_bytes();need(hashlib.sha256(raw).hexdigest()==RECEIPT_SHA,'receipt bytes differ')
r=json.loads(raw)
paths={
 'source_sha256':CAND/'cpu_jax_parent0_parity_v2.py',
 'CPU_reference_sha256':BASE/'parent0-native-reference-v2/result.json',
 'all_parent_structure_sha256':BASE/'all-parent-graph-structure-v1/result.json',
 'adapter_boundary_sha256':BASE/'candidate-boundary-validation-v2/result.json',
 'boundary_sha256':CAND/'cpu-jax-parent0-boundary-validation-v2.json',
 'original_plan_sha256':BASE.parent/'national-multicycle-execution-astra-v2/A2022/UNADMITTED-plan.json',
}
for key,path in paths.items():need(sha(path)==r[key],f'{key} hash differs')
reference=json.loads(paths['CPU_reference_sha256'].read_bytes())
structure=json.loads(paths['all_parent_structure_sha256'].read_bytes())
boundary=json.loads(paths['boundary_sha256'].read_bytes())
need(reference['status']=='PASS_native_parent0_reference' and reference['states']==5 and reference['full_gradients']==5,'native reference metadata differs')
names=reference['physical_names'];need(len(names)==815 and len(set(names))==815,'native physical roster is not unique 815')
need(structure['status']=='PASS_all_parent_structure_only' and structure['parent_indices']==list(range(8)),'structure metadata differs')
need(boundary['status']=='PASS_synthetic_boundaries_only','boundary status differs')
need(reference['original_plan_sha256']==structure['original_plan_sha256']==r['original_plan_sha256']==sha(paths['original_plan_sha256']),'plan pin chain differs')
expected={str((CAND/n).resolve()) for n in ('cpu_jax_parent0_parity_v2.py','test_cpu_jax_parent0_parity_v2.py')}
pins=boundary['source_pins'];need(set(pins)==expected and all(sha(Path(p))==h for p,h in pins.items()),'boundary source pins differ')
need(sha(CAND/'cpu-jax-parent0-parity-v2/cpu_jax_parent0_parity_v2.py')==r['source_sha256'],'saved runtime copy differs')
rows=r['checks'];need(len(rows)==5 and [x['point'] for x in rows]==list(range(5)),'point order/roster differs')
want=['logp','full_gradient',*names];maxfrac=0.0;count=0
for row in rows:
    checks=row['checks'];need(row['passed'] is True and len(checks)==817,'saved point record incomplete')
    need([x['variable'] for x in checks]==want,'check roster/order differs')
    for x in checks:
        value=x['max_tolerance_fraction']
        need(x['pass_check'] is True and x['failures']==0 and x['nonfinite_count']==0,'saved comparison failure flag')
        need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'invalid tolerance fraction')
        maxfrac=max(maxfrac,float(value));count+=1
need(count==4085 and r['complete_gradients_compared']==5 and r['physical_comparisons']==4075,'aggregate comparison count differs')
need(r['schema']=='CorrectedNationalParent0CPUJaxParityCandidateV1' and r['status']=='PASS_parent0_native_CPU_JAX_parity' and r['stage']=='complete','receipt status differs')
need(r['GPU_used'] is False and r['sampler_transitions']==0 and r['fit_admitted'] is False and r['forecast_qualified'] is False,'qualification flags differ')
need(sha(RECEIPT)==RECEIPT_SHA and all(sha(p)==r[k] for k,p in paths.items()),'sources changed during review')
print(json.dumps({'schema':'Independent2022CPUParent0ReceiptMetadataCheckV1','receipt_sha256':RECEIPT_SHA,
 'pins_rechecked':len(paths),'boundary_source_pins_rechecked':len(pins),'reference_physical_names':len(names),
 'points':len(rows),'checks_per_point':817,'total_saved_checks':count,'physical_comparisons':4075,
 'max_saved_tolerance_fraction':maxfrac,'numeric_vectors_loaded':False,'raw_comparisons_recomputed':False,
 'all_parent_numeric_parity':False,'GPU_parity':False,'fit_admitted':False,'forecast_qualified':False},indent=2,sort_keys=True))
