import json,hashlib,math
from pathlib import Path
import numpy as np
B=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-compressed-actual-equivalence-probe-sol-v2');R=B/'attempt-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'sha256':sha(p),'bytes':Path(p).stat().st_size}
def read(p):return json.loads(Path(p).read_text())
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
c=read(B/'contract.json');review=read(B.parent/'m01-compressed-actual-equivalence-review-luna-v2/independent-review.json')
for p,h in c['pins'].items():assert sha(p)==h,p
for n,h in review['packet_pins'].items():assert sha(B/n)==h,n
assert sha(B/'contract.json')==review['contract_sha256']
expected={'original':'0b79a9c9cd9088ccd42e67b7abaf8ff03ce0a8272e5c98248f16e66aca95c2cf','compressed':'987742fe0bdc301dd0b6fbbaa32cc399e422defe3206d9a093ccbe248ed2c8c8'}
arrays={};envs={};records={};points={};layouts={}
for arm in ['original','compressed']:
 a=R/arm;r=read(a/'result.json');t=read(a/'terminal.json');ack=read(a/'controller-success.json');au=read(a/'authorization.json');e=read(a/'entry-receipt.json');la=read(a/'launch.json');ev=read(a/'evaluation-receipt.json');layout=read(a/'layout.json');en=read(a/'model-envelope.json')
 assert pin(a/'controller-success.json')=={'sha256':expected[arm],'bytes':374}
 assert ack['result']==pin(a/'result.json') and ack['terminal']==pin(a/'terminal.json')
 assert ack['contract_sha256']==r['contract_sha256']==au['contract_sha256']==sha(B/'contract.json')
 assert au['review']==review and dig(review)==au['review_digest'] and dig(au['resource_receipt'])==au['resources_digest']
 assert au['resource_receipt']['independent_review_sha256']==au['review_digest']
 assert 0<=la['live_resources']['checked_unix']-au['resource_receipt']['checked_unix']<=300
 assert la['live_resources']['mem_available_bytes']>=au['resource_receipt']['minimum_mem_available_bytes'] and la['live_resources']['disk_free_bytes']>=au['resource_receipt']['minimum_disk_free_bytes']
 assert r['arm']==t['arm']==la['arm']==arm and r['status']=='completed'
 for n,p in r['artifacts'].items():assert pin(a/n)==p,n
 assert t['returncode']==0 and t['reason'] is None and t['controller_error'] is None
 assert r['pid']==t['pid']==e['pid']==la['pid'] and au['controller_pid']==e['controller_pid']
 assert e['controller_start_ticks']==la['controller_start_ticks'] and e['channel_sha256']==la['control_channel_sha256']
 assert e['nice']==15 and e['cpu_affinity']==[au['resource_receipt']['cpu']] and e['wall_alarm_seconds']==600 and e['parent_death_signal']=='SIGKILL'
 assert e['limits']=={'0':[540,550],'9':[8589934592]*2,'1':[67108864]*2,'4':[0,0]}
 assert not Path('/proc/'+str(e['pid'])).exists() and not Path('/proc/'+str(e['controller_pid'])).exists()
 assert t['peak_threads']==1 and t['peak_rss_bytes']<c['caps']['rss_bytes'] and t['peak_disk_bytes']<c['caps']['arm_disk_bytes'] and t['seconds']<c['caps']['wall_seconds']
 assert ev['artifact']==pin(a/'evaluated-before-fd.npz') and ev['points']==pin(a/'points.npz') and ev['layout']==pin(a/'layout.json')
 assert pin(a/'evaluated-before-fd.npz')==pin(a/'numerical.npz')
 assert sum(x['size']for x in layout)==662 and len({x['name']for x in layout})==len(layout)
 with np.load(a/'points.npz',allow_pickle=False)as z:points[arm]={n:z[n].copy()for n in z.files}
 assert set(points[arm])=={str(i)+'__'+row['name']for i in range(2)for row in layout}
 for i in range(2):
  for row in layout:
   v=points[arm][str(i)+'__'+row['name']];assert np.isfinite(v).all() and list(v.shape)==row['shape'] and v.size==row['size'] and str(v.dtype)==row['dtype']
 with np.load(a/'numerical.npz',allow_pickle=False)as z:arrays[arm]={n:z[n].copy()for n in z.files}
 d=arrays[arm];assert set(d)=={'logp','gradient','poll_logp','rates','factor_logp'} and d['gradient'].shape==(2,662)
 for v in d.values():assert np.isfinite(v).all()
 assert np.max(np.abs(d['factor_logp'].sum(axis=1)-d['logp']))<1e-6
 offsets={};offset=0
 for row in layout:offsets[row['name']]=offset;offset+=row['size']
 assert [x['name']for x in r['finite_difference_checks']]==list(c['perturbations'])
 for x in r['finite_difference_checks']:
  assert x['flat_index']==offsets[x['name']] and x['step']==c['fd_step'] and x['analytic']==float(d['gradient'][1,x['flat_index']])
  assert math.isclose(x['error'],abs(x['numeric']-x['analytic']),rel_tol=1e-14) and math.isclose(x['tolerance'],c['fd_atol']+c['fd_rtol']*abs(x['numeric']),rel_tol=1e-14) and x['error']<=x['tolerance']
 assert dig(en['parent'])==c['parent_digest']==r['parent_digest']==en['joint_contract']['parent_identity']
 assert en['joint_contract']['problem_identity']==c['problem_identity'] and en['joint_contract']['parent_mass']==en['parent']['mass']==.125 and en['joint_contract']['parent_mass_used_in_logp']is False
 assert r['runtime']==c['runtime_expected']
 envs[arm]=en;layouts[arm]=layout;records[arm]={'terminal':t,'controllerPid':e['controller_pid'],'allPidAbsent':True,'receiptPins':{n:pin(a/n)for n in ['result.json','terminal.json','controller-success.json','authorization.json','launch.json','entry-receipt.json','evaluation-receipt.json','model-envelope.json','child.log']},'finiteDifferenceChecks':len(r['finite_difference_checks']),'factorSumMaxError':float(np.max(np.abs(d['factor_logp'].sum(axis=1)-d['logp'])))}
assert (R/'original/layout.json').read_bytes()==(R/'compressed/layout.json').read_bytes()
assert points['original'].keys()==points['compressed'].keys()
for n in points['original']:assert np.array_equal(points['original'][n],points['compressed'][n]),n
comp=read(R/'comparison.json');assert pin(R/'comparison.json')=={'sha256':'79bc079ae74bf3e2779dfbdbe17bfd5809d732a4c82d8a5667c183334d2950cc','bytes':1024}
assert comp['artifacts']['differences.npz']==pin(R/'differences.npz')
for arm in records:assert comp['arms'][arm]==pin(R/arm/'result.json')
errors={};counts={}
with np.load(R/'differences.npz',allow_pickle=False)as z:
 assert set(z.files)==set(arrays['original'])
 for n in arrays['original']:
  a=arrays['original'][n];b=arrays['compressed'][n];assert a.shape==b.shape
  diff=b-a;assert np.array_equal(z[n],diff) and np.isfinite(z[n]).all(),n
  assert np.all(np.abs(diff)<=c['comparison_atol']+c['comparison_rtol']*np.maximum(np.abs(a),np.abs(b))),n
  errors[n]=float(np.max(np.abs(diff),initial=0));counts[n]=int(diff.size)
assert errors==comp['max_errors']
a,b=envs['original'],envs['compressed'];ja=dict(a['joint_contract']);jb=dict(b['joint_contract'])
for k in ['composition_identity','scientific_target_id']:
 assert ja[k]!=jb[k];ja.pop(k);jb.pop(k)
assert ja==jb and a['parent']==b['parent'] and a['model_ids']==b['model_ids']
assert set(a['component_inventory'])==set(b['component_inventory'])
for k,x in a['component_inventory'].items():
 y=b['component_inventory'][k]
 if k.startswith('measurement:'):
  x=dict(x);y=dict(y);assert x.pop('implementation')!=y.pop('implementation');assert x==y
 else:assert x==y,k
ar=a['composition_record'];br=b['composition_record'];assert ar['compiler_and_baseline_dependencies']==br['compiler_and_baseline_dependencies']
changes=[(x,y)for x,y in zip(ar['implementations'],br['implementations'])if x!=y];assert len(ar['implementations'])==len(br['implementations']) and len(changes)==1
assert changes[0][0]['role']==changes[0][1]['role']=='measurement'
print(json.dumps({'status':'passed_fixed_state_comparison_audit','sourcePinsVerified':len(c['pins']),'contract':pin(B/'contract.json'),'review':pin(B.parent/'m01-compressed-actual-equivalence-review-luna-v2/independent-review.json'),'comparison':pin(R/'comparison.json'),'differences':pin(R/'differences.npz'),'everyNumericEntryCompared':counts,'maxErrorsRecomputed':errors,'allDifferencesExactlyRecomputed':True,'exactSharedPointArrays':len(points['original']),'layoutVariables':len(layouts['original']),'nonMeasurementInventoryExact':True,'measurementInventoryExactExceptDistinctImplementation':True,'changedProviderSpecs':changes,'distinctCompositionAndTargetIds':True,'remainingJointContractExact':True,'arms':records,'callerScope':'Acknowledgments supplied by parent retained sessions85285/79247; reviewer independently matches disk hashes but cannot access original tool objects.','scope':'Two shared states, first protected parent, actual_calibration_bank only. No graph rerun, pickle loading, all-parent proof, forecast or performance claim.'},indent=2))
