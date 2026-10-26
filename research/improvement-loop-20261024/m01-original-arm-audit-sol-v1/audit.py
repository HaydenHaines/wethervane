import json,hashlib,os,math
from pathlib import Path
import numpy as np
B=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-compressed-actual-equivalence-probe-sol-v2');A=B/'attempt-v1/original'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'sha256':sha(p),'bytes':Path(p).stat().st_size}
def read(p):return json.loads(Path(p).read_text())
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
c=read(B/'contract.json');r=read(A/'result.json');t=read(A/'terminal.json');ack=read(A/'controller-success.json');auth=read(A/'authorization.json');entry=read(A/'entry-receipt.json');launch=read(A/'launch.json');ev=read(A/'evaluation-receipt.json');layout=read(A/'layout.json');review=read(B.parent/'m01-compressed-actual-equivalence-review-luna-v2/independent-review.json')
assert pin(A/'controller-success.json')=={'sha256':'0b79a9c9cd9088ccd42e67b7abaf8ff03ce0a8272e5c98248f16e66aca95c2cf','bytes':374}
assert ack['result']==pin(A/'result.json') and ack['terminal']==pin(A/'terminal.json')
assert ack['contract_sha256']==sha(B/'contract.json')==r['contract_sha256']==auth['contract_sha256']
assert auth['review']==review and dig(review)==auth['review_digest']
assert dig(auth['resource_receipt'])==auth['resources_digest']
for n,h in review['packet_pins'].items():assert sha(B/n)==h,n
for p,h in c['pins'].items():assert sha(p)==h,p
for n,p in r['artifacts'].items():assert pin(A/n)==p,n
assert r['status']=='completed' and r['arm']==t['arm']==launch['arm']=='original'
assert t['returncode']==0 and t['reason'] is None and t['controller_error'] is None
assert r['pid']==t['pid']==entry['pid']==launch['pid']==1987686
assert entry['controller_pid']==auth['controller_pid']==1987682
assert entry['controller_start_ticks']==launch['controller_start_ticks']
assert entry['channel_sha256']==launch['control_channel_sha256']
assert entry['nice']==15 and entry['cpu_affinity']==[3] and entry['parent_death_signal']=='SIGKILL'
assert entry['wall_alarm_seconds']==c['caps']['wall_seconds']
assert entry['limits']=={'0':[540,550],'9':[8589934592]*2,'1':[67108864]*2,'4':[0,0]}
assert not Path('/proc/1987686').exists() and not Path('/proc/1987682').exists()
assert t['peak_rss_bytes']<c['caps']['rss_bytes'] and t['peak_threads']==1 and t['peak_disk_bytes']<c['caps']['arm_disk_bytes'] and t['seconds']<c['caps']['wall_seconds']
assert ev['artifact']==pin(A/'evaluated-before-fd.npz') and ev['points']==pin(A/'points.npz') and ev['layout']==pin(A/'layout.json')
assert pin(A/'numerical.npz')==pin(A/'evaluated-before-fd.npz')
assert len({x['name'] for x in layout})==len(layout) and sum(x['size'] for x in layout)==662
with np.load(A/'points.npz',allow_pickle=False) as p:
 assert set(p.files)=={str(i)+'__'+row['name'] for i in range(2) for row in layout}
 for i in range(2):
  for row in layout:
   a=p[str(i)+'__'+row['name']];assert list(a.shape)==row['shape'] and a.size==row['size'] and str(a.dtype)==row['dtype'] and np.isfinite(a).all()
 for row in layout:
  delta=p['1__'+row['name']]-p['0__'+row['name']];expected=np.zeros(row['shape']);
  if row['name'] in c['perturbations']:expected.flat[0]=c['perturbations'][row['name']]
  assert np.allclose(delta,expected,rtol=0,atol=1e-14)
with np.load(A/'numerical.npz',allow_pickle=False) as d:
 shapes={n:list(d[n].shape) for n in d.files}
 assert d['gradient'].shape==(2,662) and d['logp'].shape==(2,)
 for n in d.files:assert np.isfinite(d[n]).all(),n
 assert np.max(np.abs(d['factor_logp'].sum(axis=1)-d['logp']))<1e-6
 vals=d['logp'].tolist();factor_error=float(np.max(np.abs(d['factor_logp'].sum(axis=1)-d['logp'])))
 offsets={};k=0
 for row in layout:offsets[row['name']]=k;k+=row['size']
 assert [x['name'] for x in r['finite_difference_checks']]==list(c['perturbations'])
 for x in r['finite_difference_checks']:
  assert x['flat_index']==offsets[x['name']] and x['step']==c['fd_step']
  assert x['analytic']==float(d['gradient'][1,x['flat_index']])
  assert math.isclose(x['error'],abs(x['numeric']-x['analytic']),rel_tol=1e-14)
  assert math.isclose(x['tolerance'],c['fd_atol']+c['fd_rtol']*abs(x['numeric']),rel_tol=1e-14)
  assert x['error']<=x['tolerance']
env=read(A/'model-envelope.json');assert dig(env['parent'])==c['parent_digest']==r['parent_digest']
assert env['joint_contract']['problem_identity']==c['problem_identity'] and env['joint_contract']['parent_identity']==c['parent_digest']
assert env['joint_contract']['parent_mass_used_in_logp'] is False and env['joint_contract']['parent_mass']==env['parent']['mass']==0.125
for n,p in r['runtime'].items():assert p==c['runtime_expected'][n],n
print(json.dumps({'status':'passed_original_only','scope':'read-only JSON/hash and allow_pickle=False numerical audit; no graph/model import','contract':pin(B/'contract.json'),'review':pin(B.parent/'m01-compressed-actual-equivalence-review-luna-v2/independent-review.json'),'controllerSuccess':pin(A/'controller-success.json'),'sourcePinsVerified':len(c['pins']),'artifactPinsVerified':len(r['artifacts']),'pidAbsentAtAudit':[1987686,1987682],'shapes':shapes,'logp':vals,'maxFactorSumError':factor_error,'fdChecks':len(r['finite_difference_checks']),'fdAdditionalFullFunctionEvaluations':2*len(r['finite_difference_checks']),'baseFunctionEvaluations':2,'terminal':t,'runtime':r['runtime'],'receiptPins':{n:pin(A/n) for n in ['authorization.json','launch.json','entry-receipt.json','result.json','terminal.json','controller-success.json','evaluation-receipt.json','model-envelope.json','stage.json','child.log']},'callerToolReturn':'Parent-provided retained session 85285 exit 0 and acknowledgment digest; not independently accessible in this process.'},indent=2))
