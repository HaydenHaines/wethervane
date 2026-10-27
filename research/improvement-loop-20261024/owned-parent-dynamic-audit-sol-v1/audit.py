import json,hashlib,math,copy
from pathlib import Path
import numpy as np
L=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024');B=L/'m01-owned-parent-evaluator-sol-v3';A=B/'attempt-v1/dynamic'
def raw(p):return p.read_bytes()
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def read(p):return json.loads(raw(p))
def pin(p):return {'sha256':sha(p),'bytes':p.stat().st_size}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def dig(x):return hashlib.sha256(canonical(x)).hexdigest()
def finite(x):
 if isinstance(x,dict):return all(finite(v)for v in x.values())
 if isinstance(x,list):return all(finite(v)for v in x)
 return isinstance(x,(float,int))and math.isfinite(x)
def close(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all();d=np.abs(a-b);assert np.all(d<=1e-6+1e-9*np.maximum(np.abs(a),np.abs(b)));return float(d.max(initial=0))
c=read(B/'contract.json');r=read(A/'result.json');t=read(A/'terminal.json');ack=read(A/'controller-success.json');au=read(A/'authorization.json');e=read(A/'entry.json');la=read(A/'launch.json');k=read(A/'kernel.json');ex=read(A/'comparison-execution.json');base=read(B/'attempt-v1/dynamic-baseline.json');caller=read(L/'m01-owned-parent-evaluator-admission-root-v2/dynamic-caller-return.json');out=json.loads(caller['write_stdin_return']['output']);assert caller['write_stdin_return']['exit_code']==0
assert out['controller_success']==pin(A/'controller-success.json') and out['comparisons']['execution']==pin(A/'comparison-execution.json') and out['comparisons']['baseline']==pin(B/'attempt-v1/dynamic-baseline.json')
assert ack['result']==pin(A/'result.json') and ack['terminal']==pin(A/'terminal.json') and ack['comparison_artifacts']==out['comparisons']
assert ack['contract_sha256']==r['contract_sha256']==ex['contract_sha256']==sha(B/'contract.json')
for p,h in c['pins'].items():assert sha(Path(p))==h,p
for n,h in au['review']['packet_pins'].items():assert sha(B/n)==h==ex['source_sha256'][n]
assert dig(au['review'])==au['review_digest']==ex['review_digest'] and dig(au['resource_receipt'])==au['resources_digest']
assert 0<=la['live_resources']['checked_unix']-au['resource_receipt']['checked_unix']<=300
for n,p in r['artifacts'].items():assert pin(A/n)==p
assert t['returncode']==0 and t['reason']is None and t['controller_error']is None and t['arm']=='dynamic'
assert e['pid']==t['pid']==la['pid']==k['pid'] and e['controller_pid']==au['controller_pid'] and e['controller_start_ticks']==la['controller_start_ticks'] and e['channel_sha256']==la['control_channel_sha256']
assert not Path('/proc/'+str(e['pid'])).exists()and not Path('/proc/'+str(e['controller_pid'])).exists()
assert e['nice']==15 and e['parent_death_signal']=='SIGKILL' and e['cpu_affinity']==[au['resource_receipt']['cpu']]
assert t['peak_threads']==1 and t['peak_rss_bytes']<c['caps']['rss_bytes'] and t['seconds']<c['caps']['wall_seconds']
assert k['source_bundle_sha256']==dig(c['pins']) and k['runtime']==c['runtime_expected'] and k['gradient']=='not_constructed_logp_only'
env=read(A/'model-envelope.json');assert dig(env['inventory'])==k['model_inventory_sha256']and env['layout']==k['coordinate_layout']and env['contract']==k['compiler_parent_contract']
manifest=Path(c['parent_binding']).parent/'parents-manifest.json';m=read(manifest);assert sha(manifest)==k['parent_source_manifest_sha256']
for p,h in m['pins'].items():assert sha(Path(p))==h
layout=read(Path(c['saved_layout']));points=[]
with np.load(c['saved_points'],allow_pickle=False)as z:
 for i in range(2):points.append({'layout':layout,'values':[z[str(i)+'__'+row['name']].reshape(-1).tolist()for row in layout]})
points.append(copy.deepcopy(points[1]))
for name,delta in c['third_point_deltas'].items():points[2]['values'][[x['name']for x in layout].index(name)][0]+=delta
rows=read(A/'call-index.json');assert rows==r['calls'];assert [(x['parent_index'],x['point_index'])for x in rows]==[(i,j)for i in list(range(8))+[0]for j in range(3)]
calls=[];numbercount=0
for seq,row in enumerate(rows,1):
 q=A/row['file'];assert pin(q)==row['pin'];x=read(q);parent=m['parents'][row['parent_index']]['record'];pe=x['parent_envelope'];assert x['sequence']==seq and x['parent_record']==parent
 assert dig(parent)==row['parent_identity']==pe['parent_identity'] and pe['parent_id']==parent['parent_id'] and x['outer_mass']==pe['mass']==parent['mass']==.125
 exposure=[[float(v)for v in rr]for rr in parent['exposure']];assert x['exposure']==exposure and dig(exposure)==x['input_exposure_sha256']==pe['numerical_input_sha256']
 assert x['point']==points[row['point_index']]and dig(x['point'])==x['point_sha256']==row['point_sha256']
 assert x['provenance']==k and x['kernel_receipt_sha256']==dig(k) and x['conditional_normalization']=='unverified' and pe['parent_mass_used_in_logp']is False
 assert len(x['output']['factor_values'])==env['factor_count']and set(x['output']['rates'])==set(env['contests'])and set(x['output']['physical'])==set(env['physical'])
 assert pe['reference_fixed_parent_target_id']==m['parents'][row['parent_index']]['reference_fixed_parent_target_id']
 assert finite(x['output']);assert abs(sum(np.asarray(v).sum()for v in x['output']['factor_values'])-x['output']['logp'])<1e-6
 for name in ['rates','physical']:
  for v in x['output'][name].values():assert np.all(np.asarray(v)>0)
 calls.append(x)
for a,b in zip(calls[:3],calls[-3:]):assert canonical(a['output'])==canonical(b['output'])
errors=[]
with np.load(c['saved_numerical'],allow_pickle=False)as z:
 for i in range(2):
  o=calls[i]['output'];errors.append({'point':i,'logp':close(o['logp'],z['logp'][i]),'poll_logp':close(o['poll_logp'],z['poll_logp'][i]),'factors':close([np.asarray(v).sum()for v in o['factor_values']],z['factor_logp'][i]),'selected_contest_rate':close(o['rates'][c['target_contest']],z['rates'][i])})
assert errors==base['errors'] and base['missing_fixed_baselines']==list(range(1,8))
fd=read(A/'finite-differences.json');assert [x['variable']for x in fd]==c['finite_difference_variables']
for index,row in enumerate(fd):
 assert row['coordinate']==0 and row['step']==c['fd_step']
 for sign,item in zip([-1,1],row['calls']):
  q=A/item['file'];assert pin(q)==item['pin'];x=read(q);expected=copy.deepcopy(points[1]);expected['values'][[v['name']for v in layout].index(row['variable'])][0]+=sign*c['fd_step']
  assert x['point']==expected and x['point_sha256']==dig(expected)and x['parent_record']==m['parents'][c['anchor_index']]['record']and x['provenance']==k and finite(x['output'])
  assert x['sequence']==28+index*2+(sign==1)and item['value']==x['output']['logp']
 assert row['derivative']==(row['calls'][1]['value']-row['calls'][0]['value'])/(2*c['fd_step'])and math.isfinite(row['derivative'])
assert not (B/'attempt-v1/fixed-B').exists()and not(B/'attempt-v1/comparison.json').exists()
print(json.dumps({'status':'dynamic_receipts_and_saved_numerical_comparison_passed','sourceDependenciesVerified':len(c['pins']),'calls':len(calls),'parentIndices':list(range(8)),'points':len(points),'AreturnAllOutputsBitwiseJSONEqual':True,'baselineErrorsRecomputed':errors,'finiteDifferenceDerivatives':[{'variable':x['variable'],'derivative':x['derivative']}for x in fd],'analyticGradientConstructed':False,'FDScalarCalls':4,'fixedBPresent':False,'conditionalNormalization':'unverified','terminal':t,'entry':e,'callerReturnFile':pin(L/'m01-owned-parent-evaluator-admission-root-v2/dynamic-caller-return.json'),'pins':{n:pin(A/n)for n in ['result.json','terminal.json','controller-success.json','comparison-execution.json','kernel.json','model-envelope.json','call-index.json','finite-differences.json','entry.json','launch.json','authorization.json']},'baseline':pin(B/'attempt-v1/dynamic-baseline.json'),'scope':'Parent0 saved two-point baseline only; three owned points for all8 and A-return,2 scalar central differences at B. No all-parent native baseline, full gradient, fixedB, normalizers, posterior or forecast.'},indent=2))
