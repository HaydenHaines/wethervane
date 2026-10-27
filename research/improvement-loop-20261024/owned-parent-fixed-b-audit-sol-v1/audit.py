"""JSON/source-only independent fixed-B audit; never imports owner/model modules."""
import json,hashlib,math,copy
from pathlib import Path
L=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024');B=L/'m01-owned-parent-evaluator-sol-v3';R=B/'attempt-v1';A=R/'fixed-B'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def pin(p):return {'sha256':sha(p),'bytes':p.stat().st_size}
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def dig(v):return hashlib.sha256(canon(v)).hexdigest()
def numbers(v):
 if isinstance(v,list):return sum((numbers(x)for x in v),[])
 assert isinstance(v,(int,float))and math.isfinite(v);return [v]
def close(a,b,atol=1e-6,rtol=1e-9):
 if isinstance(a,list)or isinstance(b,list):
  assert isinstance(a,list)and isinstance(b,list)and len(a)==len(b)
  return max((close(x,y,atol,rtol)for x,y in zip(a,b)),default=0.)
 assert math.isfinite(a)and math.isfinite(b);err=abs(a-b);assert err<=atol+rtol*max(abs(a),abs(b));return err
c=read(B/'contract.json');r=read(A/'result.json');t=read(A/'terminal.json');ack=read(A/'controller-success.json');au=read(A/'authorization.json');entry=read(A/'entry.json');launch=read(A/'launch.json');kernel=read(A/'kernel.json');execution=read(A/'comparison-execution.json');comparison=read(R/'comparison.json');caller=read(L/'m01-owned-parent-evaluator-admission-root-v2/fixed-b-caller-return.json');returned=json.loads(caller['write_stdin_return']['output'])
assert caller['write_stdin_return']['exit_code']==0 and returned['controller_success']==pin(A/'controller-success.json')and returned['comparisons']['comparison']==pin(R/'comparison.json')and returned['comparisons']['execution']==pin(A/'comparison-execution.json')
assert sha(A/'controller-success.json')=='099e9df34ef49244eab866ea90a86606b67aa5a1836a213dc3bd48a061c1fa3a'
assert ack['result']==pin(A/'result.json')and ack['terminal']==pin(A/'terminal.json')and ack['comparison_artifacts']==returned['comparisons']
assert sha(B/'contract.json')==ack['contract_sha256']==r['contract_sha256']==execution['contract_sha256']==au['contract_sha256']
for p,h in c['pins'].items():assert sha(Path(p))==h
for n,h in au['review']['packet_pins'].items():assert sha(B/n)==h==execution['source_sha256'][n]
assert dig(au['review'])==au['review_digest']==execution['review_digest']and dig(au['resource_receipt'])==au['resources_digest']
assert 'fixed-B'in au['review']['allowed_phases']and au['resource_receipt']['phase']=='fixed-B'
assert 0<=launch['live_resources']['checked_unix']-au['resource_receipt']['checked_unix']<=300
assert launch['live_resources']['mem_available_bytes']>=au['resource_receipt']['minimum_mem_available_bytes']and launch['live_resources']['disk_free_bytes']>=au['resource_receipt']['minimum_disk_free_bytes']
for n,p in r['artifacts'].items():assert pin(A/n)==p
assert t['arm']=='fixed-B'and t['returncode']==0 and t['reason']is None and t['controller_error']is None
assert t['pid']==entry['pid']==launch['pid']==kernel['pid']and entry['controller_pid']==au['controller_pid']and entry['controller_start_ticks']==launch['controller_start_ticks']and entry['channel_sha256']==launch['control_channel_sha256']
assert entry['nice']==15 and entry['cpu_affinity']==[au['resource_receipt']['cpu']]and entry['parent_death_signal']=='SIGKILL'
assert not Path('/proc/'+str(entry['pid'])).exists()and not Path('/proc/'+str(entry['controller_pid'])).exists()
assert t['peak_threads']==1 and t['peak_rss_bytes']<c['caps']['rss_bytes']and t['seconds']<c['caps']['wall_seconds']
assert kernel['runtime']==c['runtime_expected']and kernel['source_bundle_sha256']==dig(c['pins'])and kernel['gradient']=='not_constructed_logp_only'
m=read(Path(c['parent_binding']).parent/'parents-manifest.json');assert sha(Path(c['parent_binding']).parent/'parents-manifest.json')==kernel['parent_source_manifest_sha256']
for p,h in m['pins'].items():assert sha(Path(p))==h
parent=m['parents'][1]['record'];assert dig(parent)==kernel['compiler_parent_contract']['parent_identity']and parent['mass']==.125
old=read(L/'owned-parent-dynamic-audit-sol-v1/result.json')
for n,p in old['pins'].items():assert pin(R/'dynamic'/n)==p
assert pin(R/'dynamic-baseline.json')==old['baseline']
dynamic=read(R/'dynamic/result.json');drows=[row for row in dynamic['calls']if row['parent_index']==1];assert len(drows)==3
fixed=read(A/'call-index.json');assert fixed==r['calls']and [(x['parent_index'],x['point_index'])for x in fixed]==[(1,i)for i in range(3)]
errors=[];counts=[];fixedcalls=[]
for i,(dr,fr)in enumerate(zip(drows,fixed)):
 assert pin(R/'dynamic'/dr['file'])==dr['pin']and pin(A/fr['file'])==fr['pin'];d=read(R/'dynamic'/dr['file']);f=read(A/fr['file']);fixedcalls.append(f)
 assert f['sequence']==i+1 and f['parent_record']==d['parent_record']==parent and f['point']==d['point']and dig(f['point'])==f['point_sha256']==fr['point_sha256']
 E=[[float(v)for v in row]for row in parent['exposure']];assert f['exposure']==E==d['exposure']and dig(E)==f['input_exposure_sha256']==f['parent_envelope']['numerical_input_sha256']
 assert f['provenance']==kernel and f['kernel_receipt_sha256']==dig(kernel)and f['outer_mass']==.125 and f['parent_envelope']['parent_mass_used_in_logp']is False
 assert d['provenance']['compiler_parent_contract']['parent_identity']!=f['provenance']['compiler_parent_contract']['parent_identity']
 a,b=d['output'],f['output'];assert set(a)==set(b)=={'logp','poll_logp','factor_values','rates','physical'}
 er={n:close(a[n],b[n])for n in ['logp','poll_logp']};er['factors']=close(a['factor_values'],b['factor_values'])
 for field in ['rates','physical']:
  assert set(a[field])==set(b[field]);er[field]=max(close(a[field][k],b[field][k])for k in a[field])
  assert all(v>0 for val in b[field].values()for v in numbers(val))
 assert abs(sum(numbers(b['factor_values']))-b['logp'])<1e-6
 errors.append(er);counts.append({k:sum(len(numbers(v))for v in b[k].values())if isinstance(b[k],dict)else len(numbers(b[k]))for k in b})
assert errors==comparison['errors']
ffd=read(A/'finite-differences.json');dfd=read(R/'dynamic/finite-differences.json');assert [x['variable']for x in ffd]==c['finite_difference_variables']==[x['variable']for x in dfd];fd_errors=[]
for index,(f,d)in enumerate(zip(ffd,dfd)):
 assert f['coordinate']==0 and f['step']==c['fd_step']
 for sign,item in zip([-1,1],f['calls']):
  assert pin(A/item['file'])==item['pin'];call=read(A/item['file']);expected=copy.deepcopy(fixedcalls[1]['point']);expected['values'][[x['name']for x in expected['layout']].index(f['variable'])][0]+=sign*c['fd_step']
  assert call['point']==expected and call['point_sha256']==dig(expected)and call['parent_record']==parent and call['provenance']==kernel and call['output']['logp']==item['value']
  assert call['sequence']==4+index*2+(sign==1)
 assert f['derivative']==(f['calls'][1]['value']-f['calls'][0]['value'])/(2*c['fd_step'])
 fd_errors.append({'variable':f['variable'],'error':close(d['derivative'],f['derivative'],.01,1e-5)})
assert fd_errors==comparison['finite_difference_errors']and comparison['missing_fixed_parent_baselines']==list(range(2,8))and comparison['conditional_normalization']=='unverified'and comparison['outer_masses']==[.125]*8
print(json.dumps({'status':'fixedB_audit_passed_pointwise_only','sourcePinCount':len(c['pins']),'callerReturn':pin(L/'m01-owned-parent-evaluator-admission-root-v2/fixed-b-caller-return.json'),'comparison':pin(R/'comparison.json'),'controllerSuccess':pin(A/'controller-success.json'),'terminal':t,'entry':entry,'allThreePointErrorsRecomputed':errors,'perPointNumericEntriesCompared':counts,'bothFDScalarErrorsRecomputed':fd_errors,'allOriginalDynamicReceiptPinsStillMatch':True,'fixedParentIndex':1,'fixedParentIdentity':dig(parent),'missingNativeBaselines':list(range(2,8)),'gradientConstructed':False,'conditionalNormalization':'unverified','pins':{n:pin(A/n)for n in ['authorization.json','result.json','terminal.json','kernel.json','entry.json','launch.json','call-index.json','finite-differences.json','model-envelope.json','comparison-execution.json']},'scope':'No graph/comparator rerun, no sampling, all-parent native parity, posterior/normalizer/forecast qualification.'},indent=2))
