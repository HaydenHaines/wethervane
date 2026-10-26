"""Pure bounded receipt/point/comparison helpers; no graph imports."""
import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Filled once when preparation is frozen; never discover a current digest as authorization.
EXPECTED_CONTRACT='c4b9aa3a5d14ac509bc1e25ff30bd50918d9cd331a8ba426c1537a4be0268fea'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def read_contract():
 path=HERE/'contract.json';need(sha(path)==EXPECTED_CONTRACT,'frozen contract changed');c=json.loads(path.read_text())
 for path,expected in c['pins'].items():need(sha(path)==expected,'pinned source changed: '+path)
 return c
def atomic_json(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
 # Publication errors propagate; file existence is never a completion receipt.
 fd=os.open(path.parent,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def pin(path):return {'sha256':sha(path),'bytes':Path(path).stat().st_size}
def admission(review,review_sha,permit,permit_sha,contract_sha,now=None):
 now=time.time()if now is None else now
 need(digest(review)==review_sha,'caller-retained review digest mismatch')
 need(digest(permit)==permit_sha,'caller-retained resource digest mismatch')
 need(review['schema']=='OwnedParentEvaluatorIndependentReviewV1'and review['decision']=='allow_one_bounded_attempt','review not passed')
 need(review['contract_sha256']==contract_sha,'review contract mismatch')
 need(isinstance(review['allowed_phases'],list)and bool(review['allowed_phases'])and set(review['allowed_phases'])<= {'dynamic','fixed-B'},'unsupported review phases')
 expected_names={'common.py','child.py','launch.py','compare.py','test_tiny.py','contract.json','entry.py','evaluator.py'}
 need(set(review['packet_pins'])==expected_names,'complete independent source review pins required')
 for name,expected in review['packet_pins'].items():need(sha(HERE/name)==expected,'reviewed packet source changed: '+name)
 need(permit['schema']=='OwnedParentEvaluatorFreshResourcesV1'and permit['contract_sha256']==contract_sha,'resource contract mismatch')
 need(0<=now-permit['checked_unix']<=300,'resource check stale or future')
 need(permit['independent_review_sha256']==review_sha,'resource/review identity mismatch')
 need(permit['exclusive_actual_probe_slot']is True and permit['resource_conflicts']==[] and permit['overlap_assessment']=='no_conflict_for_this_one_cpu_job','resource/staging conflict not cleared')
 need(permit['minimum_mem_available_bytes']>=6*1024**3 and permit['minimum_disk_free_bytes']>=2*1024**3,'resource floors too small')
 need(permit['phase']in review['allowed_phases'],'resource/review phase mismatch')
 need(isinstance(permit['cpu'],int)and isinstance(permit['protected_jobs'],list),'specific CPU and protected-job snapshot required')
 need(isinstance(permit['observer'],str)and bool(permit['observer']),'resource observer required')
 return True
