"""Read/hash only: no owner imports, graph or arrays."""
from pathlib import Path
import hashlib,json,datetime
B=Path('/home/hayden/projects/wethervane-redesign-20261006/integration');H=B/'research/senate-stack-integration-20261020/national-2018-gpu-preparation-root-v1';A=H/'gpu-production-v1/attempt1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def pin(p):return {'sha256':sha(p),'bytes':p.stat().st_size}
def main():
 pp=A/'plan.json';p=read(pp);la=read(A/'launch.json');rec=read(A/'receipt.json');sa=read(A/'sampling-receipt.json');rt=B/p['runtime']['path'];oldpath=B/p['numerical_preflight']['original_plan']['path']
 assert sha(pp)==sha(H/'gpu-production-v1/plan.json')==la['plan']['sha256']==sa['plan']['sha256']
 assert sha(rt)==p['runtime']['sha256'];assert sha(oldpath)==p['numerical_preflight']['original_plan']['sha256'];old=read(oldpath)
 rows=[]
 for i,(n,h)in enumerate(p['source_pins'].items()):
  s=A/'source-snapshots'/f'{i:03d}-{Path(n).name}';c=B/n
  rows.append({'originalPath':n,'expected':h,'snapshotPath':str(s),'snapshot':pin(s),'snapshotMatches':sha(s)==h,'current':pin(c)if c.is_file()else None,'currentMatches':c.is_file()and sha(c)==h,'originalScientificPinMatches':old['source_pins'].get(n)==h})
 assert len(list((A/'source-snapshots').iterdir()))==len(rows)
 assert read(A/'fit/protocol.json')['source_pins']==p['source_pins']
 for n,r in p['source_inputs'].items():assert sha(A/'input-snapshots'/f'{n}.json')==r['sha256']
 selected=[r for r in rows if '/national-problem-assembly-astra-v1/'in r['originalPath']]
 assert len(selected)==11 and all(r['snapshotMatches']and r['currentMatches']for r in selected)
 assert read(rt)['inventory']==p['runtime_inventory']
 meta=[pp,H/'gpu-production-v1/plan.json',A/'launch.json',A/'receipt.json',A/'sampling-receipt.json',rt,oldpath,A/'fit/protocol.json',H/'gpu_runner_v2.py',H/'cached_provider.py']
 return {'schema':'National2018RetainedSourceLineageIndependentV1','observedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sourceEntries':len(rows),'snapshotMatches':sum(r['snapshotMatches']for r in rows),'currentMatches':sum(r['currentMatches']for r in rows),'snapshotBytesRead':sum(r['snapshot']['bytes']for r in rows),'selectedPackage':selected,'allSources':rows,'metadataPins':{str(x):pin(x)for x in meta},'runtimeInventoryMatchesPlan':True,'attemptProtocolSourcePinsMatch':True,'inputSnapshotsMatchSourceInputs':True,'receiptStatus':rec['status'],'scope':'Read-only hashes/JSON; retained local bytes are not immutable/external execution attestation.'}
if __name__=='__main__':print(json.dumps(main(),indent=2))
