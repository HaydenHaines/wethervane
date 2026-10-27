"""Verify the frozen architecture packet and its 20 source pins."""
import hashlib,json
from pathlib import Path
TARGET=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/queue-protected-history-architecture-sol-v1')
manifest=json.loads((TARGET/'source-pins.json').read_bytes())
fail=[];checked=[]
for raw,spec in manifest.items():
    path=Path(raw);data=path.read_bytes();got=hashlib.sha256(data).hexdigest()
    checked.append({'path':raw,'sha256':got,'bytes':len(data)})
    if got!=spec['sha256'] or len(data)!=spec['bytes']:
        fail.append(raw)
packet=json.loads((TARGET/'evidence.json').read_bytes())
selfcheck={}
for name,spec in packet['pins'].items():
    data=(TARGET/name).read_bytes();got=hashlib.sha256(data).hexdigest()
    selfcheck[name]={'sha256':got,'bytes':len(data),'match':got==spec['sha256'] and len(data)==spec['bytes']}
if fail or not all(x['match'] for x in selfcheck.values()):
    raise SystemExit(json.dumps({'source_mismatches':fail,'packet_checks':selfcheck},indent=2))
print(json.dumps({'schema':'ProtectedHistoryArchitecturePinReviewV1','source_count':len(checked),
 'all_source_pins_match':True,'source_mismatches':fail,'packet_self_pins':selfcheck},indent=2,sort_keys=True))
