"""Export actual calibration conditioning labels; never invent native factor tokens."""
import ast,collections,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
nodes=ast.parse((HERE/'provider.py').read_text()).body
expected=next(n.value.value for n in nodes if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='EXPECTED_SELECTION_SHA256'for t in n.targets))
raw=(HERE/'selection.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==expected;s=json.loads(raw);problem=json.loads(Path(s['problem_path']).read_text())
source=Path('/home/hayden/projects/wethervane-redesign-20261006/integration/research/senate-stack-integration-20261020/tx-disjoint-calibration-sol-v1/attempt-v1/selected-disjoint-training.csv')
b=source.read_bytes();digest=hashlib.sha256(b).hexdigest();assert digest==problem['polls'][0]['covariance_lineage']['retained_training']
rows=list(csv.DictReader(b.decode().splitlines()));counts=collections.Counter(r['race_id']for r in rows);assert len(rows)==1483 and len(counts)==105
native_official_events=['2020:TX:senate:regular','2024:TX:senate:regular'];assert not set(counts)&set(native_official_events)
result=dict(schema='ActualM01CalibrationConditioningLabelInventoryV1',source=dict(path=str(source),sha256=digest,bytes=len(b)),rows=len(rows),events=len(counts),native_official_event_keys=native_official_events,intersection=[],conditioned_event_labels=[dict(event_key=k,residual_rows=v,use='fixed observed logit residual conditioning value in historical calibration law',native_marginal_factor_claim=None)for k,v in sorted(counts.items())],unknown_sample_identity_rows=sum(r['sample_overlap_status'].startswith('unknown')for r in rows),scope='Pinned source-row/event provenance only; no original official-count byte/vintage authentication, respondent identity, native calibration token or independence admission. Polling posterior bank is already SHA-bound separately; no re-fit.')
(HERE/'calibration-label-ledger.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=len(rows),events=len(counts),native_official_intersection=0)))
