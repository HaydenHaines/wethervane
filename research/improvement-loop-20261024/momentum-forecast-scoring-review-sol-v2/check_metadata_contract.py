"""Actual JSON pin maps only; deliberately never call an NPZ/label loader."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;B=HERE.parent;P=B/'momentum-forecast-scoring-sol-v2'
sys.path[:0]=[str(P),str(P/'snapshots')]
import score_reader as r
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((HERE/'actual-metadata-contract.json').read_text());checked=0
for row in audit['jobs']:
 tp=Path(row['terminal']['path']);assert sha(tp)==row['terminal']['sha256'];t=json.loads(tp.read_text())
 mp=Path(row['trace_metadata']['path']);assert sha(mp)==row['trace_metadata']['sha256'];m=json.loads(mp.read_text())
 caller=t['provenance']['trace_source_pins'];expected=r.expected_trace_source_pins(caller)
 assert expected==m['source_pins'] and len(expected)==20 and len(caller)==16
 bad={**caller,'forecast_design':'0'*64}
 try:r.expected_trace_source_pins(bad)
 except r.ArtifactHardFailure:pass
 else:raise AssertionError('Conflicting required pin accepted')
 # Extra/omitted caller pins cannot be satisfied by unchanged stored metadata.
 assert r.expected_trace_source_pins({**caller,'invented_unrecorded':'a'*64})!=m['source_pins']
 omitted=dict(caller);omitted.pop(next(iter(omitted)))
 assert r.expected_trace_source_pins(omitted)!=m['source_pins']
 checked+=1
out=dict(schema='IndependentActualMetadataV2RepairCheckSolV1',metadata_only=True,jobs_checked=checked,all_exact_fixed_plus_caller_unions=True,conflicting_fixed_pin_rejected=True,omitted_or_added_caller_not_subset_accepted=True,numerical_arrays_opened=False,target_labels_opened=False)
(HERE/'metadata-repair-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
