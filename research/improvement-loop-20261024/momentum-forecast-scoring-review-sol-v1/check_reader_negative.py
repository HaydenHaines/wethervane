"""Coherently rebound synthetic truncation must fail before invented label I/O."""
from pathlib import Path
from unittest.mock import patch
import sys,tempfile,json
import numpy as np
HERE=Path(__file__).resolve().parent;P=HERE.parent/'momentum-forecast-scoring-sol-v1'
sys.path[:0]=[str(P),str(P/'snapshots')]
import score_reader as r
import synthetic_files as f
from trace_diagnostics import save_trace
opens=[];checked=r.checked_bytes
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);pin,admission,_=f.make_bundle(root)
 fold=admission['selected_folds'][0];record=fold['jobs']['FS5']
 terminal=r.json_pin(record['terminal']);seal=r.json_pin(record['prediction_seal'])
 oldtrace=terminal['artifacts']['trace']
 trace=r.load_trace(Path(oldtrace['path']).parent,expected_source_pins=terminal['provenance']['trace_source_pins'],expected_receipt_sha256=oldtrace['sha256'])
 n=1400;arrays={k:v.copy() for k,v in trace['arrays'].items()};meta=dict(trace['metadata']);meta['draws']=n
 for key,value in list(arrays.items()):
  if value.shape[:2]==(4,1600):arrays[key]=value[:,:n].copy()
 arrays['bank_batch_weight_sums']=arrays['bank_batch_weight_sums'][:,:14]
 arrays['bank_batch_counts']=arrays['bank_batch_counts'][:,:14]
 for c in range(4):arrays['bank_sampled_occupancy'][c]=np.bincount(arrays['sampled_row_index'][c],minlength=12)
 saved=save_trace(dict(arrays=arrays,metadata=meta),root/'FS5/truncated-trace')
 newtrace=dict(path=str(root/'FS5/truncated-trace/receipt.json'),sha256=saved['receipt_sha256'])
 rawrec=r.json_pin(terminal['artifacts']['original']);raw=r.npz_record(rawrec['arrays'])
 raw={k:(v[:,:n] if v.shape[:2]==(4,1600) else v) for k,v in raw.items()}
 rawrec['arrays']=f.nz(root/'FS5/truncated-original.npz',raw)
 neworiginal=f.js(root/'FS5/truncated-original-receipt.json',rawrec)
 comp=r.npz_record(terminal['artifacts']['marginal_components'])
 comp={k:(v[:,:n] if v.shape[:2]==(4,1600) else v) for k,v in comp.items()}
 newcomp=f.nz(root/'FS5/truncated-components.npz',comp)
 terminal['artifacts'].update(original=neworiginal,trace=newtrace,marginal_components=newcomp)
 seal['artifacts'].update(original=neworiginal,trace=newtrace)
 record['terminal']=f.js(root/'FS5/rebound-terminal.json',terminal)
 record['prediction_seal']=f.js(root/'FS5/rebound-seal.json',seal)
 admission_pin=f.js(root/'rebound-admission.json',admission)
 def spy(pin):
  if str(pin['path']).endswith('SYNTHETIC-outcomes.csv'):opens.append('labels')
  return checked(pin)
 with patch.object(r,'checked_bytes',spy):
  try:r.verify_admission(admission_pin)
  except r.ArtifactHardFailure as error:
   assert 'chain-draw budget mismatch' in str(error),str(error)
   reason=str(error)
  else:raise AssertionError('Coherent truncation accepted')
 assert not opens
out=dict(schema='IndependentSyntheticCoherentTruncationCheckSolV1',synthetic_only=True,source_traces_and_originals_and_components_rebound_coherently=True,job_declared_draws=1600,rebound_draws=1400,rejection_reason=reason,label_bytes_opened=0,actual_forecast_inputs=False)
(HERE/'reader-negative-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
