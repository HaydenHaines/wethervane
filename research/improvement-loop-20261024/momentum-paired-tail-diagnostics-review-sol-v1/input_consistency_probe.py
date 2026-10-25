"""Read-only synthetic counterexample to the sealed v1 input contract."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr
HERE=Path(__file__).resolve().parent;B=HERE.parent;P=B/'momentum-paired-tail-diagnostics-sol-v1'
sys.path[:0]=[str(P),str(B/'momentum-trace-diagnostics-sol-v1'),str(B/'momentum-trace-diagnostics-sol-v1/snapshots')]
import paired_tail as p
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(P/'artifact-manifest.json')=='522b89bcb9629805735109f86da781451d77463ff978ddfa082df48677426603'
for row in json.loads((P/'artifact-manifest.json').read_text())['files']:assert sha(P/row['path'])==row['sha256']
z=np.linspace(9,10,6400).reshape(4,1600,1);win,lose=log_ndtr(z),log_ndtr(-z)
bad=p.paired_tail_diagnostics(win,lose+.1,['synthetic'])
assert bad['lose']['coordinates'][0]['status']=='measured'
discrepancy=float(np.max(abs((lose+.1)-np.log(-np.expm1(win)))))
z=np.linspace(-40,40,100001);w,l=log_ndtr(z),log_ndtr(-z)
large,small=np.maximum(w,l),np.minimum(w,l);mask=(-large>=np.finfo(float).tiny)&(large<0)
error=float(np.max(abs(small[mask]-np.log(-np.expm1(large[mask])))))
out=dict(schema='IndependentPairedTailV1ConsistencyCounterexample',producer_manifest_sha256=sha(P/'artifact-manifest.json'),synthetic_only=True,contradictory_small_log_tail_shift=.1,accepted_status=bad['lose']['coordinates'][0]['status'],recoverable_log_discrepancy=discrepancy,ordinary_negative_dominant_log=True,native_gaussian_grid_cases=int(mask.sum()),native_gaussian_grid_max_error=error,required_disposition='repair_input_guard_before_runner_acceptance')
(HERE/'input-consistency-probe.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
