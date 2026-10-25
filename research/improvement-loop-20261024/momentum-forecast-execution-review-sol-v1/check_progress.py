"""Synthetic observer-record replay and independent persisted delta reconstruction."""
from pathlib import Path
import importlib.util,json,sys,tempfile
import numpy as np
HERE=Path(__file__).resolve().parent;B=HERE.parent
sys.path.insert(0,str(B/'momentum-forecast-execution-sol-v1'))
import runner
spec=importlib.util.spec_from_file_location('trace_fixture',B/'momentum-trace-diagnostics-sol-v1/test_trace_diagnostics.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
prior,_,_,records,_=t.synthetic()
sink=t.td.RetainedTraceCollector(chains=4,draws=10,race_ids=prior.race_ids,observed_race_indices=[0,1],missing_race_indices=[2],posterior_source_chain=prior.posterior_chain,posterior_source_draw=prior.posterior_draw,source_identity=prior.source_identity,source_pins={**t.td.REQUIRED_SOURCE_PINS,'synthetic':'a'*64},batch_length=100)
reconstructed={};receipts=0
with tempfile.TemporaryDirectory() as tmp:
 observer=runner.ProgressObserver(sink,tmp,every=4)
 for record in records:observer(record)
 expected=sink.export_partial()
 for item in observer.receipts:
  receipt=json.loads(Path(item['path']).read_text());meta=json.loads(Path(receipt['metadata']['path']).read_text())
  assert meta['observer_phase_prefix_only'] and not meta['raw_predictive_draw_completeness_claimed']
  with np.load(receipt['arrays']['file']['path'],allow_pickle=False) as archive:
   for name in archive.files:
    value=archive[name]
    if not name.startswith('chain'):
     if name in reconstructed:assert np.array_equal(reconstructed[name],value)
     reconstructed[name]=value.copy();continue
    head,field=name.split('__',1);c=int(head[5:]);start=meta['delta_start_completed_records_per_chain'][c];stop=meta['delta_stop_completed_records_per_chain'][c]
    if field in {'bank_batch_weight_sums','bank_batch_counts'}:
     offset=start//100;end=(stop+99)//100
     if start==stop:
      assert value.shape[1]==0
      if name not in reconstructed:reconstructed[name]=value.copy()
      continue
     if name not in reconstructed:reconstructed[name]=np.empty((1,0,*value.shape[2:]),dtype=value.dtype)
     old=reconstructed[name]
     target=np.empty((1,max(old.shape[1],end),*value.shape[2:]),dtype=value.dtype)
     target[:,:old.shape[1]]=old;target[:,offset:end]=value;reconstructed[name]=target
    elif field in {'forecast_chain_index','bank_sampled_occupancy'}:reconstructed[name]=value.copy()
    else:
     old=reconstructed.get(name)
     assert value.shape[1]==stop-start
     if old is not None:assert old.shape[1]==start
     reconstructed[name]=value.copy() if old is None else np.concatenate((old,value),axis=1)
  receipts+=1
 for name,value in expected['arrays'].items():assert np.array_equal(reconstructed[name],value),name
 assert len(meta['final_rng_states_emitted'])==4
out=dict(schema='IndependentProgressDeltaReconstructionSolV1',synthetic_only=True,emitted_receipts=receipts,every=4,collector_batch_length=100,complete_observer_prefix_arrays_reconstructed=True,bank_batches_replaced_by_position_not_double_summed=True,raw_predictive_completeness_claimed=False,actual_target_inputs=False)
(HERE/'progress-checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
