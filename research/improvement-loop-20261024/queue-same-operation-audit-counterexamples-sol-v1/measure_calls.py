"""Count existing calls; do NOT memoize or skip validation. One actual tiny history."""
import json,os
from collections import Counter
from unittest.mock import patch
import execution as ex
import experiment_queue as q
import test_experiment_queue as fixtures
fx=fixtures.QueueTests(methodName='runTest');fx.setUp();fd=None
try:
 fx.admit([fx.job('measure')]);fx.run_queue();_,fd=q.dispatcher(fx.root)
 decode=ex.decode;pin=ex.pin;seen=Counter();calls=[];pin_calls=[]
 def observed_decode(data):
  seen[(type(data).__name__,data)]+=1;calls.append(len(data));return decode(data)
 def observed_pin(path):
  item=pin(path);pin_calls.append(item['size']);return item
 with patch.object(ex,'decode',side_effect=observed_decode),patch.object(ex,'pin',side_effect=observed_pin):
  first=ex.ownership_audit(fx.root/'executions',q.ownership_expected(fx.root));second=ex.ownership_audit(fx.root/'executions',q.ownership_expected(fx.root))
 assert first==second
 print(json.dumps({'scope':'two original full audits, continuously held D; pure instrumentation, no cache','real_tiny_jobs':1,'audits':2,'execution_decode_calls':len(calls),'distinct_exact_decode_inputs':len(seen),'duplicate_decode_calls':len(calls)-len(seen),'decode_input_bytes_including_repeats':sum(calls),'execution_pin_calls':len(pin_calls),'execution_pin_reported_bytes_including_repeats':sum(pin_calls),'counts_not_walltime_or_scale_benchmark':True,'fresh_reads_hashes_validation_all_retained':True},indent=2))
finally:
 if fd is not None:os.close(fd)
 fx.doCleanups()
