"""One tiny real history; two full audits/phase, aggregate counters only."""
import json,os,time,sys
from collections import Counter
from contextlib import ExitStack,nullcontext
from unittest.mock import patch
import execution as ex
import experiment_queue as q
import test_experiment_queue as fixtures
fx=fixtures.QueueTests(methodName='runTest');fx.setUp();fd=None
try:
 fx.admit([fx.job('measure')]);fx.run_queue();_,fd=q.dispatcher(fx.root);results=[];snapshots=[];pin_sequences=[];checked_sequences=[]
 phases=['off-before','on','off-after']if hasattr(ex,'decode_operation')else['original-off']
 for phase in phases:
  decode=ex.decode;base=getattr(ex,'_base_decode',decode);pin=ex.pin;checked=ex.checked;decode_calls=[];parses=[];pins=[];checks=[]
  def parse(data):parses.append(len(data));return base(data)
  def observe_decode(data):
   decode_calls.append(len(data))
   if hasattr(ex,'_base_decode'):return decode(data)
   return parse(data)
  def observe_pin(path):
   item=pin(path);pins.append((str(path),item));return item
  def observe_checked(item):
   value=checked(item);checks.append((item,len(value)));return value
  with ExitStack()as stack:
   stack.enter_context(patch.object(ex,'decode',side_effect=observe_decode));stack.enter_context(patch.object(ex,'pin',side_effect=observe_pin));stack.enter_context(patch.object(ex,'checked',side_effect=observe_checked))
   if hasattr(ex,'_base_decode'):stack.enter_context(patch.object(ex,'_base_decode',side_effect=parse))
   memo=stack.enter_context(ex.decode_operation())if phase=='on'else None
   started=time.perf_counter()
   first=ex.ownership_audit(fx.root/'executions',q.ownership_expected(fx.root));second=ex.ownership_audit(fx.root/'executions',q.ownership_expected(fx.root));elapsed=time.perf_counter()-started
   assert first==second
  snapshots.append(first);pin_sequences.append(pins);checked_sequences.append(checks)
  results.append({'phase':phase,'elapsed_seconds_instrumented':elapsed,'decode_calls':len(decode_calls),'base_parse_calls':len(parses),'decode_input_bytes':sum(decode_calls),'base_parse_input_bytes':sum(parses),'execution_pin_calls':len(pins),'execution_pin_reported_bytes':sum(x[1]['size']for x in pins),'execution_checked_calls':len(checks),'execution_checked_bytes':sum(x[1]for x in checks),'memo_stats_after_exit':memo.stats if memo is not None else None})
 assert all(x==snapshots[0]for x in snapshots);assert all(x==pin_sequences[0]for x in pin_sequences);assert all(x==checked_sequences[0]for x in checked_sequences)
 print(json.dumps({'real_jobs':1,'audits_per_phase':2,'phases':results,'within_store_snapshots_identical':True,'within_store_pin_sequence_identical':True,'within_store_checked_sequence_identical':True,'scope':'instrumented tiny histories, not throughput/scaling or hard RSS; separate original source stores differ in bytes/path'},indent=2))
finally:
 if fd is not None:os.close(fd)
 fx.doCleanups()
