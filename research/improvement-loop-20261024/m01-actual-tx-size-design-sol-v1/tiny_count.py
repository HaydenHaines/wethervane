"""Invented-object counting and per-write durability observation only. No TX entrypoint."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

class BudgetExceeded(ValueError):pass

class Counter:
 def __init__(self,budget,observe):
  if type(budget)is not int or budget<0:raise ValueError('nonnegative budget required')
  self.budget=budget;self.accepted=0;self.attempted=0;self.calls=0;self.hash=hashlib.sha256();self.observe=observe
 def snapshot(self):return dict(accepted_bytes=self.accepted,attempted_end=self.attempted,write_calls=self.calls,accepted_prefix_sha256=self.hash.hexdigest())
 def write(self,data):
  view=memoryview(data).cast('B');self.calls+=1;self.attempted=self.accepted+view.nbytes
  if self.attempted>self.budget:self.observe(self.snapshot());raise BudgetExceeded('logical prefix write exceeds cap')
  for offset in range(0,view.nbytes,65536):self.hash.update(view[offset:offset+65536])
  self.accepted+=view.nbytes;self.observe(self.snapshot());return view.nbytes

def invented(size):
 if type(size)is not int or not 0<=size<=65536:raise ValueError('tiny invented fixture only')
 shared=bytearray((i%251 for i in range(size)))
 return dict(schema='InventedSharedRootV1',original_cost=shared,replaced_cost=shared,gradient=[shared,7])

def count_fixture(root,size=1024,budget=1024**2,durable=True):
 root=Path(root);root.mkdir(exist_ok=False);stats=dict(progress_updates=0,fsync_calls=0,progress_io_seconds=0.0)
 def progress(value):
  if not durable:return
  start=time.perf_counter();temp=root/'progress.next'
  with temp.open('xb') as f:
   f.write((json.dumps(value,sort_keys=True)+'\n').encode());f.flush();os.fsync(f.fileno());stats['fsync_calls']+=1
  os.replace(temp,root/'progress.json');fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd);stats['fsync_calls']+=1
  finally:os.close(fd)
  stats['progress_updates']+=1;stats['progress_io_seconds']+=time.perf_counter()-start
 counter=Counter(budget,progress);start=time.perf_counter()
 try:
  pickle.Pickler(counter,protocol=5).dump(invented(size));status='tiny_pickler_returned'
 except BudgetExceeded:status='tiny_prefix_capped_not_complete'
 except BaseException as error:
  (root/'failure.json').write_text(json.dumps(dict(status='toy_failure_no_success_receipt',error=repr(error),counter=counter.snapshot())))
  raise
 return dict(schema='InventedSizeDesignObservationV1',status=status,counter=counter.snapshot(),stats=stats,seconds=time.perf_counter()-start,
  payload_persisted=False,payload_loaded=False,actual_tx=False,public_completion_receipt=False)
