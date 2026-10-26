"""One protocol-five memo; bytes are hashed/counted then discarded, never loaded."""
import hashlib,pickle,time
class LogicalCap(ValueError):pass
class Sink:
 def __init__(self,budget,progress):
  if type(budget)is not int or not 0<budget<=512*1024**2:raise ValueError('positive bounded logical stream budget required')
  self.budget=budget;self.progress=progress;self.accepted=0;self.calls=0;self.attempted=0;self.h=hashlib.sha256();self.io_seconds=0.
 def state(self):return dict(accepted_bytes=self.accepted,attempted_end=self.attempted,write_calls=self.calls,accepted_prefix_sha256=self.h.hexdigest(),progress_io_seconds=self.io_seconds)
 def write(self,data):
  view=memoryview(data).cast('B');self.calls+=1;self.attempted=self.accepted+view.nbytes
  if self.attempted>self.budget:
   self.progress(dict(status='logical_cap_before_write',**self.state()));raise LogicalCap('attempted prefix exceeds logical cap; complete stream unknown')
  for start in range(0,view.nbytes,65536):self.h.update(view[start:start+65536])
  self.accepted+=view.nbytes;start=time.monotonic();self.progress(dict(status='counting',**self.state()));self.io_seconds+=time.monotonic()-start
  return view.nbytes

def count_bundle(bundle,writer,budget):
 sink=Sink(budget,lambda value:writer.json('progress.json',value));start=time.monotonic()
 pickle.Pickler(sink,protocol=5).dump(bundle)
 return dict(status='single_pickler_returned',protocol=5,one_memo=True,buffer_callback=False,**sink.state(),seconds=time.monotonic()-start,
             pickle_payload_persisted=False,pickle_loaded=False,restoration_verified=False)
