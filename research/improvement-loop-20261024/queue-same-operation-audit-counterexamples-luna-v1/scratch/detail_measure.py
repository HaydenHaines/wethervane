import json,hashlib,os
from collections import Counter
from unittest.mock import patch
import execution as ex,experiment_queue as q,test_experiment_queue as fixtures
fx=fixtures.QueueTests(methodName='runTest');fx.setUp();fd=None
try:
 fx.admit([fx.job('measure-detail')]);fx.run_queue();_,fd=q.dispatcher(fx.root)
 decode,pin=ex.decode,ex.pin;ds=Counter();ph=Counter()
 def d(x):ds[(type(x).__name__,len(x),hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest())]+=1;return decode(x)
 def pn(x):item=pin(x);ph[item['size']]+=1;return item
 with patch.object(ex,'decode',side_effect=d),patch.object(ex,'pin',side_effect=pn):
  a=ex.ownership_audit(fx.root/'executions',q.ownership_expected(fx.root));b=ex.ownership_audit(fx.root/'executions',q.ownership_expected(fx.root))
 assert a==b
 print(json.dumps({'decode_calls':sum(ds.values()),'unique_exact':len(ds),'decode_lengths_and_counts':[{"type":k[0],"bytes":k[1],"sha256":k[2],"calls":n} for k,n in sorted(ds.items())], 'pin_calls':sum(ph.values()),'pin_size_counts':dict(sorted(ph.items()))},indent=2))
finally:
 if fd is not None:os.close(fd)
 fx.doCleanups()
