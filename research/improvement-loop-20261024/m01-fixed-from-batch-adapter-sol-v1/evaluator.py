"""Owned-call numerical receipt helper. Production engine is made only by gated child.

Private factory injection exists for invented unit fixtures, not a CLI execution path.
Trusted local code/process provenance, not adversarial authentication or a normalizer.
"""
import dataclasses,json,math,hashlib
from common import canonical,digest,need
@dataclasses.dataclass(frozen=True)
class Point:
 layout_bytes:bytes
 values_bytes:bytes

def point(layout,values):
 import numpy as np
 need([row['name']for row in layout]==list(values),'exact point variable order required');encoded=[]
 for row in layout:
  a=np.asarray(values[row['name']]);need(str(a.dtype)==row['dtype']=='float64'and list(a.shape)==row['shape']and a.size==row['size']and np.isfinite(a).all(),'point shape/dtype/support mismatch');encoded.append(a.reshape(-1).tolist())
 return Point(canonical(layout),canonical(encoded))
def materialize(p):
 import numpy as np
 need(type(p)is Point and type(p.layout_bytes)is bytes and type(p.values_bytes)is bytes,'immutable point required');layout=json.loads(p.layout_bytes);values=json.loads(p.values_bytes);need(len(layout)==len(values),'point values missing');out=[]
 for row,data in zip(layout,values):
  a=np.array(data,dtype=np.float64).reshape(row['shape']);need(a.size==row['size']and np.isfinite(a).all(),'point nonfinite/shape');a.flags.writeable=False;out.append(a)
 need(point(layout,{row['name']:a for row,a in zip(layout,out)})==p,'point canonical mismatch');return tuple(out)
def _numeric(value):
 if isinstance(value,(list,tuple)):return [_numeric(x)for x in value]
 if isinstance(value,dict):return {str(k):_numeric(v)for k,v in value.items()}
 v=float(value);need(math.isfinite(v),'nonfinite numerical output');return v
class _OwnedSession:
 def __init__(self,bindings,engine,provenance):
  self.__bindings=bindings;self.__engine=engine;self.__provenance=canonical(provenance);self.__sequence=0
 def evaluate(self,bound,p):
  import numpy as np
  b=self.__bindings.revalidate(bound);point_snapshot=(p.layout_bytes,p.values_bytes);inputs=materialize(p);exposure=np.array(b.exposure,dtype=np.float64);exposure.flags.writeable=False;before=digest({'E':exposure.tolist(),'values':[a.tolist()for a in inputs]});kernel=json.loads(self.__provenance)
  # Engine call occurs here; the caller cannot supply a numerical result or output envelope.
  output=_numeric(self.__engine(exposure,inputs));need(set(output)=={'logp','poll_logp','factor_values','rates','physical'},'complete output roster required')
  need(abs(sum(float(np.sum(a))for a in output['factor_values'])-output['logp'])<=1e-6,'full factor sum mismatch');need(all(np.all(np.asarray(v)>0)for v in output['rates'].values())and all(np.all(np.asarray(v)>0)for v in output['physical'].values()),'physical/rate support failure')
  need(before==digest({'E':exposure.tolist(),'values':[a.tolist()for a in inputs]}),'engine mutated owned inputs');self.__bindings.revalidate(bound);need((p.layout_bytes,p.values_bytes)==point_snapshot,'point mutated during engine call');self.__sequence+=1
  receipt={'schema':'OwnedParentKernelCallV1','sequence':self.__sequence,'provenance':kernel,'kernel_receipt_sha256':hashlib.sha256(self.__provenance).hexdigest(),'parent_record':json.loads(b.record_bytes),'parent_envelope':json.loads(b.envelope_bytes),'exposure':exposure.tolist(),'input_exposure_sha256':digest(exposure.tolist()),'point':{'layout':json.loads(p.layout_bytes),'values':json.loads(p.values_bytes)},'point_sha256':digest({'layout':json.loads(p.layout_bytes),'values':json.loads(p.values_bytes)}),'output':output,'conditional_normalization':'unverified','outer_mass':json.loads(b.envelope_bytes)['mass'],'population_cut':'mass outside conditional logp; no downstream reweighting','execution_scope':kernel['execution_scope']}
  return canonical(receipt)
def _session(bindings,engine,provenance):return _OwnedSession(bindings,engine,provenance)

def verify(raw,expected_sha256,bindings):
 """Needs caller-retained output digest. A discovered/self-authored hash proves no origin."""
 need(type(raw)is bytes and hashlib.sha256(raw).hexdigest()==expected_sha256,'retained call digest mismatch');r=json.loads(raw);need(raw==canonical(r),'canonical receipt required');b=bindings.bind(r['parent_record']);need(r['parent_envelope']==json.loads(b.envelope_bytes)and r['exposure']==[list(row)for row in b.exposure]and r['input_exposure_sha256']==digest(r['exposure']),'parent/exposure attribution mismatch');need(r['outer_mass']==.125 and r['conditional_normalization']=='unverified','protected cut mismatch');p=Point(canonical(r['point']['layout']),canonical(r['point']['values']));materialize(p);need(r['point_sha256']==digest(r['point']),'point digest mismatch');need(r['kernel_receipt_sha256']==digest(r['provenance']),'kernel receipt mismatch');return r
