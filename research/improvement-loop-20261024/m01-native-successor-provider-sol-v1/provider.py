"""Source-pinned lab-only M01 measurement provider. No sampler/registration side effects."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
OWNER=Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
NATIVE=OWNER/'research/senate-stack-integration-20261020/modular-count-graph-astra-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path,package=False):
 if name in sys.modules:return sys.modules[name]
 spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(Path(path).parent)]if package else None)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
# Verify all imported native/helper source bytes before imports, including selection itself.
SELECTION=HERE/'selection.json'
EXPECTED_SELECTION_SHA256='03edb47049384b662cdd645053fc6083bde8fb41d972b1a07da8b71ecf6283b5'
def parse_selection(raw):
 if hashlib.sha256(raw).hexdigest()!=EXPECTED_SELECTION_SHA256:raise ValueError('externally frozen lab selection changed')
 return json.loads(raw)
def selection():
 s=parse_selection(SELECTION.read_bytes())
 for p,h in s['pins'].items():
  if sha(p)!=h:raise ValueError('source/input hash changed: '+p)
 return s
S=selection();sys.path.insert(0,str(OWNER))
native=load('m01_successor_native',NATIVE/'package.py',True)
graph=sys.modules['m01_successor_native.graph'];components=sys.modules['m01_successor_native.components']
prototype=load('m01_successor_symbolic',LAB/'m01-count-measurement-prototype-sol-v1/prototype.py')
def validate(problem,parent,s):
 if graph.digest(problem)!=s['problem_identity']:raise ValueError('complete Problem/candidate/order/block/variance mismatch')
 if parent not in s['parents']:raise ValueError('protected parent state/mass changed')
 if len(problem['polls'])!=1 or len(problem['polls'][0]['rows'])!=6:raise ValueError('one complete six-row block required')
 ids=[r['observation_id']for r in problem['polls'][0]['rows']]
 if ids!=s['model_ids']or len(set(ids))!=6:raise ValueError('complete vector identity mismatch')
 if s['variance_owner']!='single_total_error_bank_no_added_sampling_or_movement':raise ValueError('variance ownership unsupported')
 if s['target']!='marginal_bayes_within_population_cut':raise ValueError('feedback target unsupported')
 if [p['mass']for p in s['parents']]!=[.125]*8:raise ValueError('original parent masses required')
def mixture(mode):
 s=selection();cov=np.asarray(s['incumbent_covariance'])
 if mode=='incumbent_reduction':return prototype.SharedVectorMixture(cov[None],[1.])
 if mode!='actual_calibration_bank':raise ValueError('unsupported supplied-state mode')
 consumer=LAB/'calibration-consumer-v1';coords=json.loads((consumer/'inputs/target-coordinates.json').read_text())
 cy=np.array(coords['cycles']);ra=np.array(coords['race_ids']);ho=np.array(coords['pollsters']);d=np.array(coords['days_to_election']);sc=cy[:,None]==cy;sr=(ra[:,None]==ra)&sc
 kernels=np.array([np.eye(6),sc,sr,(ho[:,None]==ho)&sc,sr*np.minimum.outer(d,d)/30],float)
 with np.load(consumer/'inputs/posterior-bank.npz',allow_pickle=False)as bank:scales=bank['standard_deviations'].reshape(4000,5)
 return prototype.SharedVectorMixture(np.einsum('sk,kij->sij',scales**2,kernels),np.full(4000,1/4000))
class M01Measurement(components.Measurement):
 implementation_id='lab_M01_complete_vector_marginal_measurement_v1'
 def __init__(self,mode='actual_calibration_bank'):
  if mode not in ('actual_calibration_bank','incumbent_reduction'):raise ValueError('unsupported mode')
  super().__init__()
  deps=tuple(sorted([(str(SELECTION),sha(SELECTION)),*S['pins'].items()]))
  self.spec=dataclasses.replace(self.spec,configuration_json=json.dumps(dict(mode=mode,selection_sha256=sha(SELECTION),computational_only=True)),dependency_pins=deps)
 def observations(self,c,scope):
  import pymc as pm
  import pytensor.tensor as pt
  s=selection();validate(dict_plain(c.data),dict_plain(c.protected),s)
  # Native method unmodified. Only its final poll loop is empty in copied context.
  data=dict(c.data);data['polls']=()
  components.Measurement.observations(self,dataclasses.replace(c,data=data),scope)
  block=c.data['polls'][0];means=[]
  for row in block['rows']:
   names=[k['id']for k in c.contest_map[row['contest_id']]['candidates']]
   first,second=[names.index(row[k])for k in ('first_candidate_id','second_candidate_id')]
   rates=scope.inputs['category_rates'][row['contest_id']]
   means.append(pt.log(rates[first])-pt.log(rates[second]))
  name='M01_complete_poll_vector'
  scope.factor(name,['poll:'+r['observation_id']for r in block['rows']])
  pm.Potential(name,mixture(self.spec.configuration['mode']).symbolic_logpdf(np.array([r['logit_value']for r in block['rows']]),pt.stack(means)))
  return {}
def dict_plain(v):
 from collections.abc import Mapping
 if isinstance(v,Mapping):return {k:dict_plain(w)for k,w in v.items()}
 if isinstance(v,(list,tuple)):return [dict_plain(w)for w in v]
 if isinstance(v,np.ndarray):return v.tolist()
 return v
