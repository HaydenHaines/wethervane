"""Prepared compressed M01 measurement successor; no actual full graph built here."""
import ast,dataclasses,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
EXPECTED_SELECTION='21d2c8daa78dae5f045fc8287051e472c5ba6d9e2b0ea6c83a3cb4e5da35f661'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def selected():
 raw=(HERE/'selection.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==EXPECTED_SELECTION,'frozen compressed selection changed');s=json.loads(raw)
 for path,digest in s['pins'].items():need(sha(path)==digest,'source/artifact changed: '+path)
 return s
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
S=selected();sys.path.append(str(LAB/'m01-native-successor-provider-sol-v1/schema-overlay'))
p=load('compressed_frozen_M01_provider',LAB/'m01-native-successor-provider-sol-v1/provider.py')
compression=load('compressed_frozen_scalar_math',LAB/'m01-exact-vector-compression-sol-v1/compression.py')

def native_mapping():
 """Compile the unmodified native poll means prefix; source hash selected first."""
 import pytensor.tensor as pt
 source=Path(S['native_components']);tree=ast.parse(source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='Measurement');method=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='observations')
 loop=next(x for x in method.body if isinstance(x,ast.For)and isinstance(x.target,ast.Name)and x.target.id=='block')
 prefix=loop.body[:2];need(isinstance(prefix[0],ast.Assign)and isinstance(prefix[1],ast.For),'native poll mapping layout changed')
 signature=hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest();need(signature==S['mapping_ast_sha256'],'exact native mapping AST changed')
 fn=ast.FunctionDef(name='exact_native_poll_means',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n)for n in ['c','v','block']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[*prefix,ast.parse('return pt.stack(means)').body[0]],decorator_list=[])
 tree=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));ns={'pt':pt};exec(compile(tree,str(source),'exec'),ns);return ns['exact_native_poll_means']
map_means=native_mapping()

def exact_symmetric(covariances):
 a=np.asarray(covariances,dtype=float);need(a.ndim==3 and a.shape[-2:]==(6,6),'complete six-poll covariance required');need(np.isfinite(a).all()and np.array_equal(a,a.transpose(0,2,1)),'exact symmetry required; no tolerated triangle ambiguity');return a

def compact(mode):
 s=selected();old=p.selection();block=s['poll_block'];y=np.array([r['logit_value']for r in block['rows']])
 need(np.array_equal(compression.pair_direction(block),np.ones(6)),'source-bound common direction required')
 if mode=='incumbent_reduction':
  cov=exact_symmetric(np.array(old['incumbent_covariance'])[None]);return compression.AffineScalarMixture(cov,[1.],y,np.ones(6),np.zeros(6))
 need(mode=='actual_calibration_bank','both original modes only')
 with np.load(s['reference_bank'],allow_pickle=False)as saved:cov=exact_symmetric(saved['covariances']);weights=saved['weights'].copy()
 need(cov.shape==(4000,6,6)and weights.shape==(4000,),'complete bank required')
 with np.load(s['coefficients'],allow_pickle=False)as saved:
  need(set(saved.files)=={'precision','center','base'},'exact coefficient fields required');arrays={n:saved[n].copy()for n in saved.files}
 need(all(a.shape==(4000,)for a in arrays.values())and np.isfinite(arrays['precision']).all()and(arrays['precision']>0).all()and np.isfinite(arrays['center']).all()and np.isfinite(arrays['base']).all(),'finite complete frozen coefficients required')
 result=object.__new__(compression.AffineScalarMixture)
 for name,array in arrays.items():object.__setattr__(result,name,compression.frozen(array))
 return result

class CompressedM01(p.M01Measurement):
 implementation_id='lab_M01_exact_scalar_restriction_measurement_v1'
 def __init__(self,mode='actual_calibration_bank'):
  need(mode in S['modes'],'both declared original modes required');super().__init__(mode)
  deps=dict(self.spec.dependency_pins);deps.update(S['pins']);deps[str(HERE/'selection.json')]=sha(HERE/'selection.json')
  config=dict(self.spec.configuration,compressed_selection_sha256=EXPECTED_SELECTION,exact_mapping_ast_sha256=S['mapping_ast_sha256'],fixed_offset='six zeros',affine_direction='six ones',symmetry='exact_only',computational_only=True)
  self.spec=dataclasses.replace(self.spec,configuration_json=json.dumps(config),dependency_pins=tuple(sorted(deps.items())))
 def observations(self,c,scope):
  import dataclasses
  import pymc as pm
  import pytensor.tensor as pt
  from pytensor.raise_op import CheckAndRaise
  s=selected();p.validate(p.dict_plain(c.data),p.dict_plain(c.protected),p.selection());need(p.dict_plain(c.data['polls'])==[s['poll_block']],'six observations/order/offsets changed')
  need(p.dict_plain(c.contest_map)=={row['id']:row for row in p.dict_plain(c.data['contests'])},'contest/candidate/rate references differ from frozen Problem')
  data=dict(c.data);data['polls']=();p.components.Measurement.observations(self,dataclasses.replace(c,data=data),scope)
  block=c.data['polls'][0];means=map_means(c,scope.inputs,block);mean=CheckAndRaise(ValueError,'all six source-bound mean references must coincide')(means[0],pt.all(pt.eq(means,means[0])))
  scope.factor('M01_complete_poll_vector',['poll:'+r['observation_id']for r in block['rows']])
  pm.Potential('M01_complete_poll_vector',compact(self.spec.configuration['mode']).symbolic_logpdf(mean));return {}
