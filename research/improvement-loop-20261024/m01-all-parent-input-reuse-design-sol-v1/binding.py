"""Authenticated whole-parent input/envelope only. No model, density or normalizer."""
import copy,hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPECTED_MANIFEST='2fe2801107c96ab16c7d5d1134bb4ea8c4ffc4c19aff9853b38c2b5e2d8eb000'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load():
 need(sha(HERE/'parents-manifest.json')==EXPECTED_MANIFEST,'frozen parent manifest changed');m=json.loads((HERE/'parents-manifest.json').read_text())
 for path,expected in m['pins'].items():need(sha(path)==expected,'source changed '+path)
 return m

def bind(record):
 """Reject merely equal exposure, modified mass, mixed rows or changed axes/lineage."""
 m=load();identity=digest(record);matches=[p for p in m['parents']if p['parent_identity']==identity]
 need(len(matches)==1 and canonical(record)==canonical(matches[0]['record']),'complete authenticated parent required')
 p=matches[0];need(record['mass']==.125,'original fixed mass required')
 exposure=tuple(tuple(float(value)for value in row)for row in record['exposure']);need(len(exposure)==3 and all(len(row)==24 for row in exposure),'complete event/cell exposure required');need(all(math.isfinite(v)and v>0 for row in exposure for v in row),'positive finite exposure required')
 envelope={'parent_identity':identity,'parent_id':record['parent_id'],'row_id':record['row_id'],'upstream_identity':record['upstream_identity'],'problem_identity':record['problem_identity'],'mass':record['mass'],'reference_fixed_parent_target_id':p['reference_fixed_parent_target_id'],'reference_target_status':'formula reference only; not a compiled/switched contract','kernel_identity':None,'conditional_normalization':'unverified','parent_mass_used_in_logp':False,'inference_target':'population_cut_v1','numerical_input_sha256':digest(record['exposure'])}
 return exposure,envelope
