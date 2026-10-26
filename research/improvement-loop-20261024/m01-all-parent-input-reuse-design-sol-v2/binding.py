"""Whole-parent metadata attribution prototype; no model or evaluator authentication."""
import hashlib,json,math
from dataclasses import dataclass
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPECTED_MANIFEST='2fe2801107c96ab16c7d5d1134bb4ea8c4ffc4c19aff9853b38c2b5e2d8eb000'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load():
 need(sha(HERE/'parents-manifest.json')==EXPECTED_MANIFEST,'frozen manifest changed');m=json.loads((HERE/'parents-manifest.json').read_text())
 for path,expected in m['pins'].items():need(sha(path)==expected,'source changed '+path)
 return m
@dataclass(frozen=True)
class BoundParent:
 record_bytes:bytes
 exposure:tuple
 envelope_bytes:bytes

def _make(record):
 m=load();identity=digest(record);rows=[p for p in m['parents']if p['parent_identity']==identity]
 need(len(rows)==1 and canonical(record)==canonical(rows[0]['record']),'complete authenticated parent required');p=rows[0];need(record['mass']==.125,'fixed .125 required')
 exposure=tuple(tuple(float(x)for x in row)for row in record['exposure']);need(len(exposure)==3 and all(len(row)==24 for row in exposure),'whole3x24 exposure required');need(all(math.isfinite(x)and x>0 for row in exposure for x in row),'finite positive exposure required')
 envelope={'parent_identity':identity,'parent_id':record['parent_id'],'row_id':record['row_id'],'upstream_identity':record['upstream_identity'],'problem_identity':record['problem_identity'],'mass':record['mass'],'reference_fixed_parent_target_id':p['reference_fixed_parent_target_id'],'reference_target_status':'formula reference only; not compiled or switched contract','kernel_identity':None,'conditional_normalization':'unverified','parent_mass_used_in_logp':False,'inference_target':'population_cut_v1','numerical_input_sha256':digest(exposure)}
 return BoundParent(canonical(record),exposure,canonical(envelope))
def bind(record):return _make(record)
def revalidate(bound):
 need(type(bound)is BoundParent and type(bound.record_bytes)is bytes and type(bound.envelope_bytes)is bytes and type(bound.exposure)is tuple and all(type(row)is tuple for row in bound.exposure),'immutable bound representation required')
 expected=_make(json.loads(bound.record_bytes));need(bound==expected,'bound record/exposure/envelope changed');return expected

def seal_output(bound,*,exposure_used,envelope_claim,kernel_id,point_id,result):
 """Checks metadata immediately; cannot prove an arbitrary callable consumed this input."""
 expected=revalidate(bound)
 need(canonical(exposure_used)==canonical(expected.exposure),'evaluated exposure differs from whole parent')
 claim=envelope_claim if type(envelope_claim)is bytes else canonical(envelope_claim)
 need(claim==expected.envelope_bytes,'output parent/target attribution changed')
 need(isinstance(kernel_id,str)and bool(kernel_id)and isinstance(point_id,str)and bool(point_id),'explicit caller kernel/point identifiers required')
 # Reconstruct metadata from authenticated bytes; never copy a caller's mutable envelope.
 output={'schema':'ParentOutputAttributionPrototypeV2','receipt_scope':'metadata consistency only; no authenticated evaluator or numerical equivalence','parent_record':json.loads(expected.record_bytes),'parent_envelope':json.loads(expected.envelope_bytes),'input_exposure_sha256':digest(expected.exposure),'caller_kernel_id':kernel_id,'caller_point_id':point_id,'evaluator_authenticated':False,'actual_conditional_normalization':'unverified','result':result}
 raw=canonical(output)
 # All reference data is rechecked immediately before returning bytes for publication.
 revalidate(bound);return raw

def verify_output(raw):
 need(type(raw)is bytes,'immutable canonical output bytes required');output=json.loads(raw);need(raw==canonical(output),'canonical output required');bound=bind(output['parent_record']);need(output['parent_envelope']==json.loads(bound.envelope_bytes)and output['input_exposure_sha256']==digest(bound.exposure),'output attribution mismatch');need(output['schema']=='ParentOutputAttributionPrototypeV2'and output['evaluator_authenticated']is False and output['actual_conditional_normalization']=='unverified','prototype cannot claim evaluator/normalization');return bound
