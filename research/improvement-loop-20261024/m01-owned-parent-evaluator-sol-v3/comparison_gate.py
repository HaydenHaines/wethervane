"""Trusted-local comparison boundary: rehash at use, execute verified source bytes."""
import hashlib,json,sys,types
from pathlib import Path
import common
from common import need,canonical,EXPECTED_CONTRACT,HERE

def execute(root,contract,review,review_sha,phase):
 need(phase in ['dynamic','fixed-B'],'unsupported comparison phase')
 # Includes common/evaluator/comparator/this gate and contract, using review-local paths.
 snapshot=common.reviewed_sources(review,review_sha,EXPECTED_CONTRACT)
 common.read_contract()
 need(Path(common.__file__).resolve()==(HERE/'common.py').resolve(),'unexpected cached common module')
 import numpy as np
 expected=contract['runtime_expected']['numpy'];need(np.__version__==expected['version']and Path(np.__file__).resolve()==Path(expected['path']).resolve(),'comparator NumPy substitution')
 previous=sys.modules.get('evaluator');ev=types.ModuleType('evaluator');ev.__file__=str(HERE/'evaluator.py');sys.modules['evaluator']=ev
 comp=types.ModuleType('_owned_verified_comparator');comp.__file__=str(HERE/'compare.py')
 try:
  # Neither comparator nor evaluator source is reopened by Python's ordinary import loader.
  exec(compile(snapshot['evaluator.py'],ev.__file__,'exec'),ev.__dict__)
  exec(compile(snapshot['compare.py'],comp.__file__,'exec'),comp.__dict__)
  value=comp.compare_dynamic(root,contract)if phase=='dynamic'else comp.compare(root,contract)
 finally:
  if previous is None:sys.modules.pop('evaluator',None)
  else:sys.modules['evaluator']=previous
 # Source drift during comparison is also a failure; it cannot create controller success.
 common.reviewed_sources(review,review_sha,EXPECTED_CONTRACT);common.read_contract()
 return value,{'schema':'VerifiedComparisonExecutionV1','phase':phase,'source_sha256':{name:hashlib.sha256(raw).hexdigest()for name,raw in snapshot.items()},'comparison_source_mode':'executed verified snapshot bytes; not a newly imported disk module','review_digest':review_sha,'contract_sha256':EXPECTED_CONTRACT,'scope':'trusted local code; no adversarial filesystem or power-loss guarantee'}
