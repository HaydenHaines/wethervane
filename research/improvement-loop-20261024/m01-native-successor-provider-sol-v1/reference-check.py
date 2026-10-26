"""Clean-process independent frozen-helper reference. No native package import."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
nodes=ast.parse((HERE/'provider.py').read_text()).body
expected=next(n.value.value for n in nodes if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='EXPECTED_SELECTION_SHA256'for t in n.targets))
raw=(HERE/'selection.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==expected;s=json.loads(raw)
for name in ['m01-count-measurement-prototype-sol-v1/prototype.py','calibration-consumer-v1/consumer.py']:
 p=LAB/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==s['pins'][str(p)]
sys.path.insert(0,str(LAB/'m01-count-measurement-prototype-sol-v1'));import prototype
case=prototype.load_frozen_current_case(LAB/'calibration-consumer-v1',target_cycle=2026,cutoff='2026-10-20T22:13:33Z',target=prototype.TARGET)
with np.load(HERE/'reference-mixture.npz',allow_pickle=False)as reference:
 assert np.array_equal(reference['covariances'],case['mixture'].covariances)
 assert np.array_equal(reference['weights'],case['mixture'].weights)
print(json.dumps(dict(status='all4000_frozen_helper_arrays_bitwise_equal',reference_sha256=hashlib.sha256((HERE/'reference-mixture.npz').read_bytes()).hexdigest(),covariance_shape=list(case['mixture'].covariances.shape))))
