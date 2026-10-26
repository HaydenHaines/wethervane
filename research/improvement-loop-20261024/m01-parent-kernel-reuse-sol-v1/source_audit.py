"""Read-only parent-constant/source audit; no native module or full graph imports."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
OLD=LAB/'m01-native-successor-provider-sol-v1';ROOT=Path('/home/hayden/projects/wethervane-redesign-20261006/integration');BASE=ROOT/'research/senate-stack-integration-20261020/modular-count-graph-astra-v1'
selection_bytes=(OLD/'selection.json').read_bytes();assert hashlib.sha256(selection_bytes).hexdigest()=='03edb47049384b662cdd645053fc6083bde8fb41d972b1a07da8b71ecf6283b5';s=json.loads(selection_bytes)
paths=[BASE/'components.py',BASE/'graph.py',ROOT/'src/senate_count_stack_v1/inputs.py',OLD/'provider.py',OLD/'selection.json',LAB/'checkpoints/continuation-v54.json']
pins={}
for p in paths:
 b=p.read_bytes();digest=hashlib.sha256(b).hexdigest()
 if str(p)in s['pins']:assert digest==s['pins'][str(p)]
 pins[str(p)]=dict(sha256=digest,bytes=len(b))
parents=s['parents'];keys=list(parents[0]);fields={k:dict(equal_across_all=all(p[k]==parents[0][k]for p in parents))for k in keys}
assert fields['exposure']['equal_across_all']is False and [p['mass']for p in parents]==[.125]*8
# Name direct read sites in selected numerical component source.
tree=ast.parse((BASE/'components.py').read_text());reads=[]
for n in ast.walk(tree):
 if isinstance(n,ast.Attribute)and isinstance(n.value,ast.Name)and n.value.id=='c'and n.attr in ['parent','protected']:reads.append(dict(attribute=n.attr,line=n.lineno))
assert reads==[dict(attribute='protected',line=56)]
result=dict(schema='FrozenTXParentKernelSourceAuditV1',pins=pins,parent_fields=fields,parent_count=8,original_masses=[p['mass']for p in parents],exposure_shape=[len(parents[0]['exposure']),len(parents[0]['exposure'][0])],direct_selected_component_parent_read_sites=reads,numerical_dependency='Only whole exposure3x24 enters selected provider numerical graph via ProtectedPopulation.exposure. IDs/mass/lineage also enter compiler contract/target metadata, not logp.',current_api_symbolic_parent_supported=False,compiler_block='compile_model requires exposure output np.ndarray exactly array_equal to ctx.protected exposure before converting to TensorConstant/SpecifyShape.',possible_route='Experimental identity-based substitution of model.typed_ports exposure in unoptimized logp/dlogp, adding explicit symbolic conditioning input and separately authenticating full selected parent envelope. Not tested on actual graph here.',scope='Source and serialized actual parent fields only; no actual graph import/build/compile, source edits or scientific qualification.')
(HERE/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(parent_count=8,exposure_shape=result['exposure_shape'],symbolic_api_supported=False,direct_parent_reads=reads)))
