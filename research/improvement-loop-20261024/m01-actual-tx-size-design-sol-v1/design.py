"""Standard-library source/AST inventory only; never imports the actual TX runner."""
import ast
import hashlib
import json
from email.parser import Parser
from pathlib import Path

HERE=Path(__file__).resolve().parent;LAB=HERE.parent
ORIGINAL=LAB/'m01-parent-gradient-after-replacement-plan-sol-v1'
BUNDLE_KEYS=['original_cost','replaced_cost','rewritten_cost','gradient','ordered_value_vars','E','segments']

def pin(path):
 raw=Path(path).read_bytes();return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

def describe():
 plan=json.loads((ORIGINAL/'plan.json').read_text())
 for item in plan['source_pins']:
  current=pin(item['path']);assert current['sha256']==item['sha256'] and current['bytes']==item['bytes']
 tree=ast.parse((ORIGINAL/'experiment.py').read_text())
 symbolic=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='symbolic')
 calls=list(n for n in ast.walk(symbolic) if isinstance(n,ast.Call))
 one=lambda predicate:next(n for n in calls if predicate(n))
 cost=one(lambda n: isinstance(n.func,ast.Attribute) and n.func.attr=='logp')
 assert any(k.arg=='jacobian' and isinstance(k.value,ast.Constant) and k.value.value is True for k in cost.keywords)
 replaced=one(lambda n:isinstance(n.func,ast.Name) and n.func.id=='graph_replace')
 rewrite=one(lambda n:isinstance(n.func,ast.Name) and n.func.id=='rewrite_pregrad')
 gradient=one(lambda n:isinstance(n.func,ast.Name) and n.func.id=='gradient')
 dumps=[n for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='pickle' and n.func.attr=='dump']
 assert len(dumps)==1;dump=dumps[0]
 assert isinstance(dump.args[0],ast.Call) and isinstance(dump.args[0].func,ast.Name) and dump.args[0].func.id=='dict'
 keys=[k.arg for k in dump.args[0].keywords];assert keys==BUNDLE_KEYS
 assert [(k.arg,ast.literal_eval(k.value)) for k in dump.keywords]==[('protocol',5)]
 assert cost.lineno<replaced.lineno<rewrite.lineno<gradient.lineno<dump.lineno
 assert ast.unparse(gradient)=='gradient(rewritten, ordered)'
 assert ast.unparse(replaced).endswith('strict=True)')
 runtime=LAB.parent.parent/'.improvement-ledger/runtime/calibration-v1-env'
 site=runtime/'lib/python3.12/site-packages';versions={}
 for name in ('pymc','pytensor','numpy','scipy','arviz','xarray','pandas'):
  candidates=[f for f in site.glob('*.dist-info/METADATA') if f.parent.name.lower().startswith(name+'-')]
  assert len(candidates)==1,(name,candidates)
  metadata=Parser().parsestr(candidates[0].read_text());versions[name]=dict(version=metadata['Version'],metadata_pin=pin(candidates[0]))
 return dict(schema='ActualTXSizeMeasurementDesignV1',status='unexecuted_source_bound_design_only',allow_actual_tx_execution=False,
  old_partial_pickle_access='prohibited; no read/hash/load/retry by this packet',
  source_experiment=pin(ORIGINAL/'experiment.py'),source_plan=pin(ORIGINAL/'plan.json'),
  verified_existing_source_pins=plan['source_pins'],source_analysis=dict(kind='AST_structure_only_not_scientific_validation',
   pipeline_lines=dict(complete_logp=cost.lineno,strict_identity_replace=replaced.lineno,rewrite_pregrad=rewrite.lineno,native_order_gradient=gradient.lineno,old_pickle_boundary=dump.lineno),
   exact_bundle_keys=keys,one_logical_stream=True,protocol=5,buffer_callback=None,independent_roots_forbidden=True),
  gradient_contract=plan['gradient_contract'],frozen_factor_inventory=plan['frozen_factor_inventory'],frozen_shape_witness=plan['frozen_shape_witness'],
  target_runtime_metadata_only=dict(environment=str(runtime),package_versions=versions,scientific_modules_imported=False),
  future_required_inputs=dict(source_bound_technical_execution_record=None,fresh_resource_observation=None,selected_controller_source=None,independent_design_review=None),
  proposed_caps=dict(wall_seconds_including_build_and_io=120,cpu_soft_seconds=90,cpu_hard_seconds=95,rss_watchdog_bytes=2*1024**3,address_space_bytes=6*1024**3,
   logical_stream_budget_bytes=512*1024**2,individual_metadata_file_bytes=1024**2,aggregate_reports_plus_fresh_cache_bytes=16*1024**2,
   nice=15,threads=1,watchdog_poll_seconds=.25),
  measurement_success='All original symbolic/parent/factor guards pass, one Pickler returns normally, exact counted bytes/digest and durable progress validated after observed worker/controller exit under cooperating exclusive-writer protocol; no payload retained.',
  independent_scientific_limits=['No numerical logp/gradient parity','No all-parent normalization or reuse qualification','No pickle restoration proof','No compile/sampler/forecast readiness','No source admission or predictive improvement'])

if __name__=='__main__':print(json.dumps(describe(),indent=2))
