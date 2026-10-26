"""Read-only receipt audit and bounded partial-file hashing. NEVER unpickle."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'m01-parent-gradient-after-replacement-plan-sol-v1';ATTEMPT=OLD/'attempt-root-v1';REC=ATTEMPT/'receipts'
def pin(p):
 before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=after.st_size,allocated_bytes=getattr(after,'st_blocks',0)*512)
def read(p):return json.loads(p.read_text())
launch=read(ATTEMPT/'launch.json');terminal=read(ATTEMPT/'terminal.json');stages=read(REC/'stages.json');plan=read(OLD/'plan.json');auth=read(OLD/'execution-receipt-root-v1.json')
assert pin(OLD/'plan.json')['sha256']==launch['plan_sha256']==auth['plan_sha256']=='1b0873938589b94ffd9b3a038077dcc6a41157ac01a77615e5f6245518a3dab8'
assert pin(OLD/'execution-receipt-root-v1.json')['sha256']=='14e19d156ef7476553782c23177e0f36dd85295ed6ed651c521085767226b212'
for item in plan['source_pins']:assert pin(Path(item['path']))['sha256']==item['sha256']
for item in [auth['independent_failure_review'],auth['resource_recheck']]:assert pin(Path(item['path']))['sha256']==item['sha256']
order=read(REC/'coordinate-order.json');assert len(order)==64 and order[0]['start']==0 and order[-1]['stop']==662 and all(a['stop']==b['start']for a,b in zip(order,order[1:]))
records=stages['records'];names=[r['stage']for r in records];assert names==['execution_gate_passed','original_graph_and_first_parent_authenticated','exact_native_continuous_value_order_saved','strict_logp_replacement_passed','native_pregradient_rewrite_checked','gradient_after_replacement_checked','failed_closed']
checks=[records[i]['detail']for i in [3,4,5]];assert all(c['explicit_E_input']and c['old_port_absent']and c['stale_whole_parent_arrays']==0 for c in checks)
assert checks[0]['value_input_names']==checks[1]['value_input_names']==checks[2]['value_input_names'] and len(checks[0]['value_input_names'])==64
inventory=read(REC/'factor-inventory.json');assert inventory==read(HERE.parent/'m01-native-successor-provider-sol-v1/compiler-inventory.json')
canonical=json.dumps(inventory,sort_keys=True,separators=(',',':')).encode();assert hashlib.sha256(canonical).hexdigest()==records[1]['detail']['inventory_sha256']
contract=read(REC/'first-parent-contract.json');assert contract==records[1]['detail']['contract']and contract['parent_mass']==.125 and contract['parent_mass_used_in_logp']is False
partial=pin(REC/'symbolic-graphs.pkl');assert partial['bytes']==terminal['limits']['individual_file_bytes']==64*1024**2
assert terminal['returncode']==1 and terminal['termination_reason']is None and'OSError: [Errno 27] File too large'in records[-1]['detail']['error']
files=[p for p in ATTEMPT.rglob('*')if p.is_file()];accounting=[pin(p)if p.name!='symbolic-graphs.pkl'else partial for p in files];total=sum(p['bytes']for p in accounting)
result=dict(schema='GradientPersistenceAttemptReadOnlyAuditV1',status='symbolic_structure_passed_serialization_file_cap_failed',source_gate_pins_verified=True,stage_order=names,coordinate_variables=64,coordinate_dimensions=662,symbolic_checks=checks,contract=contract,partial_pickle=partial,partial_handling='stat plus one bounded1MiB-stream SHA256 only; never pickle.load, parse, resume, modify or delete',terminal=terminal,final_file_accounting=accounting,final_logical_output_bytes=total,final_nonpickle_bytes=total-partial['bytes'],monitor_peak_output_bytes=terminal['peak_output_bytes'],monitor_vs_final_note='Final stat sum includes post-monitor terminal/failure/log updates; it need not equal sampled live peak.',serialization_total_required='unknown; only failed64MiB lower-bound attempt established',numerical_parity=False,complete_restorable_graph_persisted=False,new_actual_graphs_or_compiles=0,pins={name:pin(OLD/name)for name in ['plan.json','experiment.py','launch_once.py','execution-receipt-root-v1.json','independent-review-luna-v1.json']})
(HERE/'audit-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],partial_bytes=partial['bytes'],final_output_bytes=total,final_nonpickle_bytes=total-partial['bytes'],partial_sha256=partial['sha256'])))
