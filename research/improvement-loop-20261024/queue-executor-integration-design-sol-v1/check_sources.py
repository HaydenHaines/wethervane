"""Read/parse pinned source only. Does not import either queue or executor."""
import ast,hashlib,json
from pathlib import Path
LAB=Path(__file__).resolve().parent.parent
FILES={
 'queue':LAB/'queue-finite-run-sol-v2/experiment_queue.py',
 'intake':LAB/'queue-finite-run-sol-v2/finite_intake.py',
 'queue_tests':LAB/'queue-finite-run-sol-v2/test_experiment_queue.py',
 'finite_tests':LAB/'queue-finite-run-sol-v2/test_finite.py',
 'queue_evidence':LAB/'queue-finite-run-sol-v2/evidence.json',
 'executor':LAB/'executor-output-ownership-sol-v1/execution.py',
 'executor_tests':LAB/'executor-output-ownership-sol-v1/test_ownership.py',
 'executor_evidence':LAB/'executor-output-ownership-sol-v1/evidence.json',
 'executor_review':LAB/'executor-output-ownership-review-luna-v1/review.json',
}
def functions(path):
 return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
def calls(node):
 return [ast.unparse(n.func)for n in ast.walk(node)if isinstance(n,ast.Call)]
def pin(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
q,e=functions(FILES['queue']),functions(FILES['executor'])
checks={
 'released_result_does_not_call_executor_load_or_registered':not any(x in calls(q['verified_result'])for x in ['executor.load','executor.registered','executor.ownership_binding','executor.ownership_intents']),
 'historical_registry_calls_full_validate':'validate'in calls(e['ownership_intents']),
 'full_validate_calls_verify_pin':'verify_pin'in calls(e['validate']),
 'dispatcher_has_flock':'fcntl.flock'in calls(q['dispatcher']),
 'prepare_has_global_flock_and_job_lock':all(x in calls(e['prepare'])for x in ['fcntl.flock','lock']),
 'launch_has_no_global_ownership_lock_call':'ownership_intents'not in calls(e['launch']),
 'supervisor_does_not_call_prepare':'prepare'not in calls(e['supervise']),
 'queue_tick_calls_prepare_then_launch':all(x in calls(q['_tick_locked'])for x in ['executor.prepare','executor.launch']),
 'inherited_unrelated_drift_test_present':'def test_prepared_input_drift_defers_without_terminal_and_unrelated_job_runs' in FILES['queue_tests'].read_text(),
}
assert all(checks.values()),checks
result={'schema':'QueueExecutorIntegrationStaticEvidenceV1','method':'read/hash/AST only; no imports, fixtures, queue execution or model jobs','pins':{k:pin(p)for k,p in FILES.items()},'static_checks':checks,'definitions':{'queue':{n:q[n].lineno for n in ['current_sources','configuration','dispatcher','verified_result','validate_resume','_tick_locked','finalize']},'executor':{n:e[n].lineno for n in ['load','registered','ownership_binding','ownership_intents','prepare','launch','recover']}},'conclusion':'integration requires reviewed changes; not a drop-in replacement; static checks are limited source facts, not runtime/deadlock proof'}
print(json.dumps(result,indent=2))
