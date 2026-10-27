"""Read/hash/AST and arithmetic only; imports no queue/executor."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'queue-finite-owned-sol-v1'
def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def functions(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
def calls(n):return [ast.unparse(x.func)for x in ast.walk(n)if isinstance(x,ast.Call)]
def idle_model(n,executable_bytes=0):
 if type(n)is not int or n<0:raise ValueError('nonnegative integer successful job count required')
 if type(executable_bytes)is not int or executable_bytes<0:raise ValueError('nonnegative integer byte size required')
 return dict(successful_jobs=n,reservations=n,executor_intents=n,ownership_audits_per_idle_tick=2*n,containment_comparisons_per_idle_tick=n*n*(n-1),executable_rehashes_per_idle_tick=2*n*n,logical_executable_bytes_per_idle_tick=2*n*n*executable_bytes)
def main():
 q,e,f=[functions(SOURCE/name)for name in ['experiment_queue.py','execution.py','finite_intake.py']]
 checks={
  'tick_calls_released_result_and_status':all(x in calls(q['_tick_locked'])for x in ['verified_result','_status_snapshot']),
  'status_calls_verified_result':'verified_result'in calls(q['_status_snapshot']),
  'result_calls_ownership_for_job':'ownership_for_job'in calls(q['verified_result']),
  'ownership_for_job_rebuilds_expected_and_audit':all(x in calls(q['ownership_for_job'])for x in ['ownership_expected','executor.ownership_audit']),
  'audit_calls_registry_and_live_validate':all(x in calls(e['ownership_audit'])for x in ['ownership_intents','validate']),
  'registry_calls_overlap':'overlap'in calls(e['ownership_intents']),
  'live_validation_calls_executable_pin_check':'verify_executable'in calls(e['_validate']),
  'manager_gate_calls_full_admission_scan':'managed_admissions'in calls(e['assert_manager']),
  'test_runner_relative_cleanup_output':"Path('cleanup-verification.json').write_text"in(SOURCE/'run_tests.py').read_text(),
 }
 assert all(checks.values()),checks
 frozen=json.loads((SOURCE/'evidence.json').read_text());checked=[]
 for item in frozen['local_pins']:
  actual=pin(Path(item['path']));assert actual['sha256']==item['sha256']and actual['bytes']==item['bytes'],item['path'];checked.append(actual)
 executable=Path(sys.executable).resolve();size=executable.stat().st_size
 result=dict(schema='FiniteOwnedScalingSourceAnalysisV1',scope='source-only; no implementation import, execution, workload or timing benchmark',source_pins=[pin(SOURCE/name)for name in ['experiment_queue.py','execution.py','finite_intake.py','receipt_projection.py','run_tests.py','run_tests.sh','evidence.json','README.md']],frozen_local_pin_checks=len(checked),static_assertions=checks,illustrative_local_interpreter_metadata=dict(path=str(executable),bytes=size,method='stat only; not hashed or run as fixture'),healthy_released_case_assumptions=['N successful jobs, N reservations and N distinct intents in one root','valid released receipts, no corruption/contention/partial or active jobs, no early return','plain idle tick, not timing/model-cycle evidence'],arithmetic_projection=[idle_model(n,size)for n in [1,2,4,8,16,32,64,128,256,1000]],function_lines={label:{name:node.lineno for name,node in values.items()}for label,values in [('queue',q),('executor',e),('intake',f)]})
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
