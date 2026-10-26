"""Actual queued CLI processes with explicitly invented temporary native sources.

The temporary executable below patches the inspector solely for software tests,
then executes the unchanged real scoped_selection CLI. Production recipes expose
no inspector hook, and these synthetic results are not scientific evidence.
"""

import copy
import json
import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import execution
import experiment_queue as queue
import scoped_selection as selection
import selection_queue as recipe
import test_scoped_selection as fixtures


HERE = Path(__file__).resolve().parent
TEST_EXECUTABLE = '''#!/usr/bin/python3
import os, pathlib, runpy, sys
sys.path.insert(0, {here!r})
import execution, native_selection
from test_scoped_selection import fake_inspector
native_selection.inspect_count_candidate = fake_inspector
sys.argv = sys.argv[1:]
output = pathlib.Path(sys.argv[sys.argv.index('--output')+1])
if output.name == 'apply-fail':
    original = execution.publish
    def fail_after_commit(path, value):
        if pathlib.Path(path) == output/'receipt.json':
            os._exit(73)  # Only after actual registry COMMIT, before receipt publication.
        return original(path, value)
    execution.publish = fail_after_commit
runpy.run_path(sys.argv[0], run_name='__main__')
'''


class SelectionQueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SelectionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.root = self.base / "queue"
        self.interpreter = self.base / "INVENTED-synthetic-test-python"
        self.interpreter.write_text(TEST_EXECUTABLE.format(here=str(HERE)))
        self.interpreter.chmod(0o755)
        queue.initialize(self.root, 1)
        self.counter = 0

    def command(self, *argv):
        result = subprocess.run([sys.executable, "-B", *map(str, argv)], cwd=self.base,
                                env={"PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=25)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def request(self, spec, name, app=None):
        cycle=selection.document(app['decision']).get('journal',{}).get('cycle','INVENTED-count-cycle') if app else 'INVENTED-count-cycle'
        return self.fixture.write(dict(schema="ScopedSelectionQueueRecipeV1", phase="apply" if app else "prepare",
            queueId="queue-"+name, cycleId=cycle, jobKey="apply" if app else "prepare",
            attemptId=name, nativeSpec=spec, pythonExecutable=str(self.interpreter), outputDirectory=str(self.base/name),
            timeoutSeconds=15, application=app), "queue-request")

    def compile(self, request, name):
        pin = self.command(HERE/"selection_queue.py", "build", "--request", request['path'], "--sha256", request['sha256'],
                           "--output", self.base/("recipe-"+name))
        receipt = selection.document(pin)
        plan_pin = receipt['plan']
        plan = selection.document(plan_pin)
        # The temporary interpreter's fixture implementation and underlying
        # interpreter are explicit extra inputs; no production injection option.
        plan['jobs'][0]['execution_spec']['input_pins'].extend([
            execution.pin(HERE/"test_scoped_selection.py"), execution.pin(Path('/usr/bin/python3').resolve())])
        return self.fixture.write(plan,"admitted-plan"), plan

    def run_plan(self, plan_pin, plan, admitted=False):
        if not admitted:
            self.command(HERE/"experiment_queue.py", "--root", self.root, "enqueue", "--plan", plan_pin['path'], "--sha256", queue.digest(plan))
        self.counter += 1
        self.command(HERE/"experiment_queue.py", "--root", self.root, "work", "--output", self.base/("worker-"+str(self.counter)),
                     "--poll-seconds", ".02", "--max-runtime", "15")
        state=queue.status(self.root)
        self.assertTrue(state['all_jobs_recorded'],state)
        self.assertEqual(state['reserved'],0,state)
        job=plan['jobs'][0]
        result=queue.document(self.root/'queues'/plan['queue_id']/'jobs'/job['id']/'result.json')
        attempt=job['execution_spec']['attempt_id']
        status=execution.inspect(self.root/'executions',attempt) if (self.root/'executions'/'attempts'/attempt/'prepared.json').exists() else None
        if status:
            self.assertEqual(status['qualification'],'unassessed')
            self.assertEqual(status['attempt_terminal']['qualification'],'unassessed')
        return result,status

    def app(self, prep_pin, report, decision, command='INVENTED-stable-selection-command'):
        value=selection.document(decision)
        return dict(registryRoot=str(self.fixture.root),preparation=prep_pin,decision=decision,
                    expectedRevision=value['expectedRevision'],expectedPredecessor=value['expectedPredecessor'],commandId=command)

    def prepare_queued(self,name='prepare-ready',pending=False):
        spec=self.fixture.write(dict(fixtureLabel='INVENTED-A',pending=pending),'native-fixture')
        request=self.request(spec,name)
        plan_pin,plan=self.compile(request,name)
        result,status=self.run_plan(plan_pin,plan)
        self.assertEqual(result['outcome'],'accepted_execution')
        prep_pin=execution.pin(self.base/name/'receipt.json')
        return spec,prep_pin,selection.document(prep_pin)

    def event_count(self):
        db=sqlite3.connect(self.fixture.root/'selection.sqlite3')
        try:return db.execute('SELECT count(*) FROM events').fetchone()[0]
        finally:db.close()

    def test_actual_queued_pending_cli_records_state_without_establishing_selection(self):
        registry_before=execution.pin(self.fixture.root/'selection.sqlite3')
        spec,prep_pin,report=self.prepare_queued('prepare-pending',pending=True)
        self.assertEqual(report['status'],'blocked_requires_computation')
        self.assertIsNone(report['candidateRef'])
        self.assertFalse(report['selectionEstablished'])
        self.assertFalse(report['publicationAuthorized'])
        self.assertEqual(report['empiricalStatus'],'unassessed')
        self.assertEqual(registry_before,execution.pin(self.fixture.root/'selection.sqlite3'))
        self.assertFalse(self.fixture.journal.exists())
        self.assertEqual(self.event_count(),0)
        request=self.request(spec,'apply-pending',dict(registryRoot=str(self.base/'absent-registry'),preparation=prep_pin,
            decision=self.fixture.write({}),expectedRevision=-1,expectedPredecessor=None,commandId='pending-cannot-select'))
        with self.assertRaisesRegex(ValueError,'pending cannot establish selection'):
            recipe.compile_plan(request)
        self.assertFalse((self.base/'absent-registry').exists())

    def test_review_is_separate_and_actual_queued_apply_keeps_baseline_unassessed(self):
        spec,prep_pin,report=self.prepare_queued()
        self.assertEqual(report['status'],'review_required')
        self.assertEqual(self.event_count(),0)
        decision=self.fixture.authority(report)
        journal_before=execution.pin(self.fixture.journal/'ledger.sqlite3')
        request=self.request(spec,'apply-ready',self.app(prep_pin,report,decision))
        plan_pin,plan=self.compile(request,'apply-ready')
        sources={Path(item['path']).name for item in plan['jobs'][0]['execution_spec']['input_pins']}
        self.assertTrue(set(recipe.SOURCE_NAMES)<=sources)
        stale_request=selection.document(self.request(spec,'apply-stale',self.app(prep_pin,report,decision,command='INVENTED-stale-competitor')))
        stale_request['jobKey']='apply-stale-competitor'
        stale_pin,stale=self.compile(self.fixture.write(stale_request),'apply-stale')
        for admitted_pin,admitted in [(plan_pin,plan),(stale_pin,stale)]:
            self.command(HERE/'experiment_queue.py','--root',self.root,'enqueue','--plan',admitted_pin['path'],'--sha256',queue.digest(admitted))
        self.assertEqual(self.event_count(),0)  # Both competing CAS payloads were queued before selection.
        result,status=self.run_plan(plan_pin,plan,admitted=True)
        self.assertEqual(result['outcome'],'accepted_execution')
        applied=queue.document(self.base/'apply-ready'/'receipt.json')
        self.assertEqual(applied['status'],'selected')
        self.assertEqual(applied['domain'],'synthetic')
        self.assertFalse(applied['publicationAuthorized'])
        selected=selection.resolve_selected(self.fixture.root,report['taskScope'])
        self.assertEqual(selected['empiricalStatus'],'unassessed')
        self.assertFalse(selected['publicationAuthorized'])
        self.assertEqual(self.event_count(),1)
        self.assertEqual(journal_before,execution.pin(self.fixture.journal/'ledger.sqlite3'))
        stale_result=queue.document(self.root/'queues'/stale['queue_id']/'jobs'/stale['jobs'][0]['id']/'result.json')
        stale_status=execution.inspect(self.root/'executions','apply-stale')
        self.assertEqual(stale_result['outcome'],'failed_process')
        self.assertNotEqual(stale_status['attempt_terminal']['return_code'],0)
        self.assertIn('stale scoped selection predecessor/revision',
            (self.root/'executions'/'attempts'/'apply-stale'/'command.log').read_text())
        self.assertEqual(self.event_count(),1)
        self.assertEqual(journal_before,execution.pin(self.fixture.journal/'ledger.sqlite3'))

    def test_postcommit_failure_then_explicit_fresh_attempt_is_exactly_once(self):
        spec,prep_pin,report=self.prepare_queued()
        decision=self.fixture.authority(report)
        request=self.request(spec,'apply-fail',self.app(prep_pin,report,decision))
        original_pin,original=self.compile(request,'apply-fail')
        failed,status=self.run_plan(original_pin,original)
        self.assertEqual(failed['outcome'],'failed_process')
        self.assertEqual(status['attempt_terminal']['return_code'],73)
        self.assertTrue((self.base/'apply-fail').is_dir())
        self.assertEqual(list((self.base/'apply-fail').iterdir()),[])
        self.assertEqual(self.event_count(),1)
        before=execution.pin(self.fixture.root/'selection.sqlite3')
        terminal=status['terminal_receipt']
        retry_receipt=self.command(HERE/'selection_queue.py','retry','--previous-plan',original_pin['path'],
            '--previous-plan-sha256',original_pin['sha256'],'--terminal',terminal['path'],'--terminal-sha256',terminal['sha256'],
            '--queue-id','explicit-postcommit-retry','--attempt-id','apply-retry','--owner-output',self.base/'apply-retry',
            '--reason','INVENTED postcommit receipt failure; inspect stable command without duplicate event',
            '--output',self.base/'retry-recipe')
        retry_pin=selection.document(retry_receipt)['plan'];retry=selection.document(retry_pin)
        first=original['jobs'][0]['execution_spec'];second=retry['jobs'][0]['execution_spec']
        self.assertEqual(first['argv'][:-1],second['argv'][:-1])
        self.assertEqual(first['cycle_id'],second['cycle_id'])
        self.assertEqual(first['job_key'],second['job_key'])
        self.assertEqual(second['previous_attempt'],dict(attempt_id='apply-fail',receipt=terminal))
        self.assertTrue(all(pin in second['input_pins'] for pin in first['input_pins']))
        result,_=self.run_plan(retry_pin,retry)
        self.assertEqual(result['outcome'],'accepted_execution')
        applied=queue.document(self.base/'apply-retry'/'receipt.json')
        self.assertEqual(applied['status'],'already_applied')
        self.assertEqual(applied['commandEvent']['revision'],0)
        self.assertEqual(self.event_count(),1)
        self.assertEqual(before,execution.pin(self.fixture.root/'selection.sqlite3'))
        self.assertEqual(list((self.base/'apply-fail').iterdir()),[])
        for changed in ['command','decision']:
            bad=copy.deepcopy(original)
            if changed=='command':
                index=bad['jobs'][0]['execution_spec']['argv'].index('--command-id')+1
                bad['jobs'][0]['execution_spec']['argv'][index]='different-command'
            else:bad['jobs'][0]['execution_spec']['input_pins'].pop(0)
            with self.assertRaisesRegex(ValueError,'exact previous specification'):
                recipe.compile_retry(self.fixture.write(bad),terminal,'bad-retry-'+changed,'bad-attempt-'+changed,str(self.base/('bad-'+changed)),'INVENTED')

    def test_pinned_source_drift_defers_before_launch_without_selection(self):
        spec,prep_pin,report=self.prepare_queued()
        decision=self.fixture.authority(report)
        request=self.request(spec,'apply-drift',self.app(prep_pin,report,decision))
        plan_pin,plan=self.compile(request,'apply-drift')
        self.command(HERE/'experiment_queue.py','--root',self.root,'enqueue','--plan',plan_pin['path'],'--sha256',queue.digest(plan))
        Path(decision['path']).write_text('{"INVENTED":"changed after review"}\n')
        result,status=self.run_plan(plan_pin,plan,admitted=True)
        self.assertEqual(result['outcome'],'deferred_prelaunch_integrity')
        self.assertEqual(self.event_count(),0)
        self.assertFalse((self.root/'executions'/'attempts'/'apply-drift'/'launch-intent.json').exists())

    def test_prepared_then_drifted_unlaunched_attempt_resumes_existing_queue_contract(self):
        spec=self.fixture.write(dict(fixtureLabel='INVENTED-gap',pending=True),'native-fixture')
        request=self.request(spec,'prepare-gap')
        plan_pin,plan=self.compile(request,'prepare-gap')
        self.command(HERE/'experiment_queue.py','--root',self.root,'enqueue','--plan',plan_pin['path'],'--sha256',queue.digest(plan))
        source=Path(spec['path']);original=source.read_bytes();launch=execution.launch
        def drift_before_launch(root,attempt):
            source.write_bytes(original+b'\n')
            return launch(root,attempt)
        with patch.object(execution,'launch',side_effect=drift_before_launch):
            state=queue.tick(self.root)
        self.assertTrue(state['all_jobs_recorded'],state)
        self.assertEqual(state['reserved'],0)
        attempt=self.root/'executions'/'attempts'/'prepare-gap'
        self.assertTrue((attempt/'prepared.json').exists())
        self.assertFalse((attempt/'launch-intent.json').exists())
        self.assertFalse((self.base/'prepare-gap').exists())
        source.write_bytes(original)
        directory=self.root/'queues'/plan['queue_id']/'jobs'/'prepare'
        args=[]
        for key,path in [('previous-plan',Path(plan_pin['path'])),('binding',directory/'binding.json'),
                         ('result',directory/'result.json'),('release',directory/'release.json')]:
            pin=execution.pin(path);args.extend(['--'+key,pin['path'],'--'+key+'-sha256',pin['sha256']])
        receipt=self.command(HERE/'selection_queue.py','resume',*args,'--queue-id','explicit-gap-resume',
            '--reason','INVENTED exact source bytes restored before any native launch','--output',self.base/'resume-recipe')
        resumed_pin=selection.document(receipt)['plan'];resumed=selection.document(resumed_pin)
        self.assertEqual(resumed['jobs'][0]['execution_spec'],plan['jobs'][0]['execution_spec'])
        self.assertIsNotNone(resumed['jobs'][0]['resume_prelaunch'])
        result,status=self.run_plan(resumed_pin,resumed)
        self.assertEqual(result['outcome'],'accepted_execution')
        self.assertEqual(status['attempt_terminal']['return_code'],0)
        self.assertEqual(queue.document(self.base/'prepare-gap'/'receipt.json')['status'],'blocked_requires_computation')
        self.assertEqual(len(list((self.root/'executions'/'attempts').iterdir())),1)
        self.assertEqual(self.event_count(),0)
        self.assertFalse(self.fixture.journal.exists())
        # Once this resumed attempt has launched, an explicit retry must shed
        # the old prelaunch proof and use the ordinary terminal lineage.
        terminal=status['terminal_receipt']
        retry=recipe.compile_retry(resumed_pin,terminal,'explicit-after-resume','prepare-gap-retry',
            str(self.base/'prepare-gap-retry'),'INVENTED explicit pending recheck after verified terminal')
        self.assertIsNone(retry['jobs'][0]['resume_prelaunch'])
        self.assertEqual(retry['jobs'][0]['execution_spec']['previous_attempt'],
            dict(attempt_id='prepare-gap',receipt=terminal))
        retried,_=self.run_plan(self.fixture.write(retry),retry)
        self.assertEqual(retried['outcome'],'accepted_execution')
        self.assertEqual(self.event_count(),0)
        self.assertFalse(self.fixture.journal.exists())

    def test_recipe_rejects_cross_bound_authority_without_running_native_validation(self):
        spec,prep,report=self.fixture.prepared('INVENTED-bindings')
        decision=self.fixture.authority(report)
        request=selection.document(self.request(spec,'cross-bound',self.app(prep,report,decision)))
        for key,value in [('cycleId','wrong-journal-cycle'),('nativeSpec',self.fixture.write(dict(fixtureLabel='another')) )]:
            changed=copy.deepcopy(request);changed[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'binding mismatch|exact completed preparation'):
                recipe.compile_plan(self.fixture.write(changed))
        for key,value in [('expectedRevision',7),('expectedPredecessor',dict(eventSha256='0'*64,candidateRefSha256='1'*64))]:
            changed=copy.deepcopy(request);changed['application'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'binding mismatch'):
                recipe.compile_plan(self.fixture.write(changed))
        self.assertEqual(self.event_count(),0)


if __name__=='__main__':unittest.main()
