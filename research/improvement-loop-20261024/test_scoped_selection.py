"""Invented, temporary selection mechanics; no scientific qualification or real DB writes.

The fixed native inspector is mocked only inside these tests. Its fabricated
capabilities are NOT native validation evidence; native_selection tests cover that
separate boundary. Synthetic-domain stores never resolve as real research stores.
"""

import copy
import hashlib
import json
import multiprocessing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import execution
import native_selection
import scoped_selection as selection
from ledger import Ledger, canonical


def fake_inspector(spec):
    label = spec["fixtureLabel"]
    target = spec.get("target",selection.TARGET)
    scope = dict(targetId=target, outputContractId=selection.OUTPUT,
                 predictionEvents=[dict(id="TX-election", state="TX", cycle=spec.get("cycle", 2026), date="2026-11-03")],
                 predictionContests=[dict(id="TX-contest", eventId="TX-election", kind="regular", rule="plurality",
                                          candidateIds=["candidate-a", "candidate-b"], noncandidateCategories=[dict(id="blank", kind="blank")])])
    sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
    domain = spec.get("domain", "synthetic")
    result = dict(artifactRef=dict(kind="composition-application-result-v1", artifact_id=sha("artifact-" + label)),
                  record=dict(schema_version="composition-result-v1", configuration_sha256=sha("cfg-" + label), numerical_status="passed",
                              evidence_status="synthetic" if domain == "synthetic" else "unassessed"),
                  predictionRef=dict(kind="fixture-predictions", artifact_id=sha("draws-" + label)),
                  terminalReceipt=dict(path="INVENTED-terminal", sha256=sha("terminal-" + label)), commonDrawIdentity="INVENTED-shared-parent-chain-draw")
    native = dict(schema="VerifiedNativeCountCandidateV1", adapterSha256=execution.pin(Path(native_selection.__file__).resolve())["sha256"],
                  specSha256=selection.digest(spec), status="requires_computation" if spec.get("pending") else "ready",
                  request=dict(sha256=sha("request-" + label), targetId=target, outputContractId=selection.OUTPUT,
                               informationCutoff="2026-10-20T22:13:33Z", selections=[dict(role="measurement", provider="INVENTED-" + label)]),
                  configuration=dict(sha256=sha("cfg-" + label), supportId=sha("support-" + label), lawSha256=sha("law-" + label), graphSha256=sha("graph-" + label)),
                  scope=dict(kind="synthetic_lifecycle" if domain == "synthetic" else "actual_research", problemIdentity=sha("problem-" + label),
                             parentIds=["INVENTED-parent-a", "INVENTED-parent-b"], parentMasses=[.3, .7]),
                  taskScope=scope, taskScopeSha256=selection.digest(scope), observationSupport=dict(INVENTED=label),
                  observationSupportSha256=selection.digest(dict(INVENTED=label)),
                  result=None if spec.get("pending") else result)
    native['configuration']['outputs']=spec.get('outputs',[dict(contract_id=selection.OUTPUT,unit='integer_counts',
        denominator='declared_event_participants_and_contest_category_universes',
        estimand='joint candidate/category counts and shared event participants')])
    encoded = canonical(native).encode()
    return native_selection.VerifiedNativeCandidate(native["status"], native["request"]["sha256"], native["configuration"]["sha256"],
               None if native["result"] is None else result["artifactRef"]["artifact_id"], hashlib.sha256(encoded).hexdigest(), encoded, native_selection._TOKEN)


def race_apply(root, preparation, spec, decision, revision, predecessor, command, output, start, queue):
    with patch.object(native_selection, "inspect_count_candidate", side_effect=fake_inspector):
        start.wait()
        try:
            receipt = selection.apply(root, preparation, spec, decision, revision, predecessor, command, output)
            queue.put(("success", json.loads(Path(receipt["path"]).read_text())["status"]))
        except Exception as exc:
            queue.put(("rejected", str(exc)))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "registry"
        self.journal = self.base / "journal"
        self.patch = patch.object(native_selection, "inspect_count_candidate", side_effect=fake_inspector)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.counter = 0
        selection.initialize(self.root, "synthetic")

    def write(self, value, name="metadata"):
        self.counter += 1
        path = self.base / (str(self.counter) + "-" + name + ".json")
        execution.publish(path, value)
        return execution.pin(path)

    def prepared(self, label="A", **extra):
        spec = self.write(dict(fixtureLabel=label, **extra), "native-fixture")
        report = selection.preparation(spec)
        return spec, self.write(report, "prepared"), report

    def head(self, report):
        db = selection._connect(self.root, readonly=True)
        try:
            return selection._head(self.root, db, report["taskKey"])
        finally:
            db.close()

    def transfer(self, report, prior):
        def definition(native):
            return self.write(dict(schema="ScopedCountModelDefinitionV1", providerSelections=native["request"]["selections"],
                                   sourceEvidence=[self.write(dict(scope="INVENTED model definition"))]))
        candidate_definition = definition(report["nativeReceipt"])
        incumbent_definition = definition(prior["candidateRef"]["nativeReceipt"])
        rows = []
        for tier, year in (("A", 2018), ("A", 2022), ("B", 2020), ("B", 2024)):
            scope = dict(targetId=selection.TARGET, outputContractId=selection.OUTPUT, cycle=year, office="INVENTED Senate",
                         geography="INVENTED TX", raceIds=["INVENTED-" + str(year)], informationCutoff=str(year) + "-10-20T00:00:00Z",
                         horizonDays=14, informationBudget="INVENTED identical earlier information")
            row = dict(tier=tier, cycle=year, evaluationScope=scope)
            for side, model_definition in (("incumbent", incumbent_definition), ("candidate", candidate_definition)):
                # Historical data/configuration/result identities deliberately
                # differ from the target ref. Transfer is an explicit review.
                historical_definition = self.write(dict(schema="ScopedCountModelDefinitionV1", providerSelections=[dict(INVENTED=side + str(year))]))
                native_cfg = selection.digest([side, year])
                cfg = self.write(dict(schema="ScopedCountEvaluationConfigurationV1", scope=scope, modelDefinition=historical_definition, nativeConfigurationSha256=native_cfg))
                record = self.write(dict(schema_version="composition-result-v1", configuration_sha256=native_cfg, numerical_status="passed"))
                output = self.write(dict(schema="ScopedCountEvaluationOutputIdentityV1", scope=scope, nativeConfigurationSha256=native_cfg,
                    nativeRecord=record, predictionArtifact=dict(kind="INVENTED-draws", artifact_id=selection.digest([side,year,"draws"])),
                    parentIds=["INVENTED-parent"],commonDrawIdentity="INVENTED-shared-chain-draw"))
                terminal = self.write(dict(schema="ScopedCountEvaluationTerminalV1", scope=scope, nativeConfigurationSha256=native_cfg,
                    nativeRecord=record, outputIdentity=output, status="complete"))
                result = self.write(dict(schema="ScopedCountEvaluationResultV1", scope=scope, configuration=cfg,
                                         observationSupportSha256=selection.digest(['INVENTED shared data',year]),
                                         nativeRecord=record, terminalReceipt=terminal))
                row.update({side + "Definition":historical_definition, side + "Configuration":cfg, side + "Result":result})
            plan = self.write(dict(schema="ScopedCountMatchedEvaluationPlanV1", scope=scope, metricIds=["INVENTED-proper-score", "INVENTED-empty-subset"], primaryMetricId="INVENTED-proper-score",
                limitations=dict(benchmarkExposure="retrospective_development",asOfStatus="unassessed",reasons=["INVENTED numerical checks do not establish predictive skill"]),
                **{side + key:row[side + key] for side in ("incumbent", "candidate") for key in ("Configuration", "Definition")}))
            common = dict(scope=scope, incumbentResult=row["incumbentResult"], candidateResult=row["candidateResult"])
            row["comparisonPlan"] = plan
            cells={"INVENTED-proper-score":dict(status="available",supportCount=1,supportUnit="contest",incumbent=1,candidate=.9,pairedDifference=-.1,
                    uncertainty=dict(status="estimated",standardError=.02,reason="INVENTED conditional numerical uncertainty"),reason="INVENTED score"),
                   "INVENTED-empty-subset":dict(status="unavailable",supportCount=0,supportUnit="contest",incumbent=None,candidate=None,pairedDifference=None,
                    uncertainty=dict(status="unresolved",standardError=None,reason="Empty fixture subset"),reason="Empty fixture subset")}
            row["scores"] = self.write(dict(schema="ScopedCountMatchedEvaluationScoresV1", plan=plan, metricScores=cells, **common))
            row["numericalReview"] = self.write(dict(schema="ScopedCountNumericalReviewV1", disposition="accepted_declared_gates", **common))
            row["inputBudgetReview"] = self.write(dict(schema="ScopedCountInformationMatchReviewV1", plan=plan, disposition="matched_inputs_or_explicit_data_arm",
                inputRelationship="matched",reason="INVENTED matched historical information",candidateObservationSupportSha256=selection.digest(['INVENTED shared data',year]),
                incumbentObservationSupportSha256=selection.digest(['INVENTED shared data',year]),**common))
            rows.append(row)
        transfer = self.write(dict(schema="ScopedCountTargetTransferReviewV1", taskKey=report["taskKey"], candidateRefSha256=report["candidateRefSha256"],
            incumbentRefSha256=prior["candidateRefSha256"], candidateDefinition=candidate_definition, incumbentDefinition=incumbent_definition,
            targetInputRelationship=dict(candidateObservationSupportSha256=report['nativeReceipt']['observationSupportSha256'],
                incumbentObservationSupportSha256=prior['candidateRef']['nativeReceipt']['observationSupportSha256'],inputRelationship='declared_data_arm',reason='INVENTED explicit measurement-data difference'),
            disposition="supports_scoped_target_change", evaluationMappings=[{key:row[key] for key in ("tier","cycle","candidateDefinition","incumbentDefinition","scores","inputBudgetReview")} for row in rows]))
        return dict(targetCandidateRefSha256=report["candidateRefSha256"], targetIncumbentRefSha256=prior["candidateRefSha256"],
                    candidateDefinition=candidate_definition, incumbentDefinition=incumbent_definition, evaluations=rows, transferReview=transfer)

    def authority(self, report, operation="bootstrap", prior=None, restore=None, transfer_mutator=None, empirical_override=None):
        domain = "synthetic"
        revision = -1 if prior is None else prior["revision"]
        native = report["nativeReceipt"]
        primary = "complete_baseline" if operation == "bootstrap" else "measurement"
        changed = ["measurement"] if operation == "promote" else []
        comparison = self.transfer(report, prior) if operation == "promote" else None
        if transfer_mutator:
            transfer_mutator(comparison)
        review = self.write(dict(schema="ScopedCountSelectionReviewV1", domain=domain, operation=operation, taskKey=report["taskKey"],
            candidateRefSha256=report["candidateRefSha256"], predecessor=selection.predecessor(prior), disposition=selection.OPS[operation][1],
            nativeCompatibility=dict(candidateProblemIdentity=native["scope"]["problemIdentity"], candidateSupportId=native["configuration"]["supportId"],
                incumbentProblemIdentity=prior["candidateRef"]["nativeReceipt"]["scope"]["problemIdentity"] if prior else None, taskScopeSha256=report["taskKey"]),
            evidence=[self.write(dict(INVENTED="Independent software-policy fixture, not scientific approval"))], comparisonTransfer=comparison,
            primaryLayer=primary, changedRoles=changed, interactionScope=None))
        binding = dict(incumbent_config=prior["candidateRef"] if prior else report["candidateRef"], scope=dict(taskScope=report["taskScope"],domain=domain),
                       target_layer=primary, hypothesis="INVENTED temporary selection policy fixture")
        cycle = "fixture-" + str(self.counter)
        empirical = "unassessed" if operation == "bootstrap" or prior is None else "reviewed_comparative_support" if operation == "promote" else prior["empiricalStatus"]
        if operation=='rollback' and restore:
            connection=selection._connect(self.root,readonly=True)
            empirical=json.loads(connection.execute('SELECT body FROM events WHERE hash=?',(restore,)).fetchone()[0])['empiricalStatus']
            connection.close()
        if empirical_override is not None:
            empirical=empirical_override
        decision = self.write(dict(schema="ScopedCountSelectionDecisionV1", domain=domain, operation=operation, taskKey=report["taskKey"],
            candidateRefSha256=report["candidateRefSha256"], expectedRevision=revision, expectedPredecessor=selection.predecessor(prior),
            journal=dict(root=str(self.journal),cycle=cycle,revision=1,binding=binding), review=review,
            empiricalStatus=empirical, reason="INVENTED explicit reviewed policy fixture", restoreEventSha256=restore,
            primaryLayer=primary, changedRoles=changed, interactionScope=None))
        ledger = Ledger(self.journal)
        ledger.create_cycle(cycle + "create", cycle, **binding)
        refs = {"evaluation":ledger.put_artifact(cycle+"decision",decision["path"])["sha256"], "review":ledger.put_artifact(cycle+"review",review["path"])["sha256"]}
        ledger.checkpoint(cycle+"decide",cycle,0,"decided","INVENTED reviewed fixture",refs,recommendation=selection.OPS[operation][0])
        ledger.close()
        return decision

    def apply(self, spec, prep, report, decision, prior=None, command=None, output=None):
        self.counter += 1
        return selection.apply(self.root, prep, spec, decision, -1 if prior is None else prior["revision"], selection.predecessor(prior),
                               command or "fixture-command-"+str(self.counter), output or self.base/('output-'+str(self.counter)))

    def bootstrap(self):
        spec, prep, report = self.prepared()
        decision = self.authority(report)
        self.apply(spec,prep,report,decision,command="bootstrap")
        return spec,prep,report,decision,self.head(report)

    def test_bootstrap_promote_retain_reverse_and_resolve_complete_reference(self):
        a_spec,a_prep,a,a_decision,first = self.bootstrap()
        self.assertEqual(first["empiricalStatus"],"unassessed")
        b_spec,b_prep,b = self.prepared("B")
        self.assertEqual(a["taskKey"],b["taskKey"])
        self.assertNotEqual(a["candidateRefSha256"],b["candidateRefSha256"])
        decision = self.authority(b,"promote",first)
        self.apply(b_spec,b_prep,b,decision,first)
        promoted = self.head(b)
        self.assertEqual(promoted["primaryLayer"],"measurement")
        self.assertEqual(promoted['evaluationBasis']['evaluations'][0]['metricCells']['INVENTED-empty-subset']['status'],'unavailable')
        retained = self.authority(b,"retain",promoted)
        self.apply(b_spec,b_prep,b,retained,promoted)
        hold = self.head(b)
        self.assertEqual(hold["candidateRef"],promoted["candidateRef"])
        reversal = self.authority(a,"rollback",hold,restore=first["hash"])
        self.apply(a_spec,a_prep,a,reversal,hold)
        current = selection.resolve_selected(self.root,a["taskScope"])
        self.assertEqual(current["revision"],3)
        self.assertEqual(current["candidateRef"],first["candidateRef"])
        self.assertEqual(current["domain"],"synthetic")
        self.assertFalse(current["publicationAuthorized"])
        history=selection.history(self.root,a["taskScope"])
        self.assertEqual([event['revision'] for event in history['events']],[0,1,2,3])

    def test_pending_and_serialized_receipt_cannot_initialize_or_select(self):
        spec,prep,report = self.prepared("pending",pending=True)
        absent = self.base/'no-registry'
        with self.assertRaisesRegex(ValueError,"selection blocked"):
            selection.apply(absent,prep,spec,self.write({}),-1,None,"bad",self.base/'pending-output')
        self.assertFalse(absent.exists())
        forged = copy.deepcopy(report);forged['status']='review_required';forged['candidateRef']={'qualified':True}
        with self.assertRaisesRegex(ValueError,"fresh trusted native"):
            selection.apply(self.root,self.write(forged),spec,self.write({}),-1,None,"forged",self.base/'forged-output')

    def test_no_m02_or_wrong_election_scope_substitution_and_no_first_retain(self):
        spec,prep,report = self.prepared()
        decision = self.authority(report,"retain",prior=None)
        with self.assertRaises((ValueError,TypeError)):
            self.apply(spec,prep,report,decision)
        m02=self.write(dict(fixtureLabel='M02',target='senate_aggregate_logit_empirical_mixture_v1'))
        with self.assertRaisesRegex(ValueError,'physical prediction task mismatch'):
            selection.preparation(m02)
        _,_,a,_,first = self.bootstrap()
        b_spec,b_prep,b = self.prepared("historical",cycle=2018)
        self.assertNotEqual(a['taskKey'],b['taskKey'])
        wrong = self.authority(b,"promote",first)
        with self.assertRaisesRegex(ValueError,"stale scoped selection"):
            self.apply(b_spec,b_prep,b,wrong,first)

    def test_postcommit_output_failure_retry_survives_new_journal_and_registry_head(self):
        spec,prep,report = self.prepared()
        decision=self.authority(report)
        output=self.base/'failed-output';original=execution.publish
        def fail(path,value):
            if Path(path)==output/'receipt.json':raise OSError('postcommit fixture failure')
            return original(path,value)
        with patch.object(execution,'publish',side_effect=fail):
            with self.assertRaisesRegex(OSError,'postcommit'):
                self.apply(spec,prep,report,decision,command='stable',output=output)
        first=self.head(report)
        authority=selection.document(decision);cycle=authority['journal']['cycle']
        ledger=Ledger(self.journal)
        ledger.checkpoint('later-journal',cycle,1,'decided','Later independent journal recommendation',recommendation='defer')
        ledger.close()
        b_spec,b_prep,b=self.prepared('B');promote=self.authority(b,'promote',first)
        self.apply(b_spec,b_prep,b,promote,first)
        before=execution.pin(self.root/'selection.sqlite3')
        result=self.apply(spec,prep,report,decision,command='stable',output=self.base/'retry-output')
        value=selection.document(result)
        self.assertEqual(value['status'],'already_applied')
        self.assertEqual(value['commandEvent']['revision'],0)
        self.assertEqual(value['currentHead']['revision'],1)
        self.assertEqual(before,execution.pin(self.root/'selection.sqlite3'))

    def test_stale_concurrent_candidates_accept_one_and_preserve_loser(self):
        _,_,a,_,first=self.bootstrap()
        candidates=[]
        for label in ['B','C']:
            spec,prep,report=self.prepared(label);decision=self.authority(report,'promote',first)
            candidates.append((spec,prep,report,decision))
        context=multiprocessing.get_context('fork');start=context.Event();messages=context.Queue()
        processes=[context.Process(target=race_apply,args=(str(self.root),prep,spec,decision,first['revision'],selection.predecessor(first),
                   'race-'+str(i),str(self.base/('race-out-'+str(i))),start,messages)) for i,(spec,prep,report,decision) in enumerate(candidates)]
        for process in processes:process.start()
        start.set()
        results=[messages.get(timeout=20) for _ in processes]
        for process in processes:process.join(20);self.assertEqual(process.exitcode,0)
        self.assertEqual(sum(kind=='success' for kind,_ in results),1,results)
        self.assertEqual(self.head(a)['revision'],1)
        for spec,prep,_,_ in candidates:self.assertTrue(Path(spec['path']).exists());self.assertTrue(Path(prep['path']).exists())

    def test_journal_writer_cannot_commit_while_selection_cas_holds_snapshot(self):
        spec,prep,report=self.prepared();decision=self.authority(report)
        original=selection._journal_check;probes=[]
        def checked(db,journal,decision_value,pin,replay=False):
            original(db,journal,decision_value,pin,replay)
            other=sqlite3.connect(self.journal/'ledger.sqlite3',timeout=0,isolation_level=None)
            try:
                other.execute('BEGIN IMMEDIATE')
                other.execute('INSERT INTO commands VALUES(?,?,?)',('competing-journal','{}','{}'))
                with self.assertRaisesRegex(sqlite3.OperationalError,'locked'):other.execute('COMMIT')
                probes.append(True)
            finally:other.rollback();other.close()
        with patch.object(selection,'_journal_check',side_effect=checked):self.apply(spec,prep,report,decision)
        self.assertEqual(probes,[True])

    def test_transfer_wrong_scope_result_and_missing_cycle_reject(self):
        _,_,a,_,first=self.bootstrap()
        for label,mutate in [('missing',lambda m:m['evaluations'].pop()),('wrong-result',lambda m:m['evaluations'][0].update(candidateResult=m['evaluations'][1]['candidateResult']))]:
            spec,prep,report=self.prepared(label);decision=self.authority(report,'promote',first,transfer_mutator=mutate)
            with self.assertRaises(ValueError):self.apply(spec,prep,report,decision,first)
        self.assertEqual(self.head(a)['revision'],0)

    def test_wal_journal_and_synthetic_to_research_rejected(self):
        spec,prep,report=self.prepared();decision=self.authority(report)
        connection=sqlite3.connect(self.journal/'ledger.sqlite3');connection.execute('pragma journal_mode=wal');connection.close()
        with self.assertRaisesRegex(ValueError,'WAL'):self.apply(spec,prep,report,decision)
        real=self.base/'research-registry';selection.initialize(real,'research')
        with self.assertRaisesRegex(ValueError,'domains must remain separate'):
            selection.apply(real,prep,spec,decision,-1,None,'no-synthetic',self.base/'real-out')

    def test_corrupt_native_bytes_and_immutable_registry_history_fail_closed(self):
        spec,prep,report,decision,first=self.bootstrap()
        db=selection._connect(self.root)
        for table in ['configuration','objects','events','commands']:
            with self.assertRaisesRegex(sqlite3.IntegrityError,'append-only'):db.execute('DELETE FROM '+table)
            row=db.execute('SELECT * FROM '+table+' LIMIT 1').fetchone()
            if row:
                with self.assertRaisesRegex(sqlite3.IntegrityError,'append-only'):
                    db.execute('INSERT OR REPLACE INTO '+table+' VALUES('+','.join('?' for _ in row)+')',tuple(row))
        db.close()
        sha=first['retainedEvidence'][0]['sha256'];path=self.root/'objects'/sha;path.chmod(0o644);path.write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'fingerprint'):selection.resolve_selected(self.root,report['taskScope'])

    def test_resolve_reopens_native_contract_instead_of_returning_cached_ready(self):
        _,_,report,_,_=self.bootstrap()
        with patch.object(native_selection,'inspect_count_candidate',side_effect=ValueError('native artifact drift')):
            with self.assertRaisesRegex(ValueError,'native artifact drift'):selection.resolve_selected(self.root,report['taskScope'])

    def test_missing_nonnumeric_primary_score_and_wrong_historical_target_are_rejected(self):
        _,_,a,_,first=self.bootstrap()
        for label,change in [('null',lambda d:d['metricScores'].update({'INVENTED-proper-score':None})),
                             ('nonnumeric',lambda d:d['metricScores']['INVENTED-proper-score'].update(candidate='not a number')),
                             ('unavailable',lambda d:d['metricScores'].update({'INVENTED-proper-score':copy.deepcopy(d['metricScores']['INVENTED-empty-subset'])}))]:
            def mutation(mapping):
                row=mapping['evaluations'][0];score=selection.document(row['scores']);change(score)
                row['scores']=self.write(score)
                transfer=selection.document(mapping['transferReview'])
                transfer['evaluationMappings'][0]['scores']=row['scores']
                mapping['transferReview']=self.write(transfer)
            spec,prep,report=self.prepared(label);decision=self.authority(report,'promote',first,transfer_mutator=mutation)
            with self.assertRaises(ValueError):self.apply(spec,prep,report,decision,first)
        self.assertEqual(self.head(a)['revision'],0)

    def test_wrong_score_plan_and_result_link_cannot_pass_pinned_file_check(self):
        _,_,a,_,first=self.bootstrap()
        def mutation(mapping):
            row=mapping['evaluations'][0];score=selection.document(row['scores'])
            score['plan']=mapping['evaluations'][1]['comparisonPlan'];row['scores']=self.write(score)
            transfer=selection.document(mapping['transferReview']);transfer['evaluationMappings'][0]['scores']=row['scores']
            mapping['transferReview']=self.write(transfer)
        spec,prep,report=self.prepared('wrong-plan');decision=self.authority(report,'promote',first,transfer_mutator=mutation)
        with self.assertRaisesRegex(ValueError,'scope mismatch'):self.apply(spec,prep,report,decision,first)
        self.assertEqual(self.head(a)['revision'],0)

    def test_real_policy_authority_fixture_is_explicit_not_an_actual_native_result(self):
        # Pure admission predicate only: trusted inspector double fabricates an
        # actual-shaped fixture. It cannot be supplied to the production CLI.
        spec,prep,report=self.prepared('PRODUCTION-POLICY-FIXTURE',domain='research')
        decision=self.authority(report)
        d=selection.document(decision);r=selection.document(d['review']);r['domain']='research';d['domain']='research'
        d['journal']['binding']['scope']['domain']='research';d['review']=self.write(r)
        parsed,evidence,basis=selection._authorize(self.write(d),report,None,'research')
        self.assertEqual(parsed['operation'],'bootstrap');self.assertIsNone(basis)
        self.assertEqual(parsed['empiricalStatus'],'unassessed')
        self.assertTrue(evidence)

    def test_rollback_cannot_upgrade_historical_baseline_evidence(self):
        spec,prep,a,_,baseline=self.bootstrap()
        b_spec,b_prep,b=self.prepared('B')
        self.apply(b_spec,b_prep,b,self.authority(b,'promote',baseline),baseline)
        promoted=self.head(b)
        decision=self.authority(a,'rollback',promoted,restore=baseline['hash'],empirical_override='reviewed_comparative_support')
        with self.assertRaisesRegex(ValueError,'cannot relabel the historical evidence'):
            self.apply(spec,prep,a,decision,promoted)
        self.assertEqual(self.head(a)['hash'],promoted['hash'])

    def test_historical_terminal_must_bind_exact_record_configuration_and_output(self):
        _,_,a,_,baseline=self.bootstrap()
        for label,key in [('wrong-record','nativeRecord'),('wrong-config','nativeConfigurationSha256'),('wrong-output','outputIdentity')]:
            def mutation(mapping):
                row=mapping['evaluations'][0]
                result=selection.document(row['candidateResult'])
                terminal=selection.document(result['terminalReceipt'])
                other=selection.document(mapping['evaluations'][1]['candidateResult'])
                other_terminal=selection.document(other['terminalReceipt'])
                terminal[key]=other_terminal[key]
                result['terminalReceipt']=self.write(terminal)
                row['candidateResult']=self.write(result)
            spec,prep,report=self.prepared(label)
            decision=self.authority(report,'promote',baseline,transfer_mutator=mutation)
            with self.assertRaisesRegex(ValueError,'historical (terminal|output identity)'):
                self.apply(spec,prep,report,decision,baseline)
        self.assertEqual(self.head(a)['revision'],0)

    def test_replay_response_must_name_exact_committed_event(self):
        spec,prep,report,decision,_=self.bootstrap()
        db=selection._connect(self.root)
        recorded=db.execute("SELECT payload,response FROM commands WHERE id='bootstrap'").fetchone()
        response=json.loads(recorded['response']);response['eventSha256']='0'*64
        db.execute('INSERT INTO commands VALUES(?,?,?)',('corrupt-response',recorded['payload'],canonical(response)))
        db.close()
        with self.assertRaisesRegex(ValueError,'response/event mismatch'):
            self.apply(spec,prep,report,decision,command='corrupt-response')

    def test_extra_input_relationship_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'unsupported input relationship fields'):
            selection._information_relationship(dict(candidateObservationSupportSha256='0'*64,
                incumbentObservationSupportSha256='0'*64,inputRelationship='matched',reason='INVENTED',silentlyQualified=True))

    def test_task_scope_derives_exact_native_count_output_semantics(self):
        _,_,report=self.prepared('native-output-semantics')
        output=report['nativeReceipt']['configuration']['outputs'][0]
        self.assertEqual(report['taskScope']['outputUnits'],output['unit'])
        self.assertEqual(report['taskScope']['denominator'],output['denominator'])
        self.assertEqual(report['taskScope']['outputEstimand'],output['estimand'])
        unrelated=dict(contract_id='INVENTED-other-output',unit='other',denominator='other',estimand='other')
        _,_,with_other=self.prepared('native-output-semantics',outputs=[unrelated,output])
        self.assertEqual(report['taskKey'],with_other['taskKey'])
        old_scope=copy.deepcopy(report['taskScope']);old_scope['denominator']='not_applicable_to_count_outputs';old_scope.pop('outputEstimand')
        self.assertNotEqual(report['taskKey'],selection.digest(old_scope))

    def test_missing_ambiguous_and_unsupported_native_output_semantics_reject(self):
        _,_,report=self.prepared('source-output')
        output=report['nativeReceipt']['configuration']['outputs'][0]
        for outputs in [None,[],[dict(output,contract_id='INVENTED-other-output')],[output,output]]:
            with self.subTest(outputs=outputs), self.assertRaisesRegex(ValueError,'native output contract'):
                self.prepared('bad-output',pending=True,outputs=outputs)
        for key in ['unit','denominator','estimand']:
            for value in [None,'INVENTED-unsupported']:
                bad=copy.deepcopy(output)
                if value is None:bad.pop(key)
                else:bad[key]=value
                with self.subTest(key=key,value=value), self.assertRaisesRegex(ValueError,'unsupported native count output semantics'):
                    self.prepared('bad-semantics',pending=True,outputs=[bad])


if __name__=='__main__':unittest.main()
