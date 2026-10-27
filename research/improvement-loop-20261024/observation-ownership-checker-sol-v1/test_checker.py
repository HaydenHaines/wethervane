import copy,json,tempfile,unittest
from pathlib import Path
from checker import audit,audit_a2018_aggregate,digest,Invalid
H='a'*64;J='b'*64;T='2020-01-01T00:00:00Z'
def source(id='s',sha=H,origin='release',role='observed'):
 return dict(id=id,sha256=sha,origin_namespace=origin,provenance_sha256=J,available_at=T,role=role)
def atom(token='local-secret-record',field='question1',measurement='turnout-self-report',source_id='s',target='event:participation'):
 return dict(source_id=source_id,record_token=token,field_id=field,measurement_id=measurement,target=target,raw_unit='survey_answer',model_unit='survey_answer',event_id='event',period_start=T,period_end=T,transformation_sha256=None)
def factor(id='f',components=None,kind='likelihood',sampling=None):
 components=components if components is not None else[dict(id='c',observations=[atom()])]
 f=dict(id=id,kind=kind,structure='single',implementation_sha256=H,joint_model_sha256=None,inputs=[],components=components,sampling=sampling)
 seal(f);return f
def seal(f):f['vector_binding_sha256']=digest({'components':f['components'],'implementation_sha256':f['implementation_sha256']})
def manifest(factors=None,sources=None):return dict(schema='ObservationOwnershipManifestV1',composition_id='invented',cutoff='2020-02-01T00:00:00Z',model_binding=dict(problem_sha256=H,composition_sha256=H,target_roster_sha256=H),sources=sources if sources is not None else[source()],factors=factors if factors is not None else[factor()])
def run(m):return audit(m,digest(m))
def partition(labels):return dict(schema='DeclaredSamplingPartitionV1',namespace='samples',unit='sample',labels=labels,provenance_sha256=J)
class CheckerTests(unittest.TestCase):
 def test_exact_atom_duplicate_independent_factors(self):
  r=run(manifest([factor('one'),factor('two')]));self.assertEqual(r['status'],'duplicate_declared_observed_atoms');self.assertEqual(r['cross_factor_duplicate_atom_pairs'],1)
 def test_two_fields_same_record_are_distinct_unknown_dependent(self):
  f=factor('two',[dict(id='other',observations=[atom(field='question2',measurement='candidate-answer',target='event:candidate')])]);r=run(manifest([factor(),f]));self.assertEqual(r['cross_factor_duplicate_atom_pairs'],0);self.assertEqual(r['unknown_sampling_overlap_pairs'],1);self.assertEqual(r['status'],'overlap_review_required')
 def test_joint_factor_reuses_atom_but_needs_joint_model(self):
  f=factor(components=[dict(id='a',observations=[atom()]),dict(id='b',observations=[atom(target='event:derived-logit')])]);f['structure']='joint';f['joint_model_sha256']=J;seal(f)
  r=run(manifest([f]));self.assertEqual(r['joint_internal_atom_reuses'],1);self.assertFalse(r['joint_law_authenticated'])
  f['joint_model_sha256']=None
  with self.assertRaises(Invalid):run(manifest([f]))
 def test_duplicate_same_component_rejects(self):
  f=factor(components=[dict(id='a',observations=[atom(),atom()])])
  with self.assertRaises(Invalid):run(manifest([f]))
 def test_target_relabel_transform_and_source_alias_cannot_hide_atom(self):
  a=atom(source_id='alias',target='different-target');a['model_unit']='logit';a['transformation_sha256']=J
  f=factor('two',[dict(id='a',observations=[a])]);r=run(manifest([factor(),f],[source(),source('alias')]))
  self.assertEqual(r['cross_factor_duplicate_atom_pairs'],1)
 def test_derived_source_lineage_same_atom(self):
  a=atom(source_id='derived');f=factor('two',[dict(id='a',observations=[a])]);r=run(manifest([factor(),f],[source(),source('derived',sha=J,role='derived_observed')]))
  self.assertEqual(r['cross_factor_duplicate_atom_pairs'],1)
 def test_conflicting_measurement_unit_or_time_rejects(self):
  for key,value in [('measurement_id','renamed'),('raw_unit','persons'),('event_id','different'),('period_end','2020-01-02T00:00:00Z')]:
   a=atom();a[key]=value
   if key=='raw_unit':a['transformation_sha256']=J
   with self.assertRaises(Invalid):run(manifest([factor(),factor('two',[dict(id='a',observations=[a])])]))
 def test_same_bytes_conflicting_lineage_rejects(self):
  with self.assertRaises(Invalid):run(manifest(sources=[source(),source('alias',origin='invented-new-origin')]))
 def test_disjoint_atom_sets_not_respondent_independence(self):
  f=factor('two',[dict(id='a',observations=[atom(token='other-record')])]);r=run(manifest([factor(),f]));self.assertEqual(r['distinct_observation_factor_pairs'],1);self.assertEqual(r['unknown_sampling_overlap_pairs'],1)
 def test_sampling_attestation_is_declarative_source_error_unknown(self):
  f=factor('two',[dict(id='a',observations=[atom(token='other-record')])],sampling=partition(['sample-b']));r=run(manifest([factor(sampling=partition(['sample-a'])),f]));self.assertEqual(r['declared_disjoint_sampling_pairs'],1);self.assertFalse(r['sampling_attestations_authenticated']);self.assertEqual(r['source_error_dependence_unknown_pairs'],1);self.assertEqual(r['status'],'overlap_review_required')
  f['sampling']=partition(['sample-a']);r=run(manifest([factor(sampling=partition(['sample-a'])),f]));self.assertEqual(r['known_sampling_overlap_pairs'],1)
 def test_shared_covariate_and_uncertain_input_not_observed_atoms(self):
  a=factor('prior',[],kind='prior');b=factor('input',[],kind='shared_input');a['inputs']=b['inputs']=['cov'];r=run(manifest([factor(),a,b],[source(),source('cov',sha=J,origin='covariates',role='deterministic_input')]));self.assertEqual(r['cross_factor_duplicate_atom_pairs'],0);self.assertEqual(r['shared_input_references'],2)
  r=run(manifest([factor(),a,b],[source(),source('cov',sha=J,origin='population',role='uncertain_input')]));self.assertEqual(r['reused_uncertain_input_sources'],1)
 def test_observed_input_in_prior_needs_review_not_false_clean(self):
  f=factor('coordinate',[],kind='coordinate_prior');f['inputs']=['s'];r=run(manifest([factor(),f]));self.assertEqual(r['status'],'evidence_input_review_required')
 def test_vector_reorder_and_external_manifest_tamper(self):
  m=manifest();pin=digest(m);m['factors'][0]['components'][0]['observations'][0]['target']='new'
  with self.assertRaisesRegex(Invalid,'caller_manifest_pin'):audit(m,pin)
  with self.assertRaisesRegex(Invalid,'vector_binding'):run(m)
 def test_temporal_scope_and_unknown_fields_fail(self):
  for change in [lambda m:m['sources'][0].__setitem__('available_at','2021-01-01T00:00:00Z'),lambda m:m['factors'][0]['components'][0]['observations'][0].__setitem__('period_end','2021-01-01T00:00:00Z'),lambda m:m.__setitem__('undeclared',True)]:
   m=manifest();change(m);seal(m['factors'][0])
   with self.assertRaises(Invalid):run(m)
 def test_empty_never_scientific_or_evidence_completed(self):
  r=run(manifest([]));self.assertEqual(r['status'],'no_observed_likelihoods');self.assertFalse(r['scientifically_qualified'])
 def test_output_never_contains_source_tokens(self):
  m=manifest();raw=json.dumps(run(m));self.assertNotIn('local-secret-record',raw);self.assertNotIn('question1',raw);self.assertNotIn('turnout-self-report',raw)
 def test_coherent_lineage_renaming_is_not_authenticated(self):
  # Boundary fixture: caller can author a coherent different-source claim.
  s=source('other',sha=J,origin='caller-invented-new-origin')
  f=factor('two',[dict(id='a',observations=[atom(source_id='other')])])
  r=run(manifest([factor(),f],[source(),s]));self.assertEqual(r['cross_factor_duplicate_atom_pairs'],0);self.assertFalse(r['source_admission_authenticated']);self.assertFalse(r['verified_independence']);self.assertEqual(r['status'],'overlap_review_required')
 def test_frozen_a2018_aggregate_consistency_only(self):
  path=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/m01-a2018-observation-ownership-luna-v2/aggregate-ledger.json');import hashlib
  r=audit_a2018_aggregate(path,'077e620e255fdb12474497f040afa061fcebb17c506b173692ced2acb438e423');self.assertEqual(r['memberships'],8242);self.assertFalse(r['record_ownership_audited']);self.assertEqual(r['sampling_overlap'],'unknown')
  with self.assertRaises(Invalid):audit_a2018_aggregate(path,'0'*64)
if __name__=='__main__':unittest.main(verbosity=2)
