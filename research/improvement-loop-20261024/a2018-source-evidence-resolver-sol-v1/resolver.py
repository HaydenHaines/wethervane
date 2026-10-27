"""Frozen A2018 prepared-source resolver. No individual identifiers may leave result."""
import hashlib,importlib.util,json,math
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
BINDINGS_SHA256='a8bb344fc7aa65969501635ee4075de653622273b92edc5acd53fda6eab4f838'
class Rejected(ValueError):pass
def need(x,code):
 if not x:raise Rejected(code)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read_pinned(item,reader=lambda p:Path(p).read_bytes()):
 raw=reader(item['path']);need(len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'],'pinned_source_bytes_differ');return raw
def load_bindings():
 raw=(HERE/'bindings.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==BINDINGS_SHA256,'resolver_binding_manifest_drift');return json.loads(raw)
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def inspect_prepared(problem):
 """Derive prepared-record memberships only; raw measurement lineage remains unknown."""
 sources={s['id']:s for s in problem['sources']};need(len(sources)==len(problem['sources']),'duplicate_source_identity')
 events={e['id']for e in problem['events']};contests={c['id']:c for c in problem['contests']}
 need(len(events)==len(problem['events'])and len(contests)==len(problem['contests']),'duplicate_event_or_contest')
 seen=set();survey_block_ids=set();survey_vectors=[];survey_rows=0;source_uses=Counter()
 for block in problem['surveys']:
  need(block['id']not in survey_block_ids,'duplicate_survey_block');survey_block_ids.add(block['id'])
  need(block['source_id']in sources and block['event_id']in events,'survey_source_event_missing')
  need(block['contest_id']is None or block['contest_id']in contests,'survey_contest_missing')
  vector=[]
  for row in block['rows']:
   token=row.get('record_id');need(isinstance(token,str)and bool(token),'survey_record_key_missing')
   key=(block['source_id'],token);need(key not in seen,'duplicate_source_scoped_survey_membership');seen.add(key)
   required={'record_id','cell_id','community_mask','turnout_mask','answer_mask'};need(set(row)==required,'survey_row_schema_changed')
   # Hash the full prepared representation in memory; never return tokens/masks.
   vector.append(row);survey_rows+=1
  survey_vectors.append((block['id'],block['source_id'],block['event_id'],block['contest_id'],block['groups'],vector));source_uses[block['source_id']]+=1
 poll_ids=set();poll_blocks=set();poll_vectors=[];poll_rows=0;target_rows=0
 for block in problem['polls']:
  need(block['id']not in poll_blocks,'duplicate_poll_block');poll_blocks.add(block['id'])
  need(block['source_id']in sources,'poll_source_missing');cov=block['covariance'];n=len(block['rows'])
  need(len(cov)==n and all(isinstance(row,list)and len(row)==n for row in cov),'poll_covariance_dimensions')
  need(all(type(x)in(int,float)and math.isfinite(x)for row in cov for x in row),'nonfinite_poll_covariance')
  need(all(cov[i][j]==cov[j][i]for i in range(n)for j in range(i)),'asymmetric_poll_covariance')
  need(all(cov[i][i]>0 for i in range(n)),'nonpositive_poll_covariance_diagonal')
  for row in block['rows']:
   need(row['observation_id']not in poll_ids,'duplicate_poll_measurement_key');poll_ids.add(row['observation_id'])
   need(row['contest_id']in contests,'poll_contest_missing');candidates={c['id']for c in contests[row['contest_id']]['candidates']}
   need(row['first_candidate_id']in candidates and row['second_candidate_id']in candidates and row['first_candidate_id']!=row['second_candidate_id'],'poll_candidate_target_mismatch')
   need(type(row['logit_value'])in(int,float)and math.isfinite(row['logit_value']),'nonfinite_poll_measurement')
   poll_rows+=1;target_rows+=int(row['contest_id']=='contest_az2018_regular')
  poll_vectors.append(block);source_uses[block['source_id']]+=1
 used_hashes=Counter(sources[sid]['sha256']for sid in source_uses)
 alias_groups=sum(n>1 for n in used_hashes.values())
 return {'events':len(events),'contests':len(contests),'survey_blocks':len(survey_block_ids),'survey_memberships':survey_rows,'survey_contest_joint_answer_blocks':sum(x['contest_id']is not None for x in problem['surveys']),'survey_no_contest_community_turnout_blocks':sum(x['contest_id']is None for x in problem['surveys']),'poll_blocks':len(poll_blocks),'poll_rows':poll_rows,'target_poll_rows':target_rows,'source_scoped_survey_keys_unique':True,'poll_keys_unique':True,'used_source_alias_hash_groups':alias_groups,'alias_lineage_status':'unresolved_alias_groups'if alias_groups else'no_equal_hash_alias_in_used_registry_not_proof_of_disjointness','survey_prepared_vector_sha256':digest(survey_vectors),'poll_ordered_vector_covariance_sha256':digest(poll_vectors),'covariance_checks':['dimensions','finite','exact_symmetry','positive_diagonal'],'covariance_positive_definite_verified':False}

def resolve(reader=lambda p:Path(p).read_bytes()):
 bindings=load_bindings();raw={name:read_pinned(pin,reader)for name,pin in bindings.items()}
 context=json.loads(raw['context']);plan=json.loads(raw['plan']);problem=json.loads(raw['problem']);aggregate=json.loads(raw['aggregate'])
 need(context['problem']==problem,'context_problem_mismatch')
 need(context['identity']==plan['context_identity']and context['descriptor']['problem_identity']==plan['problem_identity'],'plan_context_identity_mismatch')
 # Source-specific existing closure adapter is itself pinned; no model imports.
 closure=module('a2018_frozen_closure',bindings['closure_builder']['path'])
 c,p,d,hashes=closure.load_actual();need(c==context and p==plan and d==problem,'closure_source_bundle_mismatch')
 checks=closure.validate_closure(context,plan,problem,raw['components'].decode(),raw['rank'].decode(),raw['open'].decode())
 summary=inspect_prepared(problem);expected=aggregate['counts']
 need(summary['survey_blocks']==expected['survey_blocks']and summary['survey_memberships']==expected['survey_record_rows']and summary['poll_blocks']==expected['poll_blocks']and summary['poll_rows']==expected['poll_rows'],'ledger_aggregate_mismatch')
 need(summary['target_poll_rows']==aggregate['target']['poll_row_count'],'ledger_target_mismatch')
 checker=module('frozen_claim_checker_v2',bindings['claim_checker']['path'])
 aggregate_check=checker.audit_a2018_aggregate(bindings['aggregate']['path'],bindings['aggregate']['sha256'])
 return {'schema':'FrozenA2018SourceResolverV1','status':'prepared_mapping_audited_full_atom_admission_blocked','source_bytes_pinned_and_checked':True,'prepared_context_plan_and_closure_binding_checked':True,'actual_compiled_factor_graph_authenticated':False,'original_source_observation_atoms_authenticated':False,'exact_source_version_publication_times_verified':False,'respondent_or_sample_independence_verified':False,'scientifically_qualified':False,'summary':summary,'factor_blocks':summary['survey_blocks']+summary['poll_blocks'],'prepared_source_factor_contract':'source-declared survey Potential and poll MvNormal mappings; no compiled graph execution','prepared_measurement_types':{'survey_contest_row':'joint masked community/turnout/answer contribution; original source question identity unknown','survey_no_contest_row':'joint masked community/turnout contribution; answer mask not consumed in this branch','poll_row':'paired-candidate logit statistic in one ordered covariance block; not a person/sample identifier'},'source_registry_availability_status':'available_utc supplied and byte-pinned; exact original public version availability unverified','claim_checker':{'aggregate_consistency':'passed','full_row_audit':'not_invoked_missing_source_measurement_and_chronology','reason':'prepared survey masks do not establish original question field IDs/collection period; poll rows lack sample/publication metadata; available_utc is a supplied registry claim, not verified exact-version release'},'aggregate_check':aggregate_check,'closure_status':'source_algebra_original_eventG_conditional_on_shared_kappa_not_posterior_or_transformed_density','closure_condition_counts':{k:v for k,v in checks.items()if type(v)in(int,bool)},'unknown_or_unsupported':['original survey raw question/response lineage','respondent uniqueness and cross-source sample overlap','poll sample identity/design/error overlap','exact byte-version public availability versus registry available_utc','per-row measurement collection intervals','complete raw-source target/field semantic admission','compiled likelihood/vector/parent execution binding','covariance positive-definite numerical check'],'pins_sha256':{k:v['sha256']for k,v in bindings.items()}}

if __name__=='__main__':
 print(json.dumps(resolve(),indent=2,sort_keys=True))
