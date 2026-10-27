"""Trusted-manifest structural ownership; never serialize observed atom identifiers."""
import hashlib,json,math,re
from datetime import datetime,timezone
from itertools import combinations

class Invalid(ValueError):pass

def need(value,code):
 if not value:raise Invalid(code)
def exact(v,keys,code):need(isinstance(v,dict)and set(v)==set(keys),code)
def text(v):return isinstance(v,str)and bool(v.strip())and v==v.strip()and not any(ord(c)<32 for c in v)
def sha(v):return isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)is not None
def stamp(v):
 need(isinstance(v,str),'invalid_time')
 try:r=datetime.fromisoformat(v.replace('Z','+00:00'))
 except (ValueError,OverflowError):raise Invalid('invalid_time')from None
 need(r.tzinfo is not None,'timezone_required');return r.astimezone(timezone.utc)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def audit(manifest,expected_manifest_sha256):
 need(sha(expected_manifest_sha256)and digest(manifest)==expected_manifest_sha256,'caller_manifest_pin_mismatch')
 exact(manifest,['schema','composition_id','cutoff','model_binding','sources','factors'],'manifest_fields')
 need(manifest['schema']=='ObservationOwnershipManifestV2'and text(manifest['composition_id']),'manifest_identity')
 exact(manifest['model_binding'],['problem_sha256','composition_sha256','target_roster_sha256'],'model_binding_fields')
 need(all(sha(x)for x in manifest['model_binding'].values()),'model_binding_pins')
 cutoff=stamp(manifest['cutoff']);sources={};origin_by_bytes={}
 need(isinstance(manifest['sources'],list),'sources_required')
 for s in manifest['sources']:
  exact(s,['id','sha256','origin_namespace','provenance_sha256','available_at','first_release_at','role'],'source_fields')
  need(all(text(s[k])for k in ['id','origin_namespace','role'])and sha(s['sha256'])and sha(s['provenance_sha256']),'source_identity')
  need(s['role']in ['observed','derived_observed','deterministic_input','uncertain_input'],'typed_source_role')
  if s['sha256']in origin_by_bytes:need(origin_by_bytes[s['sha256']]==s['origin_namespace'],'source_lineage_alias_conflict')
  origin_by_bytes[s['sha256']]=s['origin_namespace']
  need(s['id']not in sources,'duplicate_source_id');need(s['available_at']is not None,'source_version_availability_unknown');available=stamp(s['available_at']);need(available<=cutoff,'source_after_cutoff')
  if s['first_release_at']is not None:need(stamp(s['first_release_at'])<=available,'first_release_after_version_availability')
  sources[s['id']]=s
 need(isinstance(manifest['factors'],list),'factors_required')
 ids=set();all_atoms={};observed=[];inputs=0;joint_reuse=0;atom_claims=0;evidence_inputs=0;uncertain_refs={}
 for index,f in enumerate(manifest['factors']):
  exact(f,['id','kind','structure','implementation_sha256','joint_model_sha256','vector_binding_sha256','inputs','components','sampling'],'factor_fields')
  need(text(f['id'])and f['id']not in ids and sha(f['implementation_sha256']),'factor_identity');ids.add(f['id'])
  need(f['kind']in ['likelihood','prior','coordinate_prior','shared_input'],'factor_kind')
  need(f['structure']in ['single','joint'],'factor_structure')
  need((sha(f['joint_model_sha256'])if f['structure']=='joint'else f['joint_model_sha256']is None),'joint_model_pin_required')
  need(isinstance(f['inputs'],list)and len(set(f['inputs']))==len(f['inputs'])and all(x in sources for x in f['inputs']),'input_source_binding');inputs+=len(f['inputs'])
  for sid in f['inputs']:
   if sources[sid]['role']in ['observed','derived_observed']:evidence_inputs+=1
   if sources[sid]['role']=='uncertain_input':uncertain_refs[sid]=uncertain_refs.get(sid,0)+1
  need(isinstance(f['components'],list),'components_required')
  need(f['vector_binding_sha256']==digest({'components':f['components'],'implementation_sha256':f['implementation_sha256']}),'vector_binding_mismatch')
  if f['kind']!='likelihood':need(not f['components']and f['sampling']is None,'nonlikelihood_cannot_claim_observations');continue
  need(bool(f['components']),'likelihood_requires_observations')
  need(f['structure']=='joint'or len(f['components'])==1,'multiple_components_need_joint_model')
  component_ids=set();factor_atoms=set()
  for comp in f['components']:
   exact(comp,['id','observations'],'component_fields');need(text(comp['id'])and comp['id']not in component_ids,'component_identity');component_ids.add(comp['id'])
   need(isinstance(comp['observations'],list)and bool(comp['observations']),'observations_required');within=set()
   for o in comp['observations']:
    exact(o,['source_id','record_token','field_id','measurement_id','target','raw_unit','model_unit','event_id','period_start','period_end','transformation_sha256'],'observation_fields')
    need(o['source_id']in sources and all(text(o[k])for k in ['record_token','field_id','measurement_id','target','raw_unit','model_unit','event_id']),'observation_identity')
    need(o['period_start']is not None and o['period_end']is not None,'observed_collection_period_unknown')
    start,end=stamp(o['period_start']),stamp(o['period_end']);need(start<=end<=cutoff,'observation_temporal_scope')
    need(end<=stamp(sources[o['source_id']]['available_at']),'observation_after_source_version_available')
    need(o['transformation_sha256']is None or sha(o['transformation_sha256']),'transformation_pin')
    need(o['model_unit']==o['raw_unit']or sha(o['transformation_sha256']),'unit_transform_required')
    source=sources[o['source_id']];need(source['role']in ['observed','derived_observed'],'observation_source_role');key=(source['origin_namespace'],o['record_token'],o['field_id'])
    # Canonical original measurement identity cannot be changed to evade reuse.
    identity=(o['measurement_id'],o['raw_unit'],o['event_id'],start,end)
    if key in all_atoms:need(all_atoms[key]==identity,'canonical_measurement_conflict')
    else:all_atoms[key]=identity
    need(key not in within,'duplicate_atom_in_component');within.add(key);atom_claims+=1
    if key in factor_atoms:joint_reuse+=1
    factor_atoms.add(key)
  sampling=f['sampling']
  if sampling is not None:
   exact(sampling,['schema','namespace','unit','labels','provenance_sha256'],'sampling_fields')
   need(sampling['schema']=='DeclaredSamplingPartitionV1'and text(sampling['namespace'])and sampling['unit']in ['respondent','sample']and sha(sampling['provenance_sha256']),'sampling_identity')
   need(isinstance(sampling['labels'],list)and sampling['labels']and all(text(x)for x in sampling['labels'])and len(set(sampling['labels']))==len(sampling['labels']),'sampling_partition_labels')
  observed.append((index,factor_atoms,sampling))
 duplicates=0;unknown=0;attested_disjoint=0;known_overlap=0;distinct_pairs=0
 for (_,a,sa),(_,b,sb)in combinations(observed,2):
  duplicates+=len(a&b)
  if not a&b:distinct_pairs+=1
  if sa is not None and sb is not None and (sa['namespace'],sa['unit'],sa['provenance_sha256'])==(sb['namespace'],sb['unit'],sb['provenance_sha256']):
   if set(sa['labels'])&set(sb['labels']):known_overlap+=1
   else:attested_disjoint+=1
  else:unknown+=1
 status='duplicate_declared_observed_atoms'if duplicates else 'evidence_input_review_required'if evidence_inputs else 'shared_uncertainty_review_required'if any(n>1 for n in uncertain_refs.values()) else 'no_observed_likelihoods'if not observed else 'overlap_review_required'if len(observed)>1 else 'structurally_consistent_claims'
 return {'schema':'ObservationOwnershipAuditV2','status':status,'scientifically_qualified':False,'verified_independence':False,'independent_factorization_authenticated':False,'source_admission_authenticated':False,'likelihood_factors':len(observed),'nonlikelihood_factors':len(ids)-len(observed),'observed_claims':atom_claims,'unique_observed_atoms':len(all_atoms),'joint_internal_atom_reuses':joint_reuse,'cross_factor_duplicate_atom_pairs':duplicates,'distinct_observation_factor_pairs':distinct_pairs,'unknown_sampling_overlap_pairs':unknown,'declared_disjoint_sampling_pairs':attested_disjoint,'known_sampling_overlap_pairs':known_overlap,'shared_input_references':inputs,'observed_lineage_input_references_requiring_review':evidence_inputs,'reused_uncertain_input_sources':sum(n>1 for n in uncertain_refs.values()),'joint_law_authenticated':False,'model_vector_binding_authenticated':False,'sampling_attestations_authenticated':False,'source_error_dependence_unknown_pairs':len(observed)*(len(observed)-1)//2,'manifest_sha256':digest(manifest),'scope':'metadata consistency under caller-supplied canonical field/lineage and sampling attestations; not likelihood normalization or qualification'}

def audit_a2018_aggregate(path,expected_sha256):
 from pathlib import Path
 raw=Path(path).read_bytes();need(sha(expected_sha256)and hashlib.sha256(raw).hexdigest()==expected_sha256,'frozen_aggregate_pin_mismatch')
 v=json.loads(raw);c=v['counts'];families=v['factor_families'];need(sum(x['factor_count']for x in families)==66,'aggregate_factor_count')
 need(c['survey_blocks']==65 and c['survey_record_rows']==7968 and c['poll_blocks']==1 and c['poll_rows']==274 and c['factor_membership_rows']==8242,'aggregate_counts')
 need(c['official_count_observations']==c['participation_report_observations']==0,'aggregate_empty_families')
 need(v['target']['poll_row_count']==20 and v['target']['survey_block_count']==v['target']['official_count_observation_count']==v['target']['participation_report_count']==0,'aggregate_target')
 need([x['mass']for x in v['parent_masses']]==[.125]*8 and not v['plan_status']['forecast_qualified'],'aggregate_scientific_hold')
 return {'schema':'FrozenA2018AggregateConsistencyV1','likelihood_factors':66,'memberships':8242,'survey_blocks':65,'survey_memberships':7968,'poll_blocks':1,'poll_rows':274,'target_poll_rows':20,'scientifically_qualified':False,'record_ownership_audited':False,'sampling_overlap':'unknown','frozen_aggregate_sha256':expected_sha256,'scope':'source-pinned aggregate consistency only; no row-level ownership inference'}
