"""Read-only bounded incumbent inventory; deliberately no model/provider import or fit."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;LAB=HERE.parent
ROOT=Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
BASE=ROOT/'research/senate-stack-integration-20261020'
PINS={};AUTH=[]
def pin(p,expected=None):
 p=Path(p);size=p.stat().st_size
 if size>8_000_000:raise ValueError('bounded audit refuses large file')
 digest=hashlib.sha256(p.read_bytes()).hexdigest();record=dict(path=str(p),sha256=digest,bytes=size);PINS[str(p)]=record
 if expected is not None:
  if digest!=expected:raise ValueError('source hash mismatch: '+str(p))
  AUTH.append(str(p))
 return record
def read(p,expected=None):pin(p,expected);return json.loads(Path(p).read_text())
def bound(record):return ROOT/record['path'] if not Path(record['path']).is_absolute() else Path(record['path'])
def required(condition,message):
 if not condition:raise ValueError(message)
plan=read(BASE/'actual-pilot-preparation-sol-v1/prepared-v4/plan.json','0b35a7054b8ec24b2269cf3e06bf37f95b70a6ab1553290d5253067c217af723')
problem=read(bound(plan['problem']),plan['problem']['sha256']);admission=read(bound(plan['admission']),plan['admission']['sha256'])
receipt=read(BASE/'actual-pilot-preparation-sol-v1/upstream-v4/receipt.json')
required(receipt['prepared_plan']['sha256']==PINS[str(BASE/'actual-pilot-preparation-sol-v1/prepared-v4/plan.json')]['sha256'],'plan receipt mismatch')
up=BASE/'actual-pilot-preparation-sol-v1/upstream-v4'
for name,sha in receipt['files'].items():pin(up/name,sha)
parents=read(up/'parents.json');meta=read(up/'bank-metadata.json')
required(len(parents)==8 and [p['mass']for p in parents]==receipt['parent_masses']==[.125]*8,'original parent masses mismatch')
required(len({p['parent_id']for p in parents})==8,'parent alias')
required(all(p['problem_identity']==plan['problem_identity']for p in parents),'parent problem mismatch')
witness=read(BASE/'modular-count-graph-astra-v1/graph-inventory-v1.json');case=next(r for r in witness['cases'] if r['case']=='actual_TX_role_graph_only')
for path,sha in witness['sources'].items():pin(ROOT/path,sha)
for item in witness['input_pins']:
 if 'actual-pilot' in item['path'] or 'candidate-role-source' in item['path']:pin(ROOT/item['path'],item['sha256'])
pin(ROOT/witness['script']['path'],witness['script']['sha256'])
cert=read(BASE/'modular-count-graph-astra-v1/equivalence-certificate-v1.json')
for item in cert['evidence']:
 if item['kind']=='complete_graph_inventory':pin(ROOT/item['path'],item['sha256'])
contract=case['modular']['contract'];required(contract['problem_identity']==plan['problem_identity'],'witness problem mismatch')
required(contract['parent_mass']==parents[0]['mass'] and contract['parent_id']==parents[0]['parent_id'],'witness parent mismatch')
required(contract['inference_target']=='population_cut_v1' and contract['parent_mass_used_in_logp'] is False,'population cut mismatch')
# The actual source inventory witness is graph construction evidence, not fresh logp evaluation.
mapdir=BASE/'actual-pilot-source-sol-v1/observation-maps-v2'
mapreceipt=read(mapdir/'receipt.json')
for name,sha in mapreceipt['files'].items():
 p=mapdir/name
 if p.suffix=='.json':pin(p,sha) # skip Parquet source payload, no respondent record export
idmap=read(mapdir/'TX2026-poll-observation-ID-map.json')
original=read(mapdir/'TX2026-six-LV-poll-original-observations.json')
consumer=LAB/'calibration-consumer-v1';cm=read(consumer/'input-manifest.json','0fbec6366c2640f30a17c2fc576b87303d9f4065a632b1488b9a72311345fcb8')
for item in cm['files']:pin(consumer/item['file'],item['sha256'])
component=read(consumer/'inputs/poll-component.json');cp=read(consumer/'inputs/plan.json');coords=read(consumer/'inputs/target-coordinates.json')
required(problem['polls']==[component],'incumbent poll component changed')
required(len(component['rows'])==6,'complete vector mismatch')
required(idmap==coords['source_map'],'source ID mapping mismatch')
required([r['observation_id']for r in component['rows']]==[r['model_observation_id']for r in idmap] and [r['original_observation_id']for r in idmap]==coords['observation_ids'],'coordinate order mismatch')
# Source-derived one-record tokens; canonical physical identities are explicit mappings,
# not assertions of respondent disjointness or scientific independence.
atoms=[]
for o in problem['official_counts']:
 contest=next(c for c in problem['contests']if c['id']==o['contest_id']);event=next(e for e in problem['events']if e['id']==contest['event_id'])
 atoms.append(dict(atom='official_event:'+event['id'],token='official:'+o['id'],factor='official_likelihood_'+o['id'],source=o['source_id'],contest=o['contest_id']))
for r in component['rows']:atoms.append(dict(atom='poll_release_mapping:'+r['observation_id'],token='poll:'+r['observation_id'],factor='poll_likelihood_'+component['id'],source='ID-map and original observation source receipt',contest=r['contest_id']))
for r in problem['participation_reports']:atoms.append(dict(atom='participation_report:'+r['id'],token='report:'+r['id'],factor='report_likelihood_'+r['id'],source=r['source_id']))
for survey in problem['surveys']:
 for r in survey['rows']:atoms.append(dict(atom='survey_record:'+survey['source_id']+':'+r['record_id'],token='survey:'+survey['source_id']+':'+r['record_id'],factor='grouped_CES_factor_defined_by_measurement_stage',source=survey['source_id']))
required(len({a['token']for a in atoms})==len(atoms),'duplicate evidence token')
receipt_owners={}
for stage,row in case['ownership'].items():
 for factor,entry in row['factors'].items():
  for token in entry['tokens']:
   required(token not in receipt_owners,'duplicate witness factor owner')
   receipt_owners[token]=dict(stage=stage,factor=factor)
required(set(receipt_owners)=={a['token']for a in atoms},'witness evidence coverage mismatch')
for atom in atoms:atom.update(receipt_owners[atom['token']])

# Tiny posterior calibration NPZ only (under .3 MB); actual weights retained.
with np.load(consumer/'inputs/posterior-bank.npz',allow_pickle=False)as bank:
 bank_info={k:dict(shape=list(bank[k].shape),dtype=str(bank[k].dtype))for k in bank.files}
# Fail-closed contract for missing successor selection; no fabricated passing ledger.
def ready(m):
 for key in ['selected_m01_composition_pin','normalized_adapter_source_pin','canonical_training_label_binding_pin','complete_vector_owner_receipt_pin']:
  required(isinstance(m.get(key),dict) and len(m[key].get('sha256',''))==64,'missing authenticated '+key)
 required(m.get('parent_masses')==[.125]*8,'changed protected parent masses')
 required(m.get('observation_ids')==coords['observation_ids'],'changed vector order')
 return True
candidate=dict(parent_masses=[p['mass']for p in parents],observation_ids=coords['observation_ids'])
negative=[]
try:ready(candidate)
except ValueError as e:negative.append(dict(case='actual incumbent inputs without successor selection',rejection=str(e)))
# Independently test each requirement; dummy pins here are contract fixtures only.
fake={k:dict(sha256='0'*64)for k in ['selected_m01_composition_pin','normalized_adapter_source_pin','canonical_training_label_binding_pin','complete_vector_owner_receipt_pin']};fake.update(candidate)
for key in list(fake):
 bad=copy.deepcopy(fake)
 if key=='parent_masses':bad[key][0]=.25
 elif key=='observation_ids':bad[key]=list(reversed(bad[key]))
 else:bad.pop(key)
 try:ready(bad);raise AssertionError('failure not rejected')
 except ValueError as e:negative.append(dict(case=key,rejection=str(e)))
# Source integrity negative uses the real pinned bytes and wrong expected digest.
try:pin(BASE/'modular-count-graph-astra-v1/components.py','0'*64)
except ValueError as e:negative.append(dict(case='source hash drift',rejection=str(e)))
for p in [LAB/'checkpoints/continuation-v51.json',LAB/'m01-native-factor-ownership-sol-v1/FINDINGS.md',LAB/'m01-native-factor-ownership-sol-v1/root-review-v1.json',LAB/'m01-native-factor-ownership-sol-v1/luna-review-v1.json',LAB/'m01-count-measurement-prototype-sol-v1/prototype.py',LAB/'m01-native-supplied-state-parity-sol-v1/check.py',BASE/'modular-count-graph-astra-v1/graph_inventory.py',BASE/'candidate-role-source-sol-v1/prepared-v2/candidate-role-manifest.json',ROOT/'research/coordination-frontend-20261026/m01-count-binding-followup-v2.md',Path(__file__)]:pin(p)
# Save exact existing witness provider/factor ports; never describe these as M01 registration.
manifest=dict(schema='TXIncumbentNativeProviderPortManifestV1',status='actual_incumbent_source_bound_M01_successor_unselected',problem_pin=PINS[str(bound(plan['problem']))],problem_identity=plan['problem_identity'],admission_pin=plan['admission'],cutoff=problem['cutoff_utc'],composition=case['composition'],provider_stage_ownership=case['ownership'],incumbent_contract=contract,protected_parent_ids=[p['parent_id']for p in parents],protected_parent_masses=[p['mass']for p in parents],upstream_identity=receipt['upstream_identity'],parent_source=meta,calibration_plan=cp,calibration_bank_arrays=bank_info,poll_component=component,original_observation_mapping=idmap,original_observation_metadata=original,sources=problem['sources'],event_menus=[dict(id=c['id'],event=c['event_id'],candidates=c['candidates'])for c in problem['contests']],missing_successor_selections=candidate)
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(HERE/'evidence-atoms.json').write_text(json.dumps(dict(schema='TXActualSourceDerivedAtomInventoryV1',scope='Native literal tokens and source map candidates; not physical person deduplication, compiler rerun or admitted dependence ledger',atoms=atoms),indent=2)+'\n')
(HERE/'verification.json').write_text(json.dumps(dict(schema='TXIncumbentNativeInventoryCheckV1',status='incumbent_inputs_consistent_successor_fails_closed',pins=PINS,expected_pin_comparisons=len(AUTH),evidence_atoms=len(atoms),negative_checks=negative,bounded_io='Every file limited to 8 MB; no large raw source/Parquet reads, sampler/model import, result targets or owner writes',limitations=['Existing graph witness source hashes verified; graph construction/logp not rerun.','Prepared plan full authenticated_sources includes large raw files; their historical hashes retained as metadata, not rehashed here.','Dummy-pin negative contract fixtures authenticate no real adapter; actual candidate is rejected.','No selected M01 provider composition or numerical ownership receipt exists in these inspected pinned artifacts.']),indent=2)+'\n')
print(json.dumps(dict(status='incumbent_inputs_consistent_successor_fails_closed',source_comparisons=len(AUTH),atoms=len(atoms),negative_checks=len(negative))))
