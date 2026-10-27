"""Source-bound partial national assembly. No inference or implicit menu closure."""
import copy, csv, hashlib, importlib.util, json, re, sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent; B=H.parent; ROOT=H.parents[2]
CUTOFF='2026-10-25T06:01:29Z'
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def load(name,p,package=False):
 if name in sys.modules:return sys.modules[name]
 spec=importlib.util.spec_from_file_location(name,p,submodule_search_locations=[str(p.parent)] if package else None);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ident(s):return re.sub('[^A-Za-z0-9_]','_',s)
def logit(x):return float(np.log(x/(1-x)))
def verify_pins():
 for r in read(H/'source-pins.json'):
  if pin(ROOT/r['path'])!=r:raise ValueError('Changed source: '+r['path'])
def package():return load('national_assembly_ranked_bridge',H/'package.py',True)
def build_inputs():
 verify_pins();p=B/'national-observation-components-astra-v1/prepared-v2';menu=read(B/'national-menu-integration-astra-v1/index.json');survey=read(p/'survey-components.json');bindings=read(p/'contest-source-bindings.json');report=read(p/'participation-report-components.json');physical=read(p/'physical-event-components.json')
 gap=read(B/'national-menu-gap-recovery-sol-v1/proposal.json');replacements={m['state']:m for m in gap['menus']};menu['menus']=[copy.deepcopy(replacements.get(m['state'],m)) for m in menu['menus']]
 features=read(ROOT/'research/national-senate-forecast-20261020/input-preparation-sol-v1/candidate-inputs.json');fr={r['state']:r for r in features['races']}
 feature_ids={c['source_support'].get('source_record_id'):c for r in features['races'] for c in [r['first_contender'],r['second_contender']]}
 historical=list(csv.DictReader((ROOT/'research/polling-layer-implementation-20261020/full-chamber-intake-sol/races.csv').open()));race={r['race_id']:r for r in historical}
 national=list(csv.DictReader((ROOT/'research/fundamentals-iteration-20261020/panel-sol/sources/presidential-national-totals.csv').open()));ns={int(r['cycle']):float(r['dem_share']) for r in national}
 statefips={e['state']:e['state_fips'] for e in physical};events=[dict(id=e['id'],state=e['state_fips'],date=e['date'],cycle=e['cycle'],midterm=e['cycle']==2022,president_party=1 if e['cycle']==2022 else -1,role='training') for e in physical]
 events += [dict(id='event_'+s.lower()+'2026_general',state=f,date='2026-11-03',cycle=2026,midterm=True,president_party=-1,role='prediction') for s,f in sorted(statefips.items())]
 policy=dict(schema='NationalCountAssemblySelectionV1',selected_by='root task steering',population_cut='population_cut_v1',purpose='current2026 conditional research with retrospective harmonized2020 survey training',generic_choice='iid eight-role baseline; no TX662 equivalence',CES_epsilon=dict(kind='beta',alpha=1,beta=49,sharing_key='CES2020'),CES_rho='uniform over observed report labels; not calibrated',sensitivity_beta=[[1,199],[1,9]],EAVS_error_law='release Normal(0,.03); state Normal(0,.05); centered log bias; NegativeBinomial alpha100',EAVS_variance='unknown source variance stays null metadata, not zero',dependence='conditional CES/EAVS observation approximation',EAVS2022='fitted training, not untouched validation',MN_unnamed_writeins='coarsened candidate-count/report binding required; unresolved hold, NEVER noncandidate marks',fit_admitted=False,public_qualified=False)
 admission=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 sources=[dict(id='CES2020_historical_training',sha256=pin(p/'survey-components.json')['sha256'],available_utc='2026-10-25T04:00:00Z',role='survey'),dict(id='historical_covariates',sha256=pin(ROOT/'research/polling-layer-implementation-20261020/full-chamber-intake-sol/races.csv')['sha256'],available_utc=CUTOFF,role='covariates'),dict(id='current_covariates',sha256=pin(ROOT/'research/national-senate-forecast-20261020/input-preparation-sol-v1/candidate-inputs.json')['sha256'],available_utc=CUTOFF,role='covariates')]
 sources += [dict(id='assembly_source_'+str(i),sha256=pin(B/path)['sha256'],available_utc=CUTOFF,role='covariates') for i,path in enumerate(['national-menu-integration-astra-v1/index.json','national-menu-gap-recovery-sol-v1/proposal.json','national-incumbency-bindings-sol-v2/bindings.json','national-integration-root-v1/selection.json','national-integration-root-v1/menu-successor-selection.json'])]
 sources += [dict(id=s,sha256=pin(p/'participation-report-components.json')['sha256'],available_utc=CUTOFF,role='participation_report') for s in ['EAVS2020','EAVS2022']]
 contests=[];crosswalk={};historical_holds=[]
 for b in bindings:
  ids=b['candidate_id_map'];byid={c['candidate_id']:c for c in b['named_candidates']};crosswalk.update({v:ident(v) for v in ids.values()});candidates=[dict(id=ident(v),bloc=byid[k]['ballot_party'] if byid[k]['ballot_party'] in ['D','R'] else 'other',incumbent=byid[k]['incumbent']) for k,v in ids.items()]
  c=dict(id=b['contest_id'],event_id=b['event_id'],kind=b['source_race_id'].split(':')[-1],rule='single_mark_plurality',relative_lean=logit(float(race[b['source_race_id']]['prior_dem_share']))-logit(ns[2016]),national_pres_logit=logit(ns[2016]),covariate_source_id='historical_covariates',candidates=candidates,noncandidate_categories=[dict(id=k+'_'+b['contest_id'],kind=k) for k in ['nonresponse','blank','invalid']]);contests.append(c)
  historical_holds.append(dict(contest_id=c['id'],source=b,scope='named-person retrospective report proxy; anonymous candidate count support unresolved; no official final-count factor',winner_rule_qualification=False))
 for s in survey:
  for g in s['groups']:g['candidate_ids']=[crosswalk[x] for x in g['candidate_ids']]
 channels={};bycontest={c['id']:c for c in contests};em=sys.modules[package().__name__+'.emission']
 for s in survey:
  ids=[c['id'] for c in bycontest[s['contest_id']]['candidates']]
  channels[s['id']]=dict(schema=em.VERSION,source_id=s['source_id'],candidate_ids=ids,report_label_ids=[g['id'] for g in s['groups']],candidate_default_labels=[next(g['id'] for g in s['groups'] if cid in g['candidate_ids']) for cid in ids],contamination_probabilities=[1/len(s['groups'])]*len(s['groups']),epsilon=policy['CES_epsilon'])
 manifest=[];incumbency=[]
 resolved=read(B/'national-incumbency-bindings-sol-v2/bindings.json')
 resolved_rows={(r['setting_id'],r['candidate_id']):r for r in resolved['rows']}
 for m in menu['menus']:
  values=[];unknown=[];holder=[c for c in m['candidates'] if feature_ids.get(c['candidate_id'],{}).get('incumbent') is True]
  for c in m['candidates']:
   f=feature_ids.get(c['candidate_id']);resolved_row=resolved_rows.get((m['setting_id'],c['candidate_id']));value=resolved_row['incumbent'] if resolved_row else f['incumbent'] if f else False if len(holder)==1 else None
   reason=resolved_row if resolved_row else 'exact source-ID feature join' if f else 'distinct named menu person excludes uniquely source-ID bound same-seat incumbent' if value is False else 'unbound same-seat incumbency'
   incumbency.append(dict(setting_id=m['setting_id'],candidate_id=c['candidate_id'],incumbent=value,basis=reason,source_feature=f,holder_source_id=holder[0]['candidate_id'] if holder else None))
   if value is None:unknown.append(c['candidate_id'])
   party=(c['party_literal'] or '').lower();bloc='D' if party in ['d','dem','democrat','democratic','democratic party','democratic-farmer-labor'] else 'R' if party in ['r','rep','republican','republican party'] else 'other'
   values.append(dict(id=ident(c['candidate_id']),bloc=bloc,incumbent=value))
  eligible=m['state'] in ['TX','AR','CO','DE','SD','OK','LA','MN','MS','ME','GA','MA','OR','RI','VA','WY','IA','SC','NH','OH'] and not unknown
  entry=dict(setting_id=m['setting_id'],source_menu=m,assembled=eligible,unbound_incumbency_ids=unknown,holds=[] if eligible else ['source/menu/rule or exact incumbent binding remains unresolved; no partial-candidate substitution'],candidate_model_ids={c['candidate_id']:ident(c['candidate_id']) for c in m['candidates']})
  if m['state']=='MN':entry['holds'].append(policy['MN_unnamed_writeins'])
  entry['source_qualified_complete']=False
  entry['conditional_research_scope']='Source-bound named menu plus selected rule/domain scenario; existing source/legal/tie limitations retained, no posterior/public qualification'
  if m['state']=='AK':entry['holds'].append('Open-person ranked support unimplemented; no finite closure by omission')
  if m['state']=='MN':
   entry['predecessor_conditions']=copy.deepcopy(m['conditions']);entry['source_menu']['conditions']=[policy['MN_unnamed_writeins'] if 'noncandidate marks' in x else x for x in m['conditions']]
  if m['state']=='SC':
   entry['selected_ballot_role_override']='Root-selected Active Name-on-Ballot→printed; predecessor unknown retained in original proposal'
   for candidate in entry['source_menu']['candidates']:candidate['predecessor_ballot_role']=candidate['ballot_role'];candidate['ballot_role']='printed'
  manifest.append(entry)
  if eligible:
   r=fr[m['state']];sh=r['presidential_dem_votes']/(r['presidential_dem_votes']+r['presidential_rep_votes']);cid='contest_'+m['state'].lower()+'2026_'+m['event_kind']
   contests.append(dict(id=cid,event_id='event_'+m['state'].lower()+'2026_general',kind=m['event_kind'],rule='single_mark_plurality',relative_lean=logit(sh)-logit(ns[2024]),national_pres_logit=logit(ns[2024]),covariate_source_id='current_covariates',candidates=values,noncandidate_categories=[dict(id=k+'_'+cid,kind=k) for k in ['nonresponse','blank','invalid']]))
 # A runoff is a separate physical event, but both contests resolve one seat.
 if any(c['id']=='contest_ga2026_regular' for c in contests):
  general=next(c for c in contests if c['id']=='contest_ga2026_regular');later=copy.deepcopy(general);later['id']='contest_ga2026_runoff';later['event_id']='event_ga2026_runoff';later['noncandidate_categories']=[dict(id=k+'_ga2026_runoff',kind=k) for k in ['nonresponse','blank','invalid']];contests.append(later);events.append(dict(id='event_ga2026_runoff',state='13',date='2026-12-01',cycle=2026,midterm=True,president_party=-1,role='prediction'))
 me=read(B/'ranked-report-measurement-astra-v1/ME-conditional-v2/problem.json');mc=next(c for c in contests if c['id']=='contest_me2020_regular');mc['noncandidate_categories']=copy.deepcopy(me['contests'][0]['noncandidate_categories'])
 reference=read(B/'actual-pilot-source-sol-v1/national-reference-v2/reference.json')
 value=dict(schema='senate_national_mixed_report_problem_v1',purpose='actual_current',cutoff_utc=CUTOFF,admission_id=admission,reference={k:reference[k] for k in ['id','weights']},states=sorted(statefips.values()),sources=sources,events=events,contests=contests,surveys=survey,report_channels=channels,official_counts=[],participation_reports=[{k:r[k] for k in ['id','source_id','event_id','release_id','count']} for r in report],polls=[],ranked_measurement=copy.deepcopy(me['ranked_measurement']))
 if any(c['id']=='contest_me2026_regular' for c in contests):
  rules=load('national_assembly_me_source',B/'national-mixed-rule-integration-astra-v1/bindings.py');source=dict(id='ME2026_selected_rule_style',sha256=pin(B/'national-mixed-rule-integration-astra-v1/source-bindings.json')['sha256'],available_utc=CUTOFF,role='rule_style');c=next(c for c in contests if c['id']=='contest_me2026_regular');c['noncandidate_categories']=[x for x in c['noncandidate_categories'] if x['kind']!='blank'];entry=rules.me_ranked_entry(value,c['id'],source,office_ballot_categories=[a['id'] for a in c['noncandidate_categories'] if a['kind']=='invalid']);value['ranked_measurement']['sources'].append(source);value['ranked_measurement']['ranked'].append(entry)
 pollpack=read(B/'national-current-poll-components-astra-v1/components.json')
 for record in pollpack['source_pins']:
  if pin(ROOT/record['path'])!=record:raise ValueError('Changed numerical poll source')
 value['sources']+=copy.deepcopy(pollpack['sources']);value['polls']=copy.deepcopy(pollpack['polls']);value['ranked_measurement']['poll_channels']=copy.deepcopy(pollpack['poll_channels'])
 return value,dict(poll_component_pin=pin(B/'national-current-poll-components-astra-v1/components.json'),poll_ledger=copy.deepcopy(pollpack['ledger']),policy=policy,settings=manifest,historical_support=historical_holds,incumbency=incumbency,original_report_metadata=report,root_selection=pin(B/'national-integration-root-v1/selection.json'),menu_successor_selection=pin(B/'national-integration-root-v1/menu-successor-selection.json'),successor_menu_proposal=pin(B/'national-menu-gap-recovery-sol-v1/proposal.json'),incumbency_binding=pin(B/'national-incumbency-bindings-sol-v2/bindings.json'),original_menu_identity=pin(B/'national-menu-integration-astra-v1/index.json'),whole_chamber=menu['holdovers'],chamber_scenarios=menu['chamber_scenarios'],candidate_crosswalk=crosswalk,national_complete=False,required_settings=35,assembled_settings=sum(x['assembled'] for x in manifest),no_posterior=True)
def make_extension(problem):
 pkg=package();inputs=sys.modules[pkg.__name__+'.inputs_v3'];core,rank=inputs.auxiliary_rule_projection(problem.to_dict());v=problem.to_dict();contests={c['id']:c for c in v['contests']}
 if 'contest_ga2026_runoff' in contests:
  source=dict(id='GA2026_selected_runoff_rule',sha256=pin(B/'national-mixed-rule-integration-astra-v1/source-bindings.json')['sha256'],available_utc=CUTOFF,role='rule_style');rank['sources'].append(source);rank['runoffs'].append(dict(general_contest_id='contest_ga2026_regular',runoff_contest_id='contest_ga2026_runoff',rule_source_id=source['id'],eligible_finalists=[c['id'] for c in contests['contest_ga2026_regular']['candidates']],observations=[]))
 openstates=['MA','OR','RI','VA','WY','IA','SC','NH'];scopes=[];items=[];sources=[]
 for st in openstates:
  cid='contest_'+st.lower()+'2026_regular'
  if cid not in contests:continue
  c=contests[cid];e=next(e for e in v['events'] if e['id']==c['event_id']);sid='open_support_'+st;sourcepath=B/('national-menu-gap-recovery-sol-v1/proposal.json' if st in ['SC','NH'] else 'national-menu-integration-astra-v1/index.json');sources.append(dict(id=sid,sha256=pin(sourcepath)['sha256'],available_utc=CUTOFF,role='open_person_support'));scopes.append(dict(id='person_scope_'+st,state=e['state'],known_person_ids=[x['id'] for x in c['candidates']],source_id=sid,concentration_group='national_open_person',identity_policy='shared_unidentified_natural_persons_unresolved_eligibility_v1'));items.append(dict(contest_id=cid,scope_id='person_scope_'+st,observations=[]))
 deps=sys.modules[pkg.__name__+'.dependencies'];mixed=deps.package('mixed-count-graph-astra-v1');openjoint=sys.modules[mixed.__name__+'.references'].open_joint
 op=dict(schema='OpenPersonPluralityV1',core_problem_identity=core.identity,cutoff_utc=CUTOFF,sources=sources,scopes=scopes,contests=items,priors=openjoint.default_priors()) if items else None
 ext=dict(schema='MixedCountExtensionV1',core_problem_identity=core.identity,cutoff_utc=CUTOFF,ranked_runoff=rank,open_person=op);pkg.validate_mixed_extension(problem,ext);return ext

def project(problem):
 verify_pins();adapter=load('national_assembly_population',B/'national-population-cutoff-binding-astra-v1/adapter.py');rp=B/'national-population-bank-root-v1/attempt1/receipt.json';port,bank,receipt=adapter.load_issued(pin(rp));v=problem.to_dict()
 if port._time(bank.pep.cutoff_utc)>port._time(v['cutoff_utc']):raise ValueError('Future population cut')
 events=[{k:e[k] for k in ['id','state','date']} for e in v['events']];exposure,mapping=port.event_projection(bank,events)
 parents=[package().validate_parent(problem,dict(schema='senate_count_joint_parent_v1',problem_identity=problem.identity,upstream_identity=bank.identity,parent_id=pid,row_id=i,mass=float(mass),event_ids=[e['id'] for e in events],cell_ids=list(port.CELLS[:24]),exposure=exposure[i,:,:24].tolist())).to_dict() for i,(pid,mass) in enumerate(zip(bank.parent_ids,bank.masses,strict=True))]
 return parents,dict(population_cut='population_cut_v1',population_cutoff_utc=bank.pep.cutoff_utc,political_cutoff_utc=v['cutoff_utc'],original_bank_identity=bank.identity,receipt=pin(rp),parent_ids=list(bank.parent_ids),masses=bank.masses.tolist(),no_resampling=True,noncitizen_adult_total=exposure[:,:,24].tolist(),resident_adult_total=exposure.sum(axis=2).tolist(),mapping=mapping)
if __name__=='__main__':
 out=H/sys.argv[1];out.mkdir(exist_ok=False);value,manifest=build_inputs();problem=package().validate_problem(value);extension=make_extension(problem);parents,binding=project(problem)
 for name,x in [('extension.json',extension),('problem.json',problem.to_dict()),('manifest.json',manifest),('parents.json',parents),('population-binding.json',binding)]:write(out/name,x)
 print('ASSEMBLED',len(value['surveys']),sum(len(s['rows']) for s in value['surveys']),len(value['participation_reports']),len(value['events']),manifest['assembled_settings'],len(parents),flush=True)
