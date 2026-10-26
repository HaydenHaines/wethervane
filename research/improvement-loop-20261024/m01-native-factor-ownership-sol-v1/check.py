"""Read source and existing bank coordinates; fixed-state arithmetic only. Writes HERE."""
import ast, collections, csv, hashlib, json, math
from pathlib import Path
import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import multivariate_normal
HERE=Path(__file__).resolve().parent
LAB=HERE.parent
PINS={}
def pin(p):
 p=Path(p); h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 PINS[str(p)]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
 return PINS[str(p)]
def read(rel):
 p=LAB/rel;pin(p)
 return json.loads(p.read_text()) if p.suffix=='.json' else list(csv.DictReader(p.open()))
def extract(p,names,ns):
 pin(p);tree=ast.parse(p.read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names];assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 return {n.name:[n.lineno,n.end_lineno] for n in nodes}
compat=read('momentum-preflight-bank-compatibility-sol-v1/compatibility.json')
plan=read('economic-experiment-v1/prepared-v1/plan.json')
ledger=[];summaries=[];bank_records=[]
for bank in compat['banks']:
 y,arm=bank['year'],bank['arm']; training=bank['training_race_ids'];fold=next(f for f in plan['folds'] if f['target_cycle']==y)
 assert training==fold['arms'][arm]['training_race_ids'];assert bank['law_hash']==fold['arms'][arm]['law_hash']
 poll=read(f'momentum-calibration-preparation-sol-v1/prepared-v1/{y}/selected-training-polls.csv');counts=collections.Counter(r['race_id'] for r in poll)
 chain_receipts=[]
 for chain,bp in enumerate(bank['raw_banks']):
  p=Path(bp['path']);assert pin(p)['sha256']==bp['sha256']
  with h5py.File(p) as f:
   ids=[v.decode() if isinstance(v,bytes) else str(v) for v in f['observed_data/source_race'][:]]
   obs=f['observed_data/historical_logit'][:];coeff=[v.decode() if isinstance(v,bytes) else str(v) for v in f['posterior/coefficient'][:]]
   assert ids==training;assert coeff==bank['coefficient_names']; assert obs.shape==(len(ids),);assert np.isfinite(obs).all()
   receipt={'chain':chain,'events':len(ids),'observed_logit_sha256':hashlib.sha256(obs.tobytes()).hexdigest(),'raw_bank':str(p)}
   chain_receipts.append(receipt)
 bank_records.append({'target':y,'arm':arm,'law_hash':bank['law_hash'],'raw_chains':chain_receipts})
 for key in sorted(set(training)|set(counts)):
  ledger.append({'target_cycle':y,'arm':arm,'canonical_event_key':key,'in_raw_bank_observed_data':key in training,'calibration_rows':counts[key],'marginal_owner_candidate':'historical_logit' if key in training else '', 'conditional_use_candidate':'Gaussian poll-logit residual given observed event logit' if key in counts else '', 'complete_native_binding':'missing'})
 summaries.append({'target_cycle':y,'arm':arm,'raw_bank_training_events':len(training),'used_calibration_events':len(counts),'shared':len(set(training)&set(counts)),'calibration_rows':len(poll)})
# Direct unmodified objective AST: no optimizer import or invocation.
source=LAB/'calibration-consumer-v1/snapshot-source/src/forecasting/poll_covariance_v1.py'
ns={'np':np,'cho_factor':cho_factor,'cho_solve':cho_solve,'COMPONENTS':('independent','cycle','race','house','movement')}
lines=extract(source,{'gaussian_error_objective'},ns)
checks=[]
for year in (2018,2020,2022,2024):
 rows=read(f'momentum-calibration-preparation-sol-v1/prepared-v1/{year}/selected-training-polls.csv')
 # Real three-row subset, explicitly not the complete vector or fitted parameters.
 selected=rows[:3];q=np.array([float(r['dem_share']) for r in selected]);truth=np.array([float(r['actual_dem_share']) for r in selected]);d=np.array([float(r['days_to_election']) for r in selected]);cy=np.array([r['cycle'] for r in selected]);ra=np.array([r['race_id'] for r in selected]);ho=np.array([r['pollster_key'] for r in selected])
 samecy=cy[:,None]==cy; samera=(ra[:,None]==ra)&samecy
 kernels=np.array([np.eye(3),samecy,samera,(ho[:,None]==ho)&samecy,samera*np.minimum.outer(d,d)/30],float)
 scales=np.array([.1,.06,.06,.03,.05]);cov=np.einsum('k,kij->ij',scales**2,kernels); residual=np.log(q/(1-q))-np.log(truth/(1-truth));x=np.log(scales)
 val,grad=ns['gaussian_error_objective'](x,[(residual,kernels)],prior_medians=scales,prior_log_sd=.7)
 direct=-multivariate_normal.logpdf(residual,mean=np.zeros(3),cov=cov);assert abs(val-direct)<1e-10
 # Integrate normalized conditional in whitened coordinates, including dx=det(L) dz.
 # Tensor Gauss-Hermite exactly integrates constant importance weight of normal law.
 from numpy.polynomial.hermite import hermgauss
 nodes,weights=hermgauss(5);L=np.linalg.cholesky(cov);integral=0.
 for idx in np.ndindex(5,5,5):
  z=np.sqrt(2)*nodes[list(idx)];v=L@z
  logratio=multivariate_normal.logpdf(v,mean=np.zeros(3),cov=cov)+np.linalg.slogdet(L)[1]+.5*float(z@z)+1.5*math.log(2*math.pi)
  integral+=float(np.prod(weights[list(idx)])/math.pi**1.5*np.exp(logratio))
 assert abs(integral-1)<1e-10
 eps=1e-6;fd=[]
 for k in range(5):
  h=np.eye(5)[k]*eps;f=lambda v:ns['gaussian_error_objective'](v,[(residual,kernels)],prior_medians=scales,prior_log_sd=.7)[0]
  fd.append((f(x+h)-f(x-h))/(2*eps))
 assert np.max(np.abs(grad-fd))<1e-6
 # On share coordinates density requires the logit Jacobian. Missing it is a distinct law.
 jac=-float(np.log(q*(1-q)).sum());assert jac>0
 checks.append({'target':year,'observation_ids':[r['observation_id'] for r in selected],'fixed_scales':scales.tolist(),'native_objective_nll':val,'reference_nll':direct,'conditional_normalization_whitened_quadrature':integral,'gradient_max_error':float(np.max(np.abs(grad-fd))),'share_coordinate_log_jacobian':jac,'scope':'actual selected rows, supplied scales, three-row subset only'})
# Ownership contract must fail duplicate marginal owner, while conditioned reads add no owner.
def validate(owners,conditional_reads):
 assert all(len(v)==1 for v in owners.values()),'multiple marginal owners'
 assert all(k in owners for k in conditional_reads),'unbound conditioned event'
 return True
rejections=[]
validate({'event':['historical_logit']},['event'])
for owners,reads in [({'event':['historical_logit','second_outcome_density']},['event']),({},['event'])]:
 try: validate(owners,reads)
 except AssertionError as e: rejections.append(str(e))
assert len(rejections)==2
for y in (2018,2020,2022,2024):
 a={r['canonical_event_key'] for r in ledger if r['target_cycle']==y and r['arm']=='H0' and r['in_raw_bank_observed_data']};b={r['canonical_event_key'] for r in ledger if r['target_cycle']==y and r['arm']=='F0' and r['in_raw_bank_observed_data']};assert a==b
with (HERE/'per-arm-event-ledger.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
owner=Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
for p in [owner/'research/senate-stack-integration-20261020/modular-count-graph-astra-v1/components.py',owner/'research/coordination-frontend-20261026/m01-count-binding-followup-v2.md',LAB/'m01-ownership-followup-sol-v1/root-review-v1.md',LAB/'m01-ownership-followup-sol-v1/luna-review-v1.json',LAB/'checkpoints/continuation-v49.json']:pin(p)
factory=LAB/'economic-execution-v1/prepared-v1/snapshots/03-aggregate_component_factory_v1.py';pin(factory)
for p in [Path(__file__),LAB/'m01-ownership-followup-sol-v1/verification.json',LAB/'m01-native-supplied-state-parity-sol-v1/result.json']:pin(p)
result={'schema':'M01NativeFactorOwnershipEvidenceV1','status':'supplied_state_passed_native_binding_missing','summaries':summaries,'bank_coordinate_authentication':bank_records,'objective_source_lines':lines,'conditional_checks':checks,'negative_checks':rejections,'2024_arm_training_symmetric_difference':[],'2024_prior_review_correction':'Pinned compatibility, prepared plan and all eight 2024 raw chain observed_data coordinates show H0=F0=387. Prior prose H0=357 and 30-key difference is incorrect.','pins':PINS,'limits':['Raw-bank byte identity and stored event coordinates checked; source event truth, original vintage and full compiler ownership not authenticated.','Historical StudentT outcome factor is aggregate factory training law; it is not a native modular count-graph outcome factor.','Gaussian normalization is on poll-logit measure at fixed scales; supplied three-row subset is not complete selected native Problem.','No sampler, optimizer, fit, score, source admission or promotion.']}
(HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'bank_chains':sum(len(b['raw_chains'])for b in bank_records),'ledger_rows':len(ledger),'2024_sets_equal':True,'conditional_checks':len(checks)}))
