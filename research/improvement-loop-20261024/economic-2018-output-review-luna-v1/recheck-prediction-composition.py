import sys,importlib.util,json
from pathlib import Path
import numpy as np,h5py
from scipy.special import expit
sys.dont_write_bytecode=True
root=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024')
owner=Path('/home/hayden/projects/wethervane-redesign-20261006/integration')
sys.path.insert(0,str(owner))
path=root/'economic-experiment-v1/prepare.py'
spec=importlib.util.spec_from_file_location('review_economic_prepare',path);prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
module,compiled,frame,scaler,evidence=prep.compile_fold(2018,prep.load_macro())
target=frame.loc[frame.cycle.eq(2018)].copy()
run=root/'economic-execution-v1/attempt-v1/2018'
plan=json.loads((root/'economic-execution-v1/prepared-v2/plan.json').read_text())
results={}
for arm in ['H0','F0','E01','E01_sd025']:
 job=next(j for j in plan['jobs'] if j['target']==2018 and j['arm']==arm)
 recipe=module.Recipe.model_validate(job['recipe'])
 outcomes=prep.earlier_outcomes(2018)
 training=module.compile_training(recipe,frame,outcomes,compiled['E01' if arm.startswith('E01') else arm].admission)
 offset,design=module._design(training.recipe,target,training.admission)
 z=np.load(run/arm/'predictions.npz',allow_pickle=False)
 saved=np.asarray(z['logits']); ids=z['race_ids'].tolist()
 assert ids==target.race_id.tolist(),(arm,ids[:2],target.race_id.tolist()[:2])
 max_abs=0.; max_shared=0.; max_share=0.
 for c in range(4):
  with h5py.File(run/arm/f'chain-{c}/raw-bank.nc','r') as f:
   p=f['posterior']; beta=np.asarray(p['beta'][0]); tc=np.asarray(p['tau_cycle'][0]); tr=np.asarray(p['tau_race'][0])
  loc=offset+beta@design.T
  pp=z['posterior_draw'];
  # Reconstruct each saved joint predictive logit from its exact retained row and shocks.
  draw=z['posterior_draw']
  expected=loc[draw]+tc[draw,None]*z['new_cycle_z'][c,:,None]+tr[draw,None]*z['new_race_z'][c]
  delta=np.abs(expected-saved[c])
  max_abs=max(max_abs,float(delta.max()))
  shared_component=saved[c]-loc[draw]-tr[draw,None]*z['new_race_z'][c]
  target_shared=tc[draw]*z['new_cycle_z'][c]
  max_shared=max(max_shared,float(np.max(np.abs(shared_component-target_shared[:,None]))))
  max_share=max(max_share,float(np.max(np.abs(expit(expected)-z['shares'][c]))))
 results[arm]={'target_races':len(ids),'model_design_shape':list(design.shape),'max_abs_reconstructed_logit_error':max_abs,'max_abs_shared_cycle_component_error':max_shared,'max_abs_share_transform_error':max_share,'joint_row_identity':'Each of 5 replicate logits per retained posterior row equals offset + beta design + tau_cycle * one shared cycle draw across all races + tau_race * race-specific Student-t draw; race order exactly equals frozen 2018 target frame.'}
print(json.dumps(results,indent=2))
