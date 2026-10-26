"""ONE full actual CPU NUMBA compile/evaluation. Invoked only by bounded launcher."""
import hashlib,json,os,sys,time,traceback,warnings
from pathlib import Path
import numpy as np
import pytensor
import preflight as pf
HERE=Path(__file__).resolve().parent
start=time.monotonic()
def stage(name):
 (HERE/'full-stage.json').write_text(json.dumps(dict(stage=name,pid=os.getpid(),elapsed=time.monotonic()-start))+'\n');print(name,flush=True)
try:
 assert pytensor.config.cxx=='' and pytensor.config.numba__fastmath is False
 assert json.loads((HERE/'preflight.json').read_text())['full_attempt_permitted']
 stage('building_original_frozen_actual_graph');model=pf.build();point=model.initial_point()
 stage('constructing_full_original_logp_and_gradient');logp=model.logp(jacobian=True);gradient=model.dlogp(jacobian=True)
 stage('single_NUMBA_compile_full_logp_and_gradient')
 with warnings.catch_warnings(record=True)as notes:
  warnings.simplefilter('always')
  fn=model.compile_fn([logp,gradient],inputs=model.value_vars,point_fn=True,mode='NUMBA')
  stage('evaluating_full_gradient')
  value,derivative=fn(point);value=float(value);derivative=np.asarray(derivative)
  assert derivative.shape==(662,) and np.isfinite(derivative).all() and np.isfinite(value)
  prior=json.loads((pf.OLD/'result.json').read_text());expected=prior['full_joint_fixed_state']['multi_logp'];assert abs(value-expected)<1e-6,(value,expected)
  stage('checking_six_transformed_coordinates')
  requested=['choice_intercept','turnout_intercept','event_gamma_shape_log__','final_race_sd_log__','community_interaction_sd_log__','candidate_role_mean']
  checks=[];offset=0
  for var in model.value_vars:
   size=point[var.name].size
   if var.name in requested:
    step=1e-5;plus={k:v.copy()for k,v in point.items()};minus={k:v.copy()for k,v in point.items()}
    plus[var.name].flat[0]+=step;minus[var.name].flat[0]-=step
    numeric=float((fn(plus)[0]-fn(minus)[0])/(2*step));analytic=float(derivative[offset]);error=abs(numeric-analytic);tolerance=.01+1e-5*abs(numeric)
    assert error<tolerance,(var.name,error,tolerance)
    checks.append(dict(value_variable=var.name,flat_index=offset,within_variable_index=0,shape=list(point[var.name].shape),step=step,numeric=numeric,analytic=analytic,absolute_error=error,tolerance=tolerance))
   offset+=size
  assert len(checks)==6
 warnings_text=[str(note.message)for note in notes];(HERE/'full-warnings.json').write_text(json.dumps(warnings_text,indent=2)+'\n')
 stage('completed')
 import numba,llvmlite,pymc,arviz
 result=dict(schema='ActualM01SingleNumbaAttemptV1',status='full_gradient_and_selected_finite_differences_passed',seconds=time.monotonic()-start,pid=os.getpid(),logp=value,frozen_reference_logp=expected,logp_error=abs(value-expected),gradient_shape=list(derivative.shape),finite_coordinates=int(np.isfinite(derivative).sum()),finite_difference_checks=checks,warnings=warnings_text,fastmath=pytensor.config.numba__fastmath,cxx=pytensor.config.cxx,runtime={name:dict(version=module.__version__,path=module.__file__)for name,module in [('numpy',np),('pytensor',pytensor),('numba',numba),('llvmlite',llvmlite),('pymc',pymc),('arviz',arviz)]},scope='One actual frozen first-parent graph at one fixed point only; no fits, sampling, other-parent normalization or forecast qualification.')
 (HERE/'full-result.json').write_text(json.dumps(result,indent=2)+'\n')
except BaseException:
 (HERE/'full-failure.log').write_text(traceback.format_exc());raise
