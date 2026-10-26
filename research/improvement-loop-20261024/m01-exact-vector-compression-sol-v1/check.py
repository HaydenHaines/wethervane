"""Bounded actual-bank supplied-state arithmetic and tiny scalar graph only."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(30,35));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));resource.setrlimit(resource.RLIMIT_FSIZE,(8*1024**2,8*1024**2))
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import compression as c
HERE=Path(__file__).resolve().parent;LAB=HERE.parent;start=time.monotonic()
sys.path.insert(0,str(LAB/'m01-count-measurement-prototype-sol-v1'));import prototype
need=c.need
need(hashlib.sha256(Path(prototype.__file__).read_bytes()).hexdigest()=='244c8214b8e004bb84ad14adb7a0ac27c439bc6283705db532413a4efe015d07','prototype source changed')
case=prototype.load_frozen_current_case(LAB/'calibration-consumer-v1',target_cycle=2026,cutoff='2026-10-20T22:13:33Z',target=prototype.TARGET)
selection=LAB/'m01-native-successor-provider-sol-v1/selection.json';need(hashlib.sha256(selection.read_bytes()).hexdigest()=='03edb47049384b662cdd645053fc6083bde8fb41d972b1a07da8b71ecf6283b5','frozen native fixture selection changed');s=json.loads(selection.read_text());problem_path=Path(s['problem_path']);need(hashlib.sha256(problem_path.read_bytes()).hexdigest()==s['pins'][str(problem_path)],'actual Problem bytes changed');problem=json.loads(problem_path.read_text());need(problem['polls']==[case['component']],'complete actual native vector differs from consumer')
full=case['mixture'];y=case['y'];direction=c.pair_direction(case['component']);offset=np.zeros(6);compressed=c.AffineScalarMixture(full.covariances,full.weights,y,direction,offset)
# Include distant logit boundaries and each component's likelihood center.
means=np.unique(np.r_[[-20.,-8.,-3.,-1.,-.5,0.,.5,1.,3.,8.,20.],np.quantile(compressed.center,[0,.25,.5,.75,1])]);checks=[]
for mean in means:
 value,gradient,r=compressed.evaluate(mean);direct,g,R=full.evaluate(y,offset+direction*mean);scalar_gradient=float(g@direction)
 need(abs(value-direct)<1e-8 and abs(gradient-scalar_gradient)<1e-8,'actual component density/gradient identity failed');need(np.max(np.abs(r-R))<1e-9,'actual shared-component responsibilities changed')
 checks.append(dict(mean=float(mean),density_error=abs(value-direct),gradient_error=abs(gradient-scalar_gradient),responsibility_error=float(np.max(np.abs(r-R)))))
# Symbolic compilation is only one scalar factor, never the actual count graph.
import pytensor
import pytensor.tensor as pt
mu=pt.dscalar('invented_supplied_logit_mean');expression=compressed.symbolic_logpdf(mu);f=pytensor.function([mu],[expression,pt.grad(expression,mu)],mode='FAST_COMPILE');symbolic=[]
for mean in [-8.,-.5,0.,.5,8.]:
 value,gradient=f(mean);direct,g=compressed.evaluate(mean)[:2];h=1e-5;fd=(compressed.evaluate(mean+h)[0]-compressed.evaluate(mean-h)[0])/(2*h)
 need(abs(value-direct)<1e-8 and abs(gradient-g)<1e-8 and abs(fd-g)<1e-5,'scalar symbolic/finite difference mismatch');symbolic.append(dict(mean=mean,value_error=abs(float(value)-direct),gradient_error=abs(float(gradient)-g),finite_difference_error=abs(fd-g)))
# Independent SciPy full multivariate density oracle for all4,000 components.
from scipy.stats import multivariate_normal
scipy_value=float(logsumexp([np.log(w)+multivariate_normal.logpdf(y,mean=offset,cov=C)for w,C in zip(full.weights,full.covariances)]));need(abs(scipy_value-compressed.evaluate(0.)[0])<1e-9,'SciPy full multivariate oracle mismatch')
need(np.array_equal(full.weights,case['bank'].weights),'original weights changed')
# Negative shortcut: discard residual/profile evidence from component offsets.
with np.errstate(divide='ignore'):naive_base=np.log(full.weights)+.5*np.log(compressed.precision/(2*np.pi))
naive=[float(logsumexp(naive_base-.5*compressed.precision*(v-compressed.center)**2))for v in [-1.,0.,1.]];proper=[compressed.evaluate(v)[0]for v in [-1.,0.,1.]];contrast_error=(naive[2]-naive[1])-(proper[2]-proper[1]);need(abs(contrast_error)>1e-5,'shortcut counterexample unexpectedly absent')
negative=[]
def reject(name,fn):
 try:fn()
 except (ValueError,np.linalg.LinAlgError)as e:negative.append(dict(case=name,error=str(e)));return
 raise AssertionError('accepted '+name)
for kind in ['different_contest','third_candidate','duplicate_id']:
 block=copy.deepcopy(case['component'])
 if kind=='different_contest':block['rows'][1]['contest_id']='invented_other_contest'
 elif kind=='third_candidate':block['rows'][1]['second_candidate_id']='invented_third'
 else:block['rows'][1]['observation_id']=block['rows'][0]['observation_id']
 reject(kind,lambda block=block:c.pair_direction(block))
reject('infinite_mean',lambda:compressed.evaluate(float('inf')));reject('floating_overflow',lambda:compressed.evaluate(1e308));reject('zero_direction',lambda:c.AffineScalarMixture(full.covariances[:1],[1.],y,np.zeros(6),offset));reject('wrong_weights',lambda:c.AffineScalarMixture(full.covariances[:1],[.5],y,direction,offset))
# Independent invented nonzero offsets, reversed directions and zero component weight.
rng=np.random.default_rng(20261026);M=rng.normal(size=(3,4,4));cov=np.einsum('sij,skj->sik',M,M)+np.eye(4)[None]*.2;w=np.array([.2,.8,0.]);obs=np.array([.4,-.2,.1,-.7]);t=np.array([1.,-1.,1.,-1.]);d=np.array([.1,.2,-.1,.05]);toy=c.AffineScalarMixture(cov,w,obs,t,d);original=prototype.SharedVectorMixture(cov,w);toy_checks=[]
for mean in [-4.,0.,.2,4.]:
 value,gradient,r=toy.evaluate(mean);direct,g,R=original.evaluate(obs,d+t*mean);need(abs(value-direct)<1e-10 and abs(gradient-g@t)<1e-10,'offset/reversal/zero-weight test failed');toy_checks.append(dict(mean=mean,density_error=abs(value-direct),gradient_error=abs(float(gradient-g@t))))
np.savez(HERE/'coefficients.npz',precision=compressed.precision,center=compressed.center,base=compressed.base)
paths=[Path(__file__),HERE/'compression.py',Path(prototype.__file__),selection,problem_path,LAB/'calibration-consumer-v1/input-manifest.json',LAB/'calibration-consumer-v1/consumer.py',LAB/'calibration-consumer-v1/inputs/posterior-bank.npz',LAB/'calibration-consumer-v1/inputs/independent-review.json',LAB/'calibration-consumer-v1/inputs/poll-component.json',LAB/'calibration-consumer-v1/inputs/target-coordinates.json',HERE/'coefficients.npz']
result=dict(schema='M01ExactAffineScalarCompressionV1',status='actual_bank_supplied_state_identity_passed',seconds=time.monotonic()-start,component_count=4000,observation_count=6,scipy_full_multivariate_oracle=dict(mean=0.,value=scipy_value,error=abs(scipy_value-compressed.evaluate(0.)[0])),original_weight_bytes_sha256=hashlib.sha256(full.weights.tobytes()).hexdigest(),original_weight_sum=float(full.weights.sum()),direction=direction.tolist(),offset=offset.tolist(),actual_mean_checks=checks,symbolic_scalar_checks=symbolic,wrong_original_weight_1D_normal_shortcut=dict(log_likelihood_contrast_error=contrast_error,meaning='same original mixing weights on normalized1D Gaussian components changes mean-dependent likelihood; component-specific offsets cannot be dropped'),invented_offset_direction_zero_weight_checks=toy_checks,negative_checks=negative,constant_payload_bytes=dict(original_symbolic_precision_plus_logoffset=full.precisions.nbytes+full.log_constants.nbytes,compressed_three_arrays=compressed.precision.nbytes+compressed.center.nbytes+compressed.base.nbytes),pins={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)for p in paths},scope='Exact likelihood implementation identity at supplied scalar means using actual frozen bank. No full native graph build/compile, fit, predictive benefit, source admission or parent normalization. Fixed affine offsets retained; no original mixing weights modified.')
(HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],seconds=result['seconds'],means=len(means),maximum_density_error=max(r['density_error']for r in checks),maximum_gradient_error=max(r['gradient_error']for r in checks),negative_checks=len(negative),shortcut_contrast_error=contrast_error)))
