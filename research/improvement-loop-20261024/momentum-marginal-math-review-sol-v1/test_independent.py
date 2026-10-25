"""Independent supplied-only numerical checks. No chains, labels or data files."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import IntegrationWarning, quad
from scipy.special import expit, log_ndtr, logsumexp, ndtri
from scipy.stats import norm

SOURCE=Path('/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/momentum-marginal-math-root-v1/marginal.py')
spec=importlib.util.spec_from_file_location('reviewed_marginal_v1',SOURCE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


class IndependentChecks(unittest.TestCase):
    def setUp(self):
        self.means=np.array([-1.7,.2,1.3])
        self.scales=np.array([.4,.8,1.1])
        self.weights=np.array([.2,.3,.5])
        self.model=m.GaussianLogitMixture(self.means,self.scales,self.weights)

    def test_cdf_survival_density_direct_normal_oracle(self):
        for x in [.001,.1,.3,.5,.8,.999]:
            z=(np.log(x/(1-x))-self.means)/self.scales
            self.assertAlmostEqual(self.model.cdf(x),self.weights@norm.cdf(z),places=14)
            self.assertAlmostEqual(self.model.sf(x),self.weights@norm.sf(z),places=14)
            self.assertAlmostEqual(self.model.density(x),self.weights@(norm.pdf(z)/(self.scales*x*(1-x))),places=12)
            self.assertAlmostEqual(self.model.cdf(x)+self.model.sf(x),1,places=14)
        for endpoint in [0,1]:
            self.assertEqual(self.model.density(endpoint),0)
            self.assertEqual(self.model.cdf(endpoint),endpoint)
            self.assertEqual(self.model.sf(endpoint),1-endpoint)

    def test_quantile_single_and_mixture_brackets(self):
        single=m.GaussianLogitMixture([-.6],[1.4])
        for q in [.025,.1,.25,.5,.75,.9,.975]:
            value,receipt=single.quantile(q)
            self.assertAlmostEqual(value,expit(-.6+1.4*ndtri(q)),delta=5.1e-11)
            value,receipt=self.model.quantile(q)
            self.assertLessEqual(receipt['width'],1e-10)
            zlo=(np.log(receipt['lower']/(1-receipt['lower']))-self.means)/self.scales
            zhi=(np.log(receipt['upper']/(1-receipt['upper']))-self.means)/self.scales
            self.assertLessEqual(self.weights@norm.cdf(zlo),q+1e-15)
            self.assertGreaterEqual(self.weights@norm.cdf(zhi),q-1e-15)

    def test_mean_against_normal_domain_and_component_weight_identity(self):
        expected=np.array([quad(lambda z:expit(mu+sd*z)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-12,12,epsabs=1e-12)[0] for mu,sd in zip(self.means,self.scales)])
        actual,_=self.model.component_mean_share()
        np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=0)
        self.assertAlmostEqual(self.model.mean_share()[0],self.weights@expected,delta=2e-9)
        self.assertGreater(abs(expected[-1]-expit(self.means[-1])),.01)

    def test_crps_against_fixed_legendre_quadrature_and_endpoints(self):
        nodes,weights=leggauss(320)
        for y in [0,.23,.68,1]:
            total=0
            for lo,hi,use_survival in [(0,y,False),(y,1,True)]:
                if hi==lo:continue
                x=lo+(nodes+1)*(hi-lo)/2
                z=(np.log(x/(1-x))[:,None]-self.means)/self.scales
                probability=norm.sf(z)@self.weights if use_survival else norm.cdf(z)@self.weights
                total+=(hi-lo)/2*(weights@probability**2)
            actual,receipt=self.model.crps(y)
            self.assertAlmostEqual(actual,total,delta=2e-9)
            self.assertLessEqual(receipt['estimated_error'],1e-8)

    def test_crps_centered_influence_and_directional_finite_difference(self):
        direction=np.array([.1,-.3,.2]);epsilon=1e-4
        for y in [0,.23,.68,1]:
            influence,_=self.model.crps_influence(y)
            self.assertAlmostEqual(self.weights@influence,0,delta=2e-10)
            up=m.GaussianLogitMixture(self.means,self.scales,self.weights+epsilon*direction).crps(y)[0]
            down=m.GaussianLogitMixture(self.means,self.scales,self.weights-epsilon*direction).crps(y)[0]
            self.assertAlmostEqual((up-down)/(2*epsilon),direction@influence,delta=2e-8)
        # y=0 isolates the upper-survival integral and exposes a sign reversal.
        influence,_=self.model.crps_influence(0)
        nodes,weights=leggauss(320);x=(nodes+1)/2
        components=norm.sf((np.log(x/(1-x))[:,None]-self.means)/self.scales)
        sf=components@self.weights
        independent=(weights[:,None]*sf[:,None]*(components-sf[:,None])).sum(axis=0)
        np.testing.assert_allclose(influence,independent,atol=2e-9,rtol=0)

    def test_quantile_and_wis_weight_influences(self):
        direction=np.array([.1,-.3,.2]);epsilon=1e-4
        plus=m.GaussianLogitMixture(self.means,self.scales,self.weights+epsilon*direction)
        minus=m.GaussianLogitMixture(self.means,self.scales,self.weights-epsilon*direction)
        levels=[.025,.1,.25,.5,.75,.9,.975]
        baseline={q:self.model.quantile(q)[0] for q in levels}
        up={q:plus.quantile(q)[0] for q in levels}
        down={q:minus.quantile(q)[0] for q in levels}
        dq={}
        for q in levels:
            x=baseline[q]
            component_cdf=norm.cdf((np.log(x/(1-x))-self.means)/self.scales)
            dq[q]=-(direction@component_cdf)/self.model.density(x)
            self.assertAlmostEqual(dq[q],(up[q]-down[q])/(2*epsilon),delta=2e-6)
        for y in [0,.39,1]:
            expected=sum((int(y<baseline[q])-q)*dq[q] for q in levels)/3.5
            actual=(m.weighted_interval_score(y,up)-m.weighted_interval_score(y,down))/(2*epsilon)
            self.assertAlmostEqual(actual,expected,delta=2e-6)
            pinball=sum((y-baseline[q])*(q-int(y<baseline[q])) for q in levels)/3.5
            self.assertAlmostEqual(m.weighted_interval_score(y,baseline),pinball,places=14)

    def test_extreme_normal_tail_and_winner_score_oracles(self):
        model=m.GaussianLogitMixture([50],[1])
        win,lose=model.log_win_lose()
        # Mills expansion, independent of log_ndtr: log Phi(-z).
        z=50.;series=1-1/z**2+3/z**4-15/z**6+105/z**8
        oracle=-z*z/2-np.log(z)-.5*np.log(2*np.pi)+np.log(series)
        self.assertAlmostEqual(lose,oracle,delta=2e-11)
        self.assertEqual(m.winner_scores(win,lose,False)['negative_log_score'],-lose)
        for p in [.01,.3,.5,.9]:
            for winner in [False,True]:
                scores=m.winner_scores(np.log(p),np.log1p(-p),winner)
                self.assertAlmostEqual(scores['brier'],(p-int(winner))**2,places=15)
                self.assertAlmostEqual(scores['negative_log_score'],-np.log(p if winner else 1-p),places=15)

    def test_scalar_bounded_retry_success_and_two_attempt_failure(self):
        ledger=[]
        def warned(*args,**kwargs):
            ledger.append(kwargs)
            if len(ledger)==1:warnings.warn('forced first-attempt warning',IntegrationWarning)
            return .5,1e-10
        with patch.object(m,'quad',side_effect=warned):
            value,receipt=m.integral(lambda x:x,0,1)
        self.assertEqual(value,.5);self.assertEqual(receipt['attempt'],1)
        self.assertEqual([x['limit'] for x in ledger],[200,400])
        self.assertEqual([x['epsabs'] for x in ledger],[1e-9,1e-10])
        with patch.object(m,'quad',return_value=(.5,1e-2)) as mock:
            with self.assertRaises(ArithmeticError):m.integral(lambda x:x,0,1)
        self.assertEqual(mock.call_count,2)
        with patch.object(m,'quad',side_effect=[(.5,1e-2),(.6,1e-12)]) as mock:
            with self.assertRaises(ArithmeticError):m.integral(lambda x:x,0,1)
        self.assertEqual(mock.call_count,2)

    def test_vector_bounded_retry_and_nonfinite_failure(self):
        class Info:
            def __init__(self,success):self.success=success
        with patch.object(m,'quad_vec',side_effect=[(np.array([.1,.2]),1e-10,Info(False)),(np.array([.1,.2]),1e-11,Info(True))]) as mock:
            _,receipt=m.integral(lambda x:np.array([x,x]),0,1,vector=True)
        self.assertEqual(receipt['attempt'],1);self.assertEqual(mock.call_count,2)
        for call in mock.call_args_list:
            self.assertEqual(call.kwargs['workers'],1);self.assertEqual(call.kwargs['norm'],'max')
        with patch.object(m,'quad',return_value=(np.nan,1e-12)) as mock:
            with self.assertRaises(ArithmeticError):m.integral(lambda x:x,0,1)
        self.assertEqual(mock.call_count,2)

    def test_input_and_copied_array_boundaries(self):
        means=np.array([0.,1.]);scales=np.array([.5,.7]);weights=np.array([.4,.6])
        model=m.GaussianLogitMixture(means,scales,weights)
        means[:]=20;scales[:]=20;weights[:]=20
        np.testing.assert_array_equal(model.mean,[0,1]);np.testing.assert_array_equal(model.sd,[.5,.7])
        np.testing.assert_array_equal(model.weights,[.4,.6])
        for array in [model.mean,model.sd,model.weights]:
            with self.assertRaises(ValueError):array.setflags(write=True)
        for bad in [-.1,np.nan,1.1]:
            with self.assertRaises(ValueError):model.cdf(bad)
            with self.assertRaises(ValueError):model.quantile(bad)
        with self.assertRaises(ValueError):m.weighted_interval_score(.5,{.5:.5})

    def test_confirm_weight_underflow_defect_requires_correction(self):
        raw=np.array([1e-300,1e300]);mu=np.array([-100.,100.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            model=m.GaussianLogitMixture(mu,[1,1],raw)
            actual=model.log_win_lose()[1]
        logw=np.log(raw)-logsumexp(np.log(raw))
        expected=float(logsumexp(logw+log_ndtr(-mu)))
        self.assertEqual(model.weights[0],0)
        self.assertGreater(abs(actual-expected),3600)

    def test_confirm_own_roundoff_logtail_rejection_requires_correction(self):
        model=m.GaussianLogitMixture(np.full(4,50.),np.ones(4),[1,2,3,4])
        win,lose=model.log_win_lose()
        self.assertGreater(win,0)
        with self.assertRaises(ValueError):m.winner_scores(win,lose,True)

    def test_confirm_zero_width_nonfinite_bypass_requires_correction(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            value,receipt=m.integral(lambda x:np.inf,.5,.5)
        self.assertTrue(np.isnan(value));self.assertEqual(receipt['estimated_error'],0)

if __name__=='__main__':unittest.main(verbosity=2)
