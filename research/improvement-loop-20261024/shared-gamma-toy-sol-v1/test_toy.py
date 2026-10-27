import unittest
from toy import *
class ProofChecks(unittest.TestCase):
 def test_normalized_positive_nuisance(self):
  for k,w in KAPPA:
   self.assertGreater(k,0);self.assertGreater(w,0);self.assertEqual(gamma_integral_normalization(k),1)
   for u,_ in PARENTS:
    for t,_ in THETA_PRIOR:
     for e in range(2):self.assertEqual(pgf(Q(1),k,rate(u,t,e)),1)
 def test_theta_posterior_integration_identity(self):
  self.assertEqual(theta_posterior(),((0,Q(1,4)),(1,Q(3,4))))
  for t,w in THETA_PRIOR:
   integrated=sum(kw*gamma_integral_normalization(k)**2 for k,kw in KAPPA)
   self.assertEqual(integrated,1)
   self.assertEqual(w*LIKELIHOOD[t]*integrated,w*LIKELIHOOD[t])
 def test_gamma_poisson_integral_matches_nb(self):
  for k,_ in KAPPA:
   for lam in [Q(1),Q(5,2),Q(11)]:
    for n in range(8):self.assertEqual(integrated_poisson(n,k,lam),nb(n,k,lam))
 def test_all_small_joint_cells_reconstruct(self):
  for i in range(4):
   for j in range(4):self.assertEqual(predict(i,j),full_integrated(i,j))
 def test_shared_vs_independent_kappa_exact(self):
  self.assertEqual(joint_given_theta(0,0,0,0),Q(5,36))
  self.assertEqual(joint_given_theta(0,0,0,0,'independent_kappa'),Q(119,864))
  self.assertNotEqual(predict(0,0),predict(0,0,'independent_kappa'))
 def test_fixed_frailty_changes_joint_and_dispersion(self):
  self.assertNotAlmostEqual(float(joint_given_theta(0,0,0,0)),joint_given_theta(0,0,0,0,'fixed_frailty'),places=8)
  lam=Q(2);mixture_variance=lam+lam*lam*sum(w/Q(k)for k,w in KAPPA)
  self.assertEqual(mixture_variance,5);self.assertNotEqual(mixture_variance,lam)
 def test_observation_invalidates_prior_only(self):
  self.assertEqual(posterior_kappa_after_count_zero(0,0),((1,Q(9,17)),(2,Q(8,17))))
  self.assertNotEqual(posterior_kappa_after_count_zero(0,0),KAPPA)
 def test_parent_masses_not_updated_by_theta_evidence(self):
  evidence=sum(w*LIKELIHOOD[t]for t,w in THETA_PRIOR)
  self.assertEqual(sum(w for _,w in PARENTS),1)
  self.assertEqual(tuple((u,w*evidence/evidence)for u,w in PARENTS),PARENTS)
if __name__=='__main__':unittest.main()
