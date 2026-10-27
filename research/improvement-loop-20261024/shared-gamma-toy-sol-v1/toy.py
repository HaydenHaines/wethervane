"""Invented exact finite-kappa Gamma/Poisson hierarchy, not owner model code."""
from fractions import Fraction as Q
from math import factorial,exp
import json

PARENTS=tuple((u,Q(1,8))for u in range(8))
KAPPA=((1,Q(1,2)),(2,Q(1,2)))
THETA_PRIOR=((0,Q(1,2)),(1,Q(1,2)))
# Observed Y=1 has likelihood 1/4 vs3/4, independent of all frailty variables.
LIKELIHOOD={0:Q(1,4),1:Q(3,4)}

def theta_posterior():
    z=sum(w*LIKELIHOOD[t]for t,w in THETA_PRIOR)
    return tuple((t,w*LIKELIHOOD[t]/z)for t,w in THETA_PRIOR)

def rate(u,t,event):return Q(1+u+2*t+event)

def gamma_integral_normalization(k):
    # Gamma(k,k) density coefficient times integral g^(k-1)e^(-kg)dg.
    return Q(k**k,factorial(k-1))*Q(factorial(k-1),k**k)

def integrated_poisson(n,k,lam):
    # Independently write the integral of Pois(n|lam*g)Gamma(g|k,k).
    return Q(k**k,factorial(k-1))*lam**n/Q(factorial(n))*Q(factorial(k+n-1))/(Q(k)+lam)**(k+n)

def nb(n,k,lam):
    return Q(factorial(k+n-1),factorial(k-1)*factorial(n))*(Q(k)/(k+lam))**k*(lam/(k+lam))**n

def pgf(z,k,lam):return (Q(k)/(k+lam*(1-z)))**k

def joint_given_theta(u,t,n1,n2,mode='shared'):
    l1,l2=rate(u,t,0),rate(u,t,1)
    if mode=='shared':return sum(w*nb(n1,k,l1)*nb(n2,k,l2)for k,w in KAPPA)
    if mode=='independent_kappa':return sum(w*nb(n1,k,l1)for k,w in KAPPA)*sum(w*nb(n2,k,l2)for k,w in KAPPA)
    if mode=='fixed_frailty':return exp(-float(l1+l2))*float(l1**n1*l2**n2)/factorial(n1)/factorial(n2)
    raise ValueError('unknown reconstruction')

def predict(n1,n2,mode='shared'):
    return sum(pw*tw*joint_given_theta(u,t,n1,n2,mode)for u,pw in PARENTS for t,tw in theta_posterior())

def full_integrated(n1,n2):
    return sum(pw*tw*kw*integrated_poisson(n1,k,rate(u,t,0))*integrated_poisson(n2,k,rate(u,t,1))for u,pw in PARENTS for t,tw in theta_posterior()for k,kw in KAPPA)

def posterior_kappa_after_count_zero(u,t):
    raw=[(k,w*nb(0,k,rate(u,t,0)))for k,w in KAPPA];z=sum(w for _,w in raw)
    return tuple((k,w/z)for k,w in raw)

def report():
    rows=[]
    for n1,n2 in [(0,0),(0,1),(1,0),(1,1),(2,2)]:
        a=predict(n1,n2);b=full_integrated(n1,n2)
        if a!=b:raise AssertionError('full integral/reconstruction differ')
        rows.append({'counts':[n1,n2],'fullIntegrated':str(b),'sharedReconstruction':str(a),'independentKappa':str(predict(n1,n2,'independent_kappa')),'fixedFrailty':predict(n1,n2,'fixed_frailty')})
    return {'scope':'invented discrete-kappa/continuous-Gamma exact mathematics; no actual model or empirical validation','parentMasses':[str(w)for _,w in PARENTS],'thetaPosterior':[(t,str(w))for t,w in theta_posterior()],'kappaPrior':[(k,str(w))for k,w in KAPPA],'jointPredictions':rows,'conditionalTheta0Parent0P00':{m:str(joint_given_theta(0,0,0,0,m))for m in ['shared','independent_kappa','fixed_frailty']},'countObservedCounterexampleKappaPosterior':[(k,str(w))for k,w in posterior_kappa_after_count_zero(0,0)]}
if __name__=='__main__':print(json.dumps(report(),indent=2))
