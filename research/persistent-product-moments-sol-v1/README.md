# Persistent product prior moments: analytical sensitivity v1

Lab-only standard-library calculation. No fit, posterior loading, model import, source mutation or adoption recommendation. The fixed k=32.2575 is an illustrative value supplied by the coordinating owner, not an independently verified posterior estimate. Exact read-only owner sources and lab outputs are hashed in manifest.json.

Let B=exp(Z), Z~Normal(-s²/2,s²), and independent U_t~Gamma(k,k), with U_t independent across distinct events conditional on fixed k. Then E[B]=E[U_t]=1; E[B²]=exp(s²) and E[U_t²]=1+1/k. Independence gives

    E[G_t] = 1, G_t = B U_t
    Var(G_t) = exp(s²)(1+1/k)-1
    Cov(G_t,G_r) = exp(s²)-1, t != r
    Corr(G_t,G_r) = (exp(s²)-1) / Var(G_t).

To match the incumbent variance v0=1/k0 using a possibly different renewed-event shape kU requires

    1/kU = (v0 - (exp(s²)-1))/exp(s²).

A finite positive gamma shape exists exactly when v0 > exp(s²)-1, equivalently s < sqrt(log(1+1/k0)) for nonnegative s. Equality requires deterministic U=1 (infinite shape); below that variance floor matching is impossible with independent mean-one U of any nonnegative variance, not merely Gamma. A feasible second-moment match does not match full marginal distributions or establish predictive skill.

For illustrative k0=32.2575, the allowable SD is about .1747; s=.35 alone contributes variance about .1303 versus incumbent .0310. Retaining the same k raises product variance to about .1654 and creates positive event covariance. Therefore a suggestion to variance-match the .35 independent-B challenger to this fixed incumbent is infeasible. A smaller s can admit variance matching, with a separately changed kU. These are analytical alternatives, not prescribed models.

The formulas describe prior factors conditional on fixed k and stated independence. With random k independent of B, marginal variance substitutes E[1/k] and cross-event covariance remains Var(B) because conditional means equal one. If B and k are dependent, product second moment becomes E[B²(1+1/k)]; posterior dependence among B, U, propensity, report errors and observed-coordinate transforms means these prior formulas cannot be applied as posterior factor decompositions. Joint parameter fitting, data constraints and count/report likelihoods may dominate actual predictive differences. Physical participant intensities multiply additional uncertain correlated factors, so these factor moments are not count moments.

Run `python3 analyze.py` here. Eleven deterministic checks cover zero-s reduction, deterministic U, exact matching boundary, impossible and feasible examples, and an independent exact finite-support enumeration of the general product/covariance identities. No stochastic simulation is used. Floating-point boundary classification is exact for the supplied same-expression fixture, not a numerical tolerance rule for estimating parameters.
