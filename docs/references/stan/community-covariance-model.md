---
source: project design — 2026-03-18
status: implemented; Stan model at src/covariance/stan/community_covariance.stan
---

# Community Covariance Stan Model

## Problem it solves

With K=7 community types and T=3 elections, estimating a full K×K covariance
matrix (28 free parameters) from T data points per community is underdetermined.
A low-rank factor model reduces this to F*(K + T) + K parameters, regularized
via priors.

---

## Model structure

```
theta[k,t] = mu[k] + lambda[k,:] · eta[:,t] + noise[k,t]

Sigma = lambda * lambda' + diag(tau^2)
```

- **mu[k]** — community k's baseline Democrat vote share
- **lambda[k,f]** — community k's loading on latent factor f
- **eta[f,t]** — factor f's score for election t
- **tau[k]** — idiosyncratic noise per community
- **Sigma** — the K×K covariance matrix used by Stage 4 poll propagation

### Why F=2?

From the empirical data:
- Factor 1 ≈ "overall partisan level" (all communities move together)
- Factor 2 ≈ "realignment direction" (c6 Hispanic trending R, c4 Asian trending D
  are negatively correlated — detected empirically in the 2016→2020 swing data)

F=3 with T=3 elections is technically over-parameterized; F=2 is the natural fit.
The runner accepts `--factors` argument to test F=1 or F=3.

---

## Identification

Factor loadings λ are NOT rotation-identified (any orthogonal rotation gives
equivalent likelihood). The covariance matrix **Sigma = λλ' + diag(τ²) IS
identified** — this is the quantity that matters for Stage 4.

Sign constraint applied: `lambda[1,f] >= 0` (first community's loading on each
factor must be non-negative). This prevents sign-switching across chains without
restricting the covariance structure. Full lower-triangular constraints are not
needed because we only care about Sigma, not the individual lambda values.

---

## Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| `mu[k]` | N(0.5, 0.15), [0.05, 0.95] | Community baselines centered around 50% |
| `lambda_raw[k,f]` | N(0, 0.08) | Small election-cycle swings (voter stability, A001) |
| `eta[f,t]` | N(0, 1) | Standardized factor scores |
| `tau[k]` | HalfNormal(0, 0.05) | Small idiosyncratic community variance |
| `sigma_floor` | HalfNormal(0, 0.02) | Minimal noise floor |

The `lambda` prior encodes the voter stability assumption (A001): communities
don't swing wildly from election to election. A 1-sigma election effect
(eta=1) moves a community ~8pp via lambda; a 2-sigma event (~8% probability)
moves it ~16pp. This is consistent with the largest observed community-level
swings in the empirical data.

---

## Observation uncertainty (theta_se)

The direct weighted mean has uncertainty determined by the effective sample size:

```python
n_eff = (sum(v_i * w_ik))^2 / sum((v_i * w_ik)^2)  # Kish effective N
theta_se_k = sqrt(theta_k * (1 - theta_k) / n_eff)
```

Communities with many high-membership tracts (c2 Black urban, c7 generic) have
large n_eff and small se. Communities with few dominant tracts (c4 Asian) have
small n_eff and larger se — correctly widening the likelihood for sparse
components.

---

## Generated quantities

The model generates three key outputs:

1. **Sigma[K,K]** — posterior covariance matrix. Primary output for Stage 4.
2. **Rho[K,K]** — posterior correlation matrix. Interpretable version of Sigma.
3. **r2_factor[K]** — proportion of each community's variance explained by the
   factor structure (vs. idiosyncratic component). High r2 = community behavior
   well-explained by shared factors. Low r2 = community has unique swing pattern.

---

## cmdstanpy interface

```python
import cmdstanpy

model = cmdstanpy.CmdStanModel(stan_file="src/covariance/stan/community_covariance.stan")
fit = model.sample(
    data=stan_data,    # dict with K, T, F, theta_obs, theta_se, obs_mask
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    seed=42,
)

# Check convergence
print(fit.diagnose())

# Extract posterior means
summary = fit.summary(percentiles=[5, 50, 95])
sigma_draws = fit.draws_pd()[["Sigma[1,1]", "Sigma[1,2]", ...]]

# Check R-hat (should be < 1.01 for all parameters)
# Check effective sample size (N_Eff > 400 per chain per parameter)
```

---

## Gotchas

**1. Missing data masking (AL 2018)**
AL 2018 governor race was uncontested. The `obs_mask` matrix encodes which
cells to include in the likelihood. Stan loops over mask > 0.5. Communities
still get baselines estimated from FL+GA; AL cells contribute no likelihood.

**2. Sign constraint is not a full identification**
The sign constraint prevents sign-flipping but not arbitrary rotation. If
you want to interpret individual lambda values (not just Sigma), you need a
stricter constraint (e.g., lower triangular structure). For Stage 4 we only
need Sigma, so this is fine.

**3. With T=3, the posterior is wide**
Three elections give very limited information about K=K covariance. The
posterior Sigma will have wide credible intervals. This is expected — the
priors do most of the regularization. Adding elections (once post-2026 data
exists) will tighten the posterior substantially.

**4. Compilation takes ~60 seconds the first time**
CmdStan compiles the Stan model to C++ on first run. Subsequent runs are
instant (cached binary). Run `model.compile(force=True)` if you modify
the .stan file.
