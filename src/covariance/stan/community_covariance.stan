/**
 * Community-type political covariance model (F=1 factor)
 *
 * Stage 3 Bayesian estimation: fits community-type Democratic vote shares
 * across T elections using a rank-1 factor model.
 *
 * Model structure:
 *   theta[k,t] = mu[k] + lambda[k] * eta[t] + noise[k,t]
 *
 *   where:
 *     mu[k]      = community k's baseline Democrat vote share
 *     lambda[k]  = community k's sensitivity to the common "national wave" factor
 *     eta[t]     = election t's national wave score
 *                  (positive = D-favorable environment)
 *
 * Induced covariance between communities k and j:
 *   Cov(theta[k,:], theta[j,:]) = lambda[k] * lambda[j] + delta_{kj} * tau[k]^2
 *   Sigma[k,j] = lambda[k] * lambda[j]   (if k != j)
 *   Sigma[k,k] = lambda[k]^2 + tau[k]^2
 *
 * Identification:
 *   F=1 has one sign ambiguity: (lambda, eta) = (-lambda, -eta) gives the
 *   same likelihood. Fixed by constraining lambda[k_ref] > 0, where k_ref=2
 *   (c2 Black urban — the most consistently Democratic community). This is
 *   implemented as a hard lower bound on the parameter, NOT a runtime flip,
 *   which avoids posterior discontinuities that break NUTS.
 *
 * Why F=1 for MVP:
 *   With T=3 elections, a factor model with F=2 has: 7 mu + 14 lambda +
 *   6 eta + 7 tau + 1 sigma = 35 parameters from 21 data points. The
 *   posterior is wide and chains mix poorly. F=1 reduces to 7+7+3+7+1=25
 *   parameters, still prior-dominated but converges cleanly. The empirical
 *   c6/c4 anti-correlation (Hispanic realignment vs Asian D-shift) is
 *   captured in their lambda signs relative to each other; F=2 is needed
 *   only to decompose this into separate factors.
 *
 * Data notes:
 *   theta_obs[k,t] = direct vote-weighted mean Democratic share
 *   theta_se[k,t]  = Kish effective-n standard error per community-election
 *   obs_mask[k,t]  = 1 if observed, 0 if missing (AL 2018 uncontested = 1 for all k,
 *                    since FL+GA communities are still observed; AL was excluded
 *                    at the tract level before computing community shares)
 */

data {
  int<lower=1> K;          // number of community types (7)
  int<lower=1> T;          // number of elections (3)

  matrix[K, T] theta_obs;  // observed community vote shares
  matrix[K, T] theta_se;   // observation standard errors per cell
  matrix[K, T] obs_mask;   // 1 = include in likelihood
}

parameters {
  // Community baseline vote shares
  vector<lower=0.05, upper=0.95>[K] mu;

  // Factor loadings: community sensitivity to national wave
  // k_ref = 2 (c2 Black urban) is constrained positive for sign identification.
  // Hard parameter bound avoids posterior discontinuity.
  real<lower=0>    lambda_ref;       // lambda for c2 (k=2), always >= 0
  vector[K-1]      lambda_other;     // lambda for c1, c3..c7 (unconstrained)

  // Election-level factor scores (national wave intensity)
  // Positive = D-favorable year, negative = R-favorable year
  vector[T] eta;

  // Idiosyncratic community variance (not explained by national wave)
  vector<lower=0>[K] tau;

  // Shared observation noise floor
  real<lower=0> sigma_obs;
}

transformed parameters {
  // Assemble full lambda vector: insert lambda_ref at position k=2
  vector[K] lambda;
  lambda[1] = lambda_other[1];   // c1: White rural homeowner
  lambda[2] = lambda_ref;        // c2: Black urban (sign-constrained reference)
  for (k in 3:K)
    lambda[k] = lambda_other[k - 1];  // c3..c7

  // Predicted community vote shares
  matrix[K, T] theta_pred;
  for (k in 1:K)
    for (t in 1:T)
      theta_pred[k, t] = mu[k] + lambda[k] * eta[t];
}

model {
  // ── Priors ────────────────────────────────────────────────────────────────

  // Community baselines: centered at 50% with wide spread to accommodate
  // strongly partisan communities (c2 at ~71%, c7 at ~40%)
  mu ~ normal(0.50, 0.18);

  // Factor loadings: communities shift modestly with national wave.
  // Prior: |lambda| ~ 0.05 (half-normal scale = 0.05 → 95th pctile at ~0.10)
  // Based on observed swings: no community moved more than ~4pp between
  // 2016 and 2020 presidential, consistent with a small common factor.
  lambda_ref ~ normal(0.03, 0.04);      // positive-constrained, small scale
  lambda_other ~ normal(0, 0.05);       // unconstrained but small

  // Election scores: standardized; scale absorbed by lambda
  eta ~ normal(0, 1);

  // Idiosyncratic noise: small — communities are fairly stable across elections
  tau ~ normal(0, 0.03);

  // Observation noise floor: tight — direct weighted means are precise aggregates
  sigma_obs ~ normal(0, 0.01);

  // ── Likelihood ─────────────────────────────────────────────────────────────

  for (k in 1:K) {
    for (t in 1:T) {
      if (obs_mask[k, t] > 0.5) {
        real se_kt = sqrt(square(theta_se[k, t]) + square(tau[k]) + square(sigma_obs));
        theta_obs[k, t] ~ normal(theta_pred[k, t], se_kt);
      }
    }
  }
}

generated quantities {
  // ── Covariance matrix (primary output → Stage 4 poll propagation) ─────────
  matrix[K, K] Sigma;
  for (k in 1:K) {
    for (j in 1:K) {
      Sigma[k, j] = lambda[k] * lambda[j];
      if (k == j) Sigma[k, j] += square(tau[k]);
    }
  }

  // ── Correlation matrix (interpretable version) ────────────────────────────
  matrix[K, K] Rho;
  {
    vector[K] sd_vec;
    for (k in 1:K)
      sd_vec[k] = sqrt(Sigma[k, k]);
    for (k in 1:K)
      for (j in 1:K)
        Rho[k, j] = Sigma[k, j] / (sd_vec[k] * sd_vec[j]);
  }

  // ── Posterior predictive (for model checking) ─────────────────────────────
  matrix[K, T] theta_rep;
  for (k in 1:K) {
    for (t in 1:T) {
      real se_kt = sqrt(square(theta_se[k, t]) + square(tau[k]) + square(sigma_obs));
      theta_rep[k, t] = normal_rng(theta_pred[k, t], se_kt);
    }
  }

  // ── Factor R² per community ───────────────────────────────────────────────
  vector[K] r2_factor;
  for (k in 1:K) {
    real var_factor = square(lambda[k]);
    real var_total = var_factor + square(tau[k]);
    r2_factor[k] = var_factor / var_total;
  }
}
