# Exact mixture-aware compression is a useful next implementation step

The frozen TX six-poll vector has one shared contest and unordered candidate pair. Its native poll means therefore lie on one scalar log-ratio axis. The existing 4,000-component full-vector likelihood can be evaluated exactly on that axis using three 4,000-element arrays, retaining **all full-multivariate normalization and residual evidence offsets**. This is a computational replacement of the same working likelihood, not a new scientific model, refit, forecast or predictive improvement.

There is an existing exact sufficient-statistic implementation: frozen `polling_v1.py:89–99` computes Gaussian precision/linear terms for the fixed-covariance aggregate sampler. It returns no full-density constant because that constant is irrelevant to its supported fixed-covariance state update. This packet supplies a centered, **mixture-aware normalizer-preserving restriction** for the native M01 factor; it does not claim that Gaussian compression was previously missing or that the existing fixed-covariance function is wrong. No existing fit or owner experiment was repeated.

## Identity and retained measure

For covariance component s, observed vector y, a **fixed** offset d and mean direction t, let C_s=L_s L_sᵀ, u_s=L_s⁻¹t, v_s=L_s⁻¹(y−d), a_s=u_sᵀu_s and m_s=u_sᵀv_s/a_s. Then

`||L_s⁻¹(y−d−t μ)||² = ||v_s−u_s m_s||² + a_s (μ−m_s)²`.

The retained base is

`base_s = log(w_s) − n/2 log(2π) − sum(log(diag(L_s))) − 0.5 ||v_s−u_s m_s||²`.

The original likelihood is exactly

`logsumexp_s(base_s − 0.5 a_s (μ−m_s)²)`.

Centering the whitened residual avoids subtracting two nearly equal large quadratic terms. The fixed offset is preserved; offsets depending on other uncertain model parameters cannot be frozen into these coefficients. On the actual fixture t is six ones and d is six zeros. The measure remains the original **six-dimensional poll-logit observation density**. This is not a normalized one-dimensional density over μ, and no new observation/parameter Jacobian is introduced.

A shortcut using normalized one-dimensional Gaussians with the original component weights changes the μ-dependent likelihood: on the actual bank its log-likelihood contrast from μ=0 to μ=1 differs by 0.0145732. This is not merely an irrelevant common additive constant. Residual evidence and component-specific normalizers affect shared-component responsibilities and must remain. The original joint five-scale row order and all original weights are retained exactly, including their floating-point sum 1.0000000000000004; there is no renormalization or independent per-scale mixing.

## Completed bounded checks

The authenticated `load_frozen_current_case` helper verifies source/snapshot/review/bank pins before loading the existing current-only bank. The native prepared-v4 Problem bytes match their frozen SHA, and its **entire** poll block equals the consumer fixture. No native graph/provider is imported or constructed. Coefficients are derived from all 4,000 covariance components, not a selected subset.

Across 16 supplied means spanning −20 to +20 and component-center quantiles:

- Full-vector versus compressed density maximum absolute error: 3.18e-12.
- Scalar derivative versus summed full-vector mean derivatives: 6.25e-13.
- Shared-component responsibility maximum error: 1.61e-15.

An independent SciPy full multivariate Gaussian/logsumexp oracle over all 4,000 components agrees exactly at μ=0 at recorded precision. A **scalar factor micrograph**, compiled with C disabled, matches numeric values/derivatives and central differences at five supplied means; maximum finite-difference error is 3.47e-9. This is not the actual 662-coordinate graph or its compilation/gradient.

Seven negatives reject multiple contests, third candidates, duplicate IDs, infinite/overflowing means, zero direction and invalid weight mass. Invented tests preserve nonzero fixed offsets, reversed mean directions and a zero-weight component. Reversing a source row still requires correctly oriented observed data/covariance; this helper does not authenticate or invent that source transformation.

The final run took 1.380 seconds under nice15, one backend thread, a 45-second wall timeout, hard CPU 30/35 seconds, 2 GiB address-space cap and 8 MiB per-file cap. No sampler, optimizer or source acquisition ran. `coefficients.npz` is an actually used derivation artifact, pinned by SHA and size in both result and evidence, with precision/center/base arrays. Its source recipe and all supplied-state checks are retained.

The direct symbolic constant-array payload falls from 1,184,000 bytes (6×6 precision bank plus component log-offset vector) to 96,000 bytes (three scalar-component vectors), about 12.3-fold. **This is only array accounting.** It is not a measured reduction in full-graph pickle size, node count, compiler memory, cold compilation time or sampler cost. CES and other graph paths may dominate. The prior 64 MiB partial pickle is untouched and its size problem is not declared solved.

## Actionable comparison path and limits

The next implementation candidate is a separately versioned measurement provider using this scalar restriction at the same named candidate count-rate port, while retaining incumbent CES/report paths, the complete six-token claim, original covariance ownership, source order and population-cut parent masses. First freeze/review that source and fresh resource envelope, then test full source-bound graph construction and saved-point/full-gradient parity under a separately bounded scope. Do not bypass the current original-parent/target envelope or silently generalize to multiple contests/blocks. That native successor has **not** been implemented or tested by this packet.

Only after owner coordination and appropriate source qualification should this help execute the substantive matched comparison: fixed-covariance incumbent versus the declared marginal calibration-mixture challenger, with identical information, complete-vector dependence and jointly fitted affected components. Do not duplicate active owner runs. Retain separate fundamentals/polls/combined references and the project A-level 2018/2022 and B-level 2020/2024 cycle reporting where supported. The TX2026 bank is not an earlier-information historical bank. Current supplied-state checks cannot establish predictive usefulness or rank the weakest of eight layers.

All original calibration/sample-overlap, publication-vintage, legal/support, finite-bank integration and outer-parent precision caveats remain. The formula preserves the current marginal-Bayes mixture law; it does not turn it into a protected calibration cut or normalize each downstream population conditional. No 2026 outcome or forecast is produced. Luna's gasoline score-concentration diagnostic is complementary and uses different evidence; no experiment was duplicated.
