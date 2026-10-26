# Shared outcome ownership is the immediate comparison-design issue

Independent read-only joins of the existing lab preparation and bank review
show that **all used historical calibration outcome keys are also training
keys in both the H0 and F0 fundamentals banks**. This is more specific than
the existing warning that some labels may be shared. It does not establish
that a probability density was counted twice.

| Target | Calibration rows | Used calibration event labels | Labels also in H0/F0 training | Existing bank training events |
|---|---:|---:|---:|---:|
| 2018 | 930 | 59 | 59 / 59 | 299 |
| 2020 | 1,218 | 85 | 85 / 85 | 329 |
| 2022 | 1,514 | 106 | 106 / 106 | 357 |
| 2024 | 1,514 | 106 | 106 / 106 | 387 |

The separate earlier-label tables have 62, 90, 118 and 118 rows; these are
not counts of labels actually used by selected polling residuals. The checks
join selected poll event keys, not the whole label table. All selected poll
cycles precede their target. None of the used calibration keys intersects its
bank's target keys. Earlier-only event selection still does not verify source
vintage or original publication availability.

Holding the existing fundamentals banks fixed and removing every calibration
event with a shared training label leaves **zero calibration rows in every
fold**. An event-disjoint alternative therefore requires a different
fundamentals training design, a different calibration source, or an explicitly
admitted conditional/joint law. Deleting a few pollsters cannot solve this
particular event-label ownership issue.

`check.py` independently reconstructs these intersections from the selected
CSV files and the bank compatibility report; it checks the F0 training lists
against a separate preflight support CSV. `event-ownership.csv` lists each
union event key, the number of calibration residuals using it, and F0 training
membership. `verification.json` binds input bytes. These bank lists are prior
review artifacts, not independent authentication of their upstream source
labels or a complete native Problem.

# Shared labels do not automatically imply a duplicated likelihood

The existing preparation explicitly calls its shared-label treatment a
conditional empirical-Bayes approximation. The operative distinction is
whether a label is a stochastic observation in another density or a fixed
conditioning value in a normalized conditional law. For an illustrative
model with independent priors,

`p(beta, theta, Y, P) = p(beta) p(theta) p(Y | beta) p(P | Y, theta)`.

Conditioning on observed historical `Y,P` gives
`p(beta | Y) p(theta | P,Y)`. That factorization can be exact for this declared
working law despite complete event-key overlap. Its poll conditional must
really be normalized, and the assumptions must hold. It does not apply
automatically to a richer count graph with shared uncertain states, dependent
priors, common errors, label measurement uncertainty, or feedback.

The executable invented two-by-two fixture verifies this equality to floating
precision. It also shows the different consequence of squaring the outcome
likelihood: a once-conditioned beta probability of 0.8 becomes
0.9411764705882353. These invented values are an illustration, not an estimated
project bias or an empirical result. Existing normalized Gaussian residual
code uses outcomes to form logit residual means; reading the same value as a
covariate does not, by itself, establish a second marginal outcome density.

Consequently the current once-only ledger's conservative rejection of labels
under two independent numerical factor owners should be resolved by an
explicit joint owner/conditional adapter, rather than declaring all overlaps
independent or claiming this audit has discovered an actual doubled factor.
Nothing here changes that validator or admits a real ledger.

# Temporal variance has a separate ownership ambiguity

The pinned covariance implementation already includes
`sigma_movement^2 * 1[same event] * min(days_i,days_j)/30` and explicitly warns
against adding another sampling/MOE term to its total-error component. A new
latent movement process with exactly that same covariance cannot be separately
identified from residual movement using only aggregate poll residuals: only
the sum of their nonnegative variance coefficients is identified. For a fixed
total variance `v`, every split `(a*v, (1-a)*v)`, `0 <= a <= 1`, has the same
marginal covariance. The executable kernel-budget fixture verifies five such
splits on the actual six-poll coordinate dates. This is a conditional algebraic
result, not evidence that the existing prototype actually adds two processes.

The current prototype targets final expected pair shares and integrates its
existing total-error covariance. The prepared momentum challenger adds a
distinct random-slope kernel while retaining Brownian movement. Distinct
kernels are not automatically duplicated variance; the split argument applies
only if another model uses the same kernel and port. A full native compiler
trace must establish whether such an additional process actually exists.

All selected historical rows have unknown source sample identity. Existing
field-end dates, different sample IDs, latest-per-bin selection and the rare
advertised field-interval intersections do not identify reused people. This
audit supplies no nontrivial respondent-overlap bound or causal separation of
true preference movement from measurement error.

# Executable next step

Run the supplied check with standard-library Python; no scientific environment
or fitting is needed:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 research/improvement-loop-20261024/m01-ownership-followup-sol-v1/check.py
```

For the next native candidate, bind the actual selected Problem and its
complete vector. For each canonical outcome in `event-ownership.csv`, record
the sole marginal outcome density owner and every conditional use. Supply a
normalized conditional adapter for the calibrated residual law and identify
the shared latent/error variables it assumes away or models. Include a
fixed-state test that the joint factor equals its declared conditional
decomposition and fails when the marginal label density is duplicated. Bind
the total temporal covariance budget at the same port and reject addition of
an identical movement kernel without an explicit allocation. This can proceed
as supplied-state integration work before any refit. If that conditional law
is not defensible, preregister an event-partitioned training design for both
arms; existing banks cannot deliver label disjointness by calibration-only
filtering.

This package creates no bank, sampler, optimizer, score, native registration,
publication, historical availability claim, predictive benefit finding, or
weakest-layer ranking. It reads and writes no owner-worktree files.
