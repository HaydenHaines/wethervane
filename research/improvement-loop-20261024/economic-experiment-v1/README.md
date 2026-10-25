# E01 aggregate preparation: frozen comparison, no fit

`prepared-v1/plan.json` (SHA-256 `cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9`) is ready for root and independent review. It compiles the
actual ordinary-Senate panel and the exact-midnight CPI v2 selections. It is an
aggregate D/(D+R) preparatory comparison, not a count-stack ranking or promotion.
No MCMC, posterior-bank reuse, current-election fit, or source-owner edit occurred.

The three matched arms are H0, the exact existing refreshed-history law; F0, the
exact existing relative-context law; and E01, F0 plus one CPI mean term. H0/F0
have the same complete normalized supplied-point densities as their existing
`fundamentals_v1` implementations. The new term and its proper Normal(0,.05)
prior are jointly owned by one environment-mean provider. Existing coefficient,
cycle and race priors stay unchanged. E01 at coefficient zero equals F0's density
times that coefficient's Normal prior density. Seven tests check these properties
and the actual source, support, cutoff, label and transformation boundaries.

The plan proposes 12 fresh joint aggregate fits: three arms for each of 2018,
2020, 2022 and 2024, with 299/329/357/387 identical earlier training races and
30/28/30/30 target races. It specifies four chains, 1,500 warmup, 4,000 retained
draws, one core, target acceptance .97, maximum depth 10 and explicit seeds.
Rhat <=1.01, bulk/tail ESS >=400 and zero divergences/depth hits remain required.
These are proposals; the source adapter has no fitting command and the plan
records `fit_authorized:false`.

The conditioning is a coherent aggregate joint posterior over coefficients,
earlier cycle effects and cycle/race scales, given only earlier outcomes and
frozen covariates. Each fold's numeric outcome loader filters cycle metadata
before parsing actual shares. Earlier evaluation cycles can legitimately train
later folds: this preparation compiles all four folds and does not claim that
every evaluation label remained globally unopened. A future runner must seal all
three predictive arms in a fold before scoring that fold's target. It must
preserve failed attempts and immutable joint draw identities. The initial stage
is F only. Matched FP predictions need a separate frozen plan binding the same
existing fold-specific polling factor; they remain the existing aggregate
posterior-mixture target, not a newly asserted calibration cut or population cut.

## Chosen coordinate and its prior

Let p be presidential party at the historical cutoff, +1 Democratic and −1
Republican, and let π be the headline CPI-U twelve-month NSA percentage change
in **percentage points**. Orient first: u=pπ. Compute μu and su over **unique
earlier election cycles**, with equal cycle weights and population SD. E01 uses
x=(u−μu)/su and adds βx to the aggregate logit mean, β~Normal(0,.05). The sign is
unrestricted; a negative coefficient can penalize the presidential party during
higher inflation. State/race incumbent party and eventual winners never orient
the feature. No gasoline, core CPI, purchasing-power or state interaction enters.

The orientation/scaler order is an explicit prior choice, not a proof of superior
prediction. The alternative x_alt=p(π−μπ)/sπ satisfies

```
x_alt = (su/sπ) x + μu/sπ − (μπ/sπ) p.
```

F0 already contains office and party coefficients, so those coordinates span the
same mean functions. Giving an independent Normal(0,.05) coefficient to x_alt
changes the joint prior: it induces a coefficient-linked party offset and a
different per-percentage-point slope. The selected coordinate centers an
oriented level and instead produces only a coefficient-linked office offset;
the already regularized party coefficient retains its separate prior. This is
why the preparation freezes the scaler and records the alternative as a
diagnostic rather than silently switching between them. Neither identifies a
causal economic-voting coefficient.

The source-bound plan contains these exact added-mean prior intervals. The
reported probability intervals are **only an incremental shift of a fixed .5
baseline**, not forecast intervals or fitted election predictions.

| Target | Chosen x | Added-mean 95% logit interval | Fixed-.5 share interval | Alternative x_alt |
|---|---:|---:|---:|---:|
| 2018 | −0.921724 | ±0.090327 | .477434–.522566 | −0.045296 |
| 2020 | −0.483851 | ±0.047417 | .488148–.511852 | 0.811855 |
| 2022 | 3.545259 | ±0.347429 | .414006–.585994 | 5.812760 |
| 2024 | 0.633380 | ±0.062070 | .484487–.515513 | −0.130472 |

The 2022 headline value 8.2% exceeds the earlier raw range 1.1–4.9%. The
alternative centering with an independent coefficient SD .05 would give that
target ±.569640 logit, or .361320–.638680 around the fixed .5 baseline. One
predeclared tighter SD **.025** sensitivity halves the selected coordinate's
incremental intervals. It is not selected by target scores. Any sensitivity fit
would require its own complete four-fold joint fits and the same numerical gates,
not a posterior-column rescaling.

Across the earlier cycles, chosen x correlates .813–.920 with party, .450–.589
with party×midterm, and .678–.892 with national carryover. Those are design
diagnostics and strong confounding warnings, not independent evidence of an
economic effect. The model has 10/11/12/13 macro realizations, not hundreds of
independent national shocks. Prior/posterior correlations and scale sensitivity
must remain visible.

The plan also stores 4,096 prior predictive draws per fold for H0/F0/E01, the
tighter E01 prior, and the alternative centering diagnostic, using fixed seeds
and shared residual/parameter standard draws. These are prior checks on actual
target covariates; they use no target outcome. Marginal prior mass below .01 or
above .99 stays approximately .13–.16% across these cases. Broad existing race
and cycle uncertainty dominates those pooled tail summaries, which therefore do
not erase the materially larger incremental 2022 shift above.

## Source and runtime boundary

The CPI v2 receipt and selection hashes are required before use. Its inherited
v1 parser/row/selection/receipt pins and all 20 original source files are checked,
as are the v2 files. The newly acquired August 2004 release supplies 2.7% at
midnight; the post-midnight September 2.5% remains confined to v1. The 2014
selection stays August 1.7%. All four A/B values are unchanged. Final rosters,
archived-copy provenance and unverified original posting histories retain their
qualifications. No 2026 economic row is available to this adapter.

Generated feature files contain actual source rows plus the one transformed
feature and **no outcome labels**. `feature-join-YYYY-evidence.json` binds the
base panel, CPI successor and transformation. Its hash is the factory Admission
metadata's receipt identity; it is explicitly not a count execution certificate.
The plan binds code/source snapshots, original source pins, design/response
hashes, joint recipes, matched training IDs and target IDs. No learned economic
coefficient or score has been generated.

Preparation tests run in the lab `.venv`: Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, pandas 2.3.3 and Pydantic 2.12.5, as recorded in the frozen plan. A working compatible PyMC/ArviZ fitter
runtime has not been certified by this package; preparation imports no sampler.
The shared lab environment's PyMC/ArviZ incompatibility must not be silently
ignored when a later fitting runner is reviewed. No environment was changed here.

From the lab root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/improvement-loop-20261024/economic-experiment-v1 -p test_prepare.py -v
.venv/bin/ruff check research/improvement-loop-20261024/economic-experiment-v1/*.py
```

`prepare.py --output NEW_DIRECTORY` creates an exclusive successor preparation
inside this package. It never overwrites `prepared-v1` and cannot launch a fit.
Do not rebuild a frozen preparation merely to change a timestamp.
