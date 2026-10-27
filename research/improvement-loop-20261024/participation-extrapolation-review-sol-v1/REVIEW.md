# Independent unadmitted participation prototype review

Pinned owner files are listed in source-pins.json. Owner code remains unchanged.8 inherited tests passed and4 independent invented tests passed in fresh temporary source copies under nice15, each suite0.001s. No actual outcomes, training data, forecast inputs, banks, model fitting or owner jobs were used.

## Verified arithmetic and scope

conditional_pmf iteratively convolves independent per-person Bernoulli terms sharing each cell probability. Its support is0..sum eligible, bounded to200 synthetic people, with total probability checked within1e−12. An independent Fraction enumeration over all8 outcomes for3 invented people reproduces the heterogeneous-cell PMF and moments. “Exact” here means the full finite distribution rather than Monte Carlo/normal approximation; implementation still uses binary floating arithmetic.

conditional_moments sums n×p and n×p×(1−p) under fixed supplied integer populations/offsets/scenario. Mixture covariance uses law of total covariance: scenario covariance of conditional means plus mean conditional variance on the diagonal, assuming events conditionally independent. Independent Fraction enumeration of the full2-event joint law confirms covariance3/8 and both event means. This dependence is shared latent-effect uncertainty, not modeled longitudinal voter identity or shared individual sampling; it does not include unknown eligible-population uncertainty, additional event shocks, or cross-contest allocation. Reusing cell IDs across events does not create shared people.

The anchored additive count-design rank uses exact Fraction elimination of0/1 rows for observed nonzero-population events. It is an honest structural diagnostic, not a fitted likelihood, a data-admission check or proof of practical identifiability. The observed flag supplies no count observation. Tests confirm missing-cycle columns remain unidentified. A full-rank one-event design can have floating sigmoid saturation/zero conditional variance at offset1000; our independent test demonstrates why full rank cannot certify usable numerical information. Fixed known cell offsets/populations are assumptions; if learned or uncertain, extra confounding may exist outside this matrix. No unseen-cycle effect is learned by this code.

README correctly warns that fractional/uncertain national exposure projections cannot simply be rounded into fixed eligible voters, and that a future cycle needs a separate justified temporal/hierarchical law. Scenarios are caller-supplied, no numerical prior or training/forecast data is inferred. No statistical improvement or default promotion follows.

## Concrete normalization boundary defect

Lines173–177 accept scenario weight sums within absolute1e−12 of1, then use unnormalized weights in means; lines183–185 use those same weights/uncentered sums in covariance. For two identical scenarios with p=1 (finite intercept1000), weights.5 and.5000000000005 are accepted. A correctly normalized mixture is deterministic atN with variance0, but accepted output can exceed support and create spurious variance:

| Invented eligible N | Mean excess above N | Reported variance |
|---:|---:|---:|
|1,000,000|5.0012e−7|2.5012e−13|
|10,000,000|4.9993e−6|2.4993e−11|
|100,000,000|5.0008e−5|2.5008e−9|
|2^53 (allowed maximum)|4504|20,286,016.00001|

This is a supported-input-domain boundary counterexample, not a demonstrated real-data impact. At election-sized invented totals the observed error is far below one person and negligible relative to ordinary forecast uncertainty; the large maximum-domain failure must not be presented as an election-sized discrepancy or causal explanation. Raw values and exact reproduction are retained in review-tests.log/test_review.py. Conditional moments/PMF are unaffected by this mixture weight issue.

Recommend a separately versioned, explicit policy: either reject weights not summing exactly under the declared representation or create a documented normalized working simplex and use it consistently for both means/covariance, without silently reweighting scientifically protected outer masses. Add identical-scenario splitting/invariance and support tests at realistic and maximum permitted populations; compare to independent normalized Fraction expectations. Do not edit the frozen prototype in place. This review does not mandate a scientific change to actual parent weights.

Remaining limitations: no derivative/compiled graph, scalable likelihood, parameter estimation, denominator/source admission, population-uncertainty integration or actual conditional/posterior normalization test. Frozen dataclasses contain caller mutable lists/mappings, so immutability/provenance protection is not provided; source currently claims synthetic arithmetic rather than an authenticated admission boundary. Supplied scenarios/events must remain stable during a calculation. Near-unit-weight tolerance and finite arithmetic require explicit numerical contract before broader reuse.

Reproduction: copy owner participation_extrapolation_prototype.py and its test into fresh temporary scratch. Run nice -n15 python3 -B -m unittest test_participation_extrapolation_prototype -v. Add this packet test_review.py and run nice -n15 python3 -B -m unittest test_review -v. No heavy computation or model dependencies. No production/scientific qualification granted.
