# Bounded Census GVF worked-example arithmetic gate

Four national worked examples reproduce their published rounded standard errors; nine tests pass. No survey microdata/replicates, model, fit or joint likelihood is used. Original addenda/reviews and owner files remain unchanged. This is documentary arithmetic, not validation of Census sampling uncertainty, source vintage, coverage, registration transitions or predictive performance.

Run from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest -v test_gvf
PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B check.py
```

examples.json records each year, primary URL, one-based PDF/printed page, web-extracted line locator, population/characteristic, parameter/illustration table, input units and published rounding. The authoritative documents are [November2018](https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov18.pdf) and [November2020](https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov20.pdf). Their exact PDF hashes are carried forward from the immediately preceding independent primary-doc retrieval in cps-gvf-independent-sol-v1/source-pins.json. Source selection/formulas were checked directly in primary web text/rendered pages. Web line numbers are retrieval locators, not stable PDF identifiers.

The two total illustrations use unemployed men ages16+ and basic-CPS Table7 parameters a=-0.000031,b=2947, yielding approximately91646 and125071 persons, rounded respectively to92000 and125000 persons. They are basic-CPS illustrative totals, not registration/voting outcomes. The two voting percentage illustrations use Table8 b=3496(2018)/5411(2020), their published education-domain bases and p=49.1/61.4 percentage points, yielding rounded SE0.19/0.23 percentage points. These are four published illustrative inputs, not new independently estimated totals. result.json retains full computed values.

Formula1: SE(x)=sqrt(a*x²+b*x), with x in persons for these examples. Negative a is permitted if the radicand remains valid; negative/nonfinite variance is rejected, never clipped. Formula2: SE(p)=sqrt((b/y)*p*(100−p)), with y the person base and p in0..100, returning percentage points. A fraction such as.491 is numerically within0..100 but represents the wrong intended input for49.1%; arithmetic cannot infer user intent. The examples use explicit units and exact inputs; no silent percent/proportion or thousands/person conversion occurs.

The mapping gate accepts only the four transcribed national mappings. It rejects a changed year, coefficient, parameter table, characteristic, unit or geography and rejects any unsupported state key. No state example is implemented, so no national parameter is silently applied at state scale and no arbitrary supported cell is invented. A future supported state case requires its independently verified worked example, appropriate Tables9–14 rule and versioned mapping. This registry is an inspectable prototype, not security against malicious same-user source edits.

Tests cover all four published rounded illustrations, invalid radicands/overflow, nonfinite/negative/bool totals, invalid percentages/bases/coefficient, wrong year/table/geography/characteristic/unit/b values, unsupported state, incompatible formula arguments, explicit unit misuse and valid negative-a behavior. The initial test invocation used the wrong working directory and failed to import test_gvf; its log is preserved as initial-wrong-cwd-tests.log. The corrected directory run passed all nine tests, and the four-example check completed. No failed result was overwritten as scientific evidence.

Not supplied: general cross-characteristic covariance, arbitrary-ratio variance, a normal observation likelihood, nonsampling-bias correction, individual registration/voting statuses, historical first-release proof or data admission. Matching rounded examples verifies transcription/formula/unit arithmetic only. Using marginal SEs as independent joint evidence would introduce an unsupported dependence assumption; these checks do not authorize that step.
