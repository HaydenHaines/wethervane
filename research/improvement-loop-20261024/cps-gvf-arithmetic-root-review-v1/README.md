# Independent worked-example check

Root checked the four transcribed examples against the 2018 and 2020 Census
technical documentation, independently re-ran all nine tests, and re-ran
`check.py`; all four rounded standard errors matched the cited illustrations.
The Census text identifies the two unemployment examples as basic-CPS Table 7
and the two reported-voting percentages as November supplement Table 8, with
person bases and percentage-point inputs exactly as transcribed. Sources:
[2018 technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov18.pdf)
and [2020 technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsnov20.pdf).

The program explicitly rejects negative radicands and unsupported state
mappings; the nine-test run covers these and other invalid inputs. Its mapping
registry checks a declared characteristic and units but cannot detect a user
entering a proportion when they mean a percentage, or validate a population
base. It only reproduces these four published national illustrations; this
does not validate arbitrary observations or a probability model. The reviewed
code and evidence are in `../cps-gvf-arithmetic-sol-v1/`. Separate geographic
rules for Tables 9–14, covariance, reporting bias, CPS item routing and
historical source availability remain outside this arithmetic check.
