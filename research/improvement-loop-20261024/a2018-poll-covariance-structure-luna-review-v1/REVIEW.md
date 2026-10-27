# Independent review: A2018 prepared poll covariance structure

Target: `/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/a2018-poll-covariance-structure-root-v1`.

Review is aggregate-only. No poll/contest identifiers, row values, or matrix entries were printed. No original packet, model job, or site was changed. Independent numerical runs used Python 3.12.3 / NumPy 2.4.4 with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

## Source pins

- Prepared problem: 6,979,007 bytes, SHA-256 `abcaf347e682422303780c5ff12bb3e6179e4b98362686c01338e29df06a385a`.
- Selected provider source `117-components.py`: 27,626 bytes, SHA-256 `1574e7831344009f08470b9287b257556a9c031fb7e33e29938a70d51b1a854d` (matches the separately pinned source-evidence resolver binding).
- Target `README.md`: `dd109f8cf11da6b5e0f07d696eff84a677dd71451ee011f36bfb4906eea1ff32`
- Target `check.py`: `dcc270947c6ce025f9d2c2ea122fe15660df97dcdce0101a42d703b0c556940a`
- Target `result.json`: `3809d1cb0d1da0eef18e50ba35c154f31a02bbb9b7f446002e72cf5eab41bb85`
- Target `test_check.py`: `c9685502df5276f5619be4fe575efe434a8807fe78f45fa1a37db70e9c05ed4f`

## Independent reproduction

The pinned prepared problem contains one polling block with 274 rows and 29 distinct contest labels, hence 37,401 unordered row pairs. Independently formed standardized covariance as `cov[i,j] / sqrt(cov[i,i] * cov[j,j])`, categorized every pair by equality of the prepared contest label, and summarized signs/ranges only. Results match the packet exactly:

- Same-contest: 2,110 pairs, all positive, no exact zeros; minimum 0.1660678695836024, maximum 0.8939563723199514.
- Different-contest: 35,291 pairs, all positive, no exact zeros; minimum 0.018865805841037368, maximum 0.3996626243329627.

The selected provider source is pinned by the resolver binding. Its poll construction loops over prepared poll blocks and issues one `pm.MvNormal` call per block, with the block's ordered means, covariance, and observed vector. Since the pinned problem has exactly one block, source-level construction declares one multivariate-normal likelihood block for these 274 rows. This is evidence of source intent only, not proof that an executed/compiled model consumed the pinned bytes or that the factor was evaluated.

Ran target tests and CLI with thread limits set to one. All three unit tests passed; CLI output matched the independent aggregates.

## Review findings and scope

No defect found in the aggregate computation or output boundaries. It verifies the source byte pin before parsing; it exports only counts and correlation extrema/sign totals, plus status flags. Pair classification is performed with prepared `contest_id` equality; the values themselves do not leave memory. The fixture helper is correctly described as unauthenticated, while the frozen path validates exact bytes.

The conclusions are appropriately bounded. Positive standardized covariance across different contest labels documents the supplied likelihood matrix's cross-contest structure. It does not establish real poll sample overlap, actual poll-error dependence, dependence among separate survey sources, or scientific adequacy. Contest labels are not respondent/sample identities. This study is not an SPD or conditioning check (those are treated separately); neither the provider source nor frozen bytes authenticate actual graph execution, historical source availability, or forecast qualification. The README's caveat that these are supplied likelihood covariances rather than posterior outcome correlations is correct.
