# E01: actual Senate integration and data audit

2026-10-25 UTC. Read-only design audit. No owner source, frozen input, fitted bank, active run, or application default was changed. No fit was launched. This applies the integration repository's `refine-model-layer` and `verify-election-model` skills and its architecture/operational specification. Evidence below concerns the files inspected, not future process liveness.

**Verdict:** a headline-CPI-only *aggregate preparatory comparison* has an existing executable extension API and actual earlier Senate labels. It does not require inventing a training panel. A full selected-count E01 replacement remains a separate scientific target and joint-refit task: the current count input contains only TX 2020/2024 training events, and its historical 2018 assembly has no historical political training likelihood. The aggregate panel is not a substitute for those missing count inputs.

## Actual models and the precise extension boundaries

The latest inspected checkpoint is Senate `checkpoint-root-v19.json`. Its selected plan is `partial-affine-completion-root-v1/plan.json`, purpose `actual_current`, cutoff `2026-10-20T22:13:33Z`. It binds eight complete protected parents with original masses 0.125. Its original/effective Problems, role manifest and source admission are explicit plan inputs. The effective Problem has TX 2020/2024 training and TX 2026 prediction events; two official candidate-count records; and earlier survey/participation evidence. It has no pre-2018 training event.

The physical selected choice mean is implemented in `cycle-separation-astra-v1/joint_role_cycle_separated.py` and preserved through the selected share/partial-affine/worker wrappers. Its decomposition is also extracted in `modular-count-graph-astra-v1/components.py`, `Fundamentals.mean`:

```
choice_intercept
+ choice_relative_lean * relative_lean
+ choice_national_carryover * national_pres_logit
+ choice_president_midterm * (president_party * midterm)
```

It separately owns candidate incumbency. These count priors are intercept Normal(0,.25), relative lean Normal(1,.35), national carryover Normal(1,.25), president×midterm Normal(0,.25), and candidate incumbency Normal(.30,.25). The `Environment` provider already owns the correlated two-dimensional turnout/choice cycle effect and Student-t5 race discrepancy. CPI must change the explained mean once; adding another independently sampled national “economic shock” would duplicate that responsibility. In `Choice.utility`, the D/R contrast is +.5/−.5, so an added structural mean δ changes the conditional D-minus-R utility by δ, with no extra factor of two. This is not automatically an equal δ change in a population-aggregated share.

Two real API limits matter:

1. `src/senate_count_stack_v1/inputs.py::validate_problem` requires exact top-level/event/contest keys. Adding an arbitrary CPI field to a frozen Problem fails. Sources can have role `covariates`, but that does not create a new numerical field or authenticate a historical release.
2. The modular compiler exposes `ComponentSpec(configuration_json, dependency_pins)`, `Registry.register`, `Composition` and `compile_model`. A new source-bound Fundamentals provider can retain the two output ports `structural_choice` and `incumbency`, own one new parameter, and bind a frozen per-cycle feature sidecar in its specification. It cannot keep hidden mutable data in the provider instance. However, the current executor resolves only its sealed standard registry and `require_baseline_eligibility` rejects a replacement. An E01 provider needs a new reviewed registry/executor selection and target/admission; disabling that guard or borrowing the baseline certificate is not an integration path. None of the live 662-dimensional affine constants or current fitted columns qualifies a changed model.

The available aggregate route is different and already extensible. `src/forecasting/fundamentals_v1.py` contains four ordinary D/(D+R) laws, all with Normal cycle and Student-t5 race discrepancies. `src/forecasting/aggregate_component_factory_v1.py` exposes `Admission.extra_feature_ids`, `ParameterSpec`, `MeanTerm`, `ProviderSpec(kind='mean', kernel='linear')`, `Recipe`, `compile_training`, `fit_compiled`, `predictive_basis` and `condition_polls`. Its `environment` mean role explicitly requires a feature to be constant within each election cycle. `application-extension-root-v1/linear_provider.py` demonstrates registration and one extra owned slope, but is explicitly synthetic and supplies no economic data admission.

No edits to the old four-law enumeration are needed: `standard_providers()` is the existing relative-context recipe; `legacy_recipe('senate_history_refreshed_v1')` is the existing refreshed-history recipe. A new economic provider can add one `MeanTerm` on an explicitly admitted extra feature. The factory does **not** independently check that each historical economic release was available at that race's own historical cutoff; its broad admission checks are insufficient for that purpose. A new source adapter must perform this join before compilation.

## Actual historical labels, not an invented panel

`research/fundamentals-iteration-20261020/panel-sol/receipt.json` binds:

- `features/all-cycles.csv`: 443 label-free feature rows, including 26 ordinary current 2026 pairs.
- `labels/outcomes-SEPARATE.csv`: 417 completed ordinary regular Senate D/(D+R) labels, 1998–2024; actual D and R candidate-vote numerators and the D+R denominator are retained.
- `eligibility-ledger.csv`: explicit exclusions rather than fabricated zero outcomes.

I counted eligible rows, matched outcome IDs and verified all 13 receipt-bound panel outputs. Completed per-cycle support is 32/31/26/31/30/30/28/31/29/31/30/28/30/30 for 1998/2000/2002/2004/2006/2008/2010/2012/2014/2016/2018/2020/2022/2024. All pre-2018 eligible IDs have an actual separate label.

| Forecast target | Tier for 2026 midterm development | Earlier CPI-compatible cycles | Actual earlier ordinary Senate labels | Gasoline 2002+ matched labels |
|---|---|---:|---:|---:|
| 2018 | A | 10: 1998–2016 | 299 | 236 |
| 2020 | B | 11: 1998–2018 | 329 | 266 |
| 2022 | A | 12: 1998–2020 | 357 | 294 |
| 2024 | B | 13: 1998–2022 | 387 | 324 |

Target support is 30/28/30/30 respectively. Preserve that exact ordinary-contest universe and existing exclusions, including specials, unsupported rules, independent-principal contests, 2000 MO, and the post-cutoff 2002 MN nominee replacement. Fusion lines were merged by named candidate before major-party totals. Missing all-candidate/participation totals remain missing. The panel's status is `qualified_retrospective_proxy`: final rosters and reconstructed source availability do not become strict historical ballot publication evidence. Its raw 1998–2018 Senate lineage is the byte-bound MEDSL/Clerk mirror; later derived official labels and the nine retained source revisions have separate lineage.

`iteration-root/fit_runner.py::training_data` already provides the correct label-access sequence: inspect cycle metadata, select earlier rows, then parse numeric outcome columns. The new experiment should reuse that sequence. Loading target outcomes and then zeroing a design column is not an ablation.

For count inference, the separate `historical-political-assembly-astra-v1/prepared-v3/candidate-problem-NOT-ADMITTED.json` has one 2018 prediction event, zero surveys, zero official-count training records, zero participation reports, and one 12-LV-poll block. Its real eight-parent population port is useful, but the political Problem is explicitly unadmitted. The historical population bank supports a 2012–2016 source period with 2010–2019 paths, not automatically all 1998–2016 elections. The aggregate D/R panel cannot silently fill missing protected population paths, respondent measurement bridges, event participation, candidate menus or noncandidate categories. In particular, D+R votes are neither event participation nor all candidate counts.

## Cutoff-compatible economic coordinates and the remaining source gap

The CPI v1 bundle has 14 even cycles, 1998–2024, with headline and core CPI-U U.S. city-average NSA twelve-month percent changes. I verified all 20 files bound by its receipt. The TWIP bundle has 12 even cycles, 2002–2024; all 32 bound files verify. It contains dated national regular-retail gasoline current/exact-91-day-lag pairs, dimensionless log changes, source units and observation ages. These are eight-calendar-day-old Monday observations, not the separate latest-Monday GDFU schedule-proxy arm. Its 2004 byte-bound date-header exception, May 2018 method break, conservative next-day availability bound and unverified original posting logs remain explicit.

The CPI selector is reusable as-is: `cpi_vintages.select(rows, cutoff_utc, election_date, horizon_days=14, series_id='CUUR0000SA0')`. Eligibility is inclusive: both release and vintage timestamps must be <= the actual cutoff, with the declared staleness bound. V1 selections use H14 **23:59:59 UTC**. The historical count builder uses H14 **00:00:00 UTC**, while aggregate `forecast_cutoff` currently stores a date parsed as midnight. Do not silently reinterpret that date as the end of the day.

I ran the actual selector at midnight. Only 2004 loses an existing v1 selection: September 2.5% was released at 12:30 UTC on the October 19 cutoff date. The present bounded inventory has no earlier eligible release and correctly returns unavailable. Missing is the **August 2004 release, September 16, 2004, 08:30 EDT**, not an invented zero or the later September value. The [BLS 2004 release schedule](https://www.bls.gov/schedule/2004/home.htm) and the actual [FRASER archived BLS PDF route](https://fraser.stlouisfed.org/files/docs/publications/bls/newsreleases/cpi/cpi_20040916.pdf) establish a concrete acquisition route; this audit did not acquire or admit that missing row. Root assigned a separate `economic-cutoff-adapter-v2` successor; v1 bytes must remain intact.

**Successor update before this audit closed:** `economic-cutoff-adapter-v2` now acquired and byte-bound the September 16 August release, preserved v1, and produced all 28 inclusive-midnight headline/core selections. Its 2004 headline/core values are 2.7/1.7; all A/B values are unchanged. The specific v1 missing-data gap is therefore resolved by this separately owned intake. `economic-experiment-v1` consumes that pinned successor rather than moving the model cutoff. The remaining missing inputs are a 2026 economic selection and the full-count political/population/history support described above.

All four A/B headline values are already before midnight: 2018 2.3%, 2020 1.4%, 2022 8.2%, 2024 2.4%. The 2014 earlier value must remain August 1.7%, because September's October 22 release follows the October 21 cutoff. There is no admitted 2026 CPI or gasoline row in these historical bundles. Neither is a real-income/wage/DPI feature bundle. Those omissions do not block a historical headline-only experiment, but they block a claim to have integrated a current 2026 economic forecast or tested purchasing power.

## E01 scalar and three matched aggregate arms

Root selected these exact existing-family comparators for the preparatory experiment:

1. **H0**: existing refreshed-history aggregate recipe, fixed preceding-presidential slope one, office intercept and signed nominee incumbency; existing cycle/race priors unchanged.
2. **F0**: existing aggregate relative-context recipe, with learned relative state lean, national carryover, president-party main effect, president×midterm and incumbent nominee adjustment; existing priors unchanged.
3. **E01**: F0 plus exactly one CPI mean coefficient. It is not the count mean, and the extra party-main term/prior differences above must not be hidden.

The retained `best_guess_polling_20261020_v2` can be shown only as a separately identified archived reference with its different source/bank/metadata history. It is not renamed the same-information F0. The earlier final decision retained that default and did not select the new aggregate arms for count deployment.

Use source-bound presidential party **p=+1 D/−1 R at the forecast cutoff**, not the incumbent Senate party or eventual presidential winner. The panel and count schema already use that orientation. In particular 2020 is −1 and 2024 is +1. Keep the original CPI value in percentage points. Define `u_c = p_c * CPI_yoy_percent_points_c`, then `x_c = (u_c - mean_train(u)) / sd_train(u)` across **unique earlier cycles**, with population SD (`ddof=0`) frozen for prediction. Standardize after orientation; centering unsigned CPI first creates a tied party-main shift that can obscure the baseline comparison. Repeat the same national x across races in the cycle, without treating those rows as independent macro realizations. The added coefficient is `Normal(0, 0.05)` logit units per earlier-cycle SD, symmetric and proper. A negative slope can encode a presidential-party inflation penalty; no sign is forced. Keep priors and residual laws from F0 unchanged, and retain zero-CPI restriction tests.

For reference, the *v1 end-of-day* oriented mean/SD before 2018 is .03/2.485176 percentage points; 2022's oriented feature is 3.562229 training SDs, and its unsigned 8.2% exceeds the earlier observed 1.1–4.9% range. These are design/extrapolation diagnostics, not coefficient estimates. They are retained only to explain the v1 diagnostic and are not frozen production scalers. The completed midnight-v2 preparation recomputes them: pre-2018 oriented mean/SD .01/2.506172, and the 2022 target is 3.545259 earlier-cycle SDs.

CPI is a national price-change measure, not observed state inflation or a measured voter budget. BLS explicitly does not publish CPI inflation for arbitrary population subgroups in its [CPI FAQ](https://www.bls.gov/cpi/questions-and-answers.htm). The new feature overlaps F0's party/context means and shared cycle discrepancy; it has 10–13 independent historical election-cycle realizations in these folds. Prior sensitivity and posterior correlation with those terms matter more than a race-row significance calculation. No new state slopes, turnout effect, interaction, core-CPI block or gasoline term belongs in E01.

Gasoline is a later incremental comparison. Its current archive begins in 2002, losing 63 ordinary 1998/2000 labels. Therefore a gas addition must compare with H0/F0/E01 refit on the same 2002+ support, or first acquire the missing two cycles. It must not call that change an algorithm gain. Headline already contains gasoline prices; any later gas term asks about incremental information, not an independent accounting component. No real-income feature can be reconstructed by relabeling current nominal-income/QCEW/BEA caches.

## Frozen aggregate preparation and prior review

The authorized new package `economic-experiment-v1/prepared-v1/plan.json` compiles all three arms on the actual matched rows and pins the midnight-v2 intake. Its SHA is `cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9`. Seven preparation tests pass, including complete H0/F0 normalized-density equivalence at supplied points in all four folds, the E01 zero restriction, the analytic economic gradient, source tampering, cutoff guards, and unique-cycle/target-isolated scaling. No fit was launched; `fit_authorized` is false.

Orientation-first and unsigned-inflation-first standardization span the same mean functions because F0 has office and party terms, but do not give the same joint prior with independent Normal(0,.05) coefficients. If x=(pπ−μu)/su and x_alt=p(π−μπ)/sπ, then x_alt=(su/sπ)x+μu/sπ−(μπ/sπ)p. The selected coordinate creates a coefficient-linked office offset rather than a coefficient-linked party offset. This is a declared shrinkage/prior convention, not an empirical superiority claim. The experiment README gives the exact rationale, correlations, units and both coordinate diagnostics.

At the 2022 covariate, the selected prior adds a mean-zero logit shift with 95% limits ±.347429 (a fixed .5 baseline would become .414006–.585994). The alternate coordinate with independent same-SD prior gives ±.569640 (.361320–.638680). A single predeclared tighter SD .025 sensitivity halves the selected incremental intervals; no target score selected it. The package records fixed-seed 4,096-draw prior predictive checks on actual covariates, with shared residual/parameter standard draws, and explicitly marks the alternative centering diagnostic as not an authorized fit arm. Party/national-feature correlations are .81–.92/.68–.89, so confounding remains substantial despite a full-rank design.

Conditioning is fold-specific: each model learns only from that fold's earlier labels; a later fold can legitimately use an earlier evaluation cycle. This preparation compiles all four folds and does not claim globally unopened labels. A future runner must seal each fold's three predictive arms before scoring its target. The initial proposed stage is F only; FP needs a later plan pinning actual matching polling factors. A compatible fitter runtime is also a separate prelaunch requirement; the package certifies preparation and supplied-point arithmetic, not PyMC execution.

## Minimal execution and review requirements

The next allowed step is a new source adapter and frozen aggregate experiment plan, not sampling. Bind both original panel receipt and cutoff-v2 receipts, all exact source/code hashes, unique-cycle scalers, per-fold training IDs and target IDs. Require every historical macro release/vintage <= that row's exact H14 midnight, and every coefficient-training outcome cycle < target. Preserve the roster proxy qualification and distinguish source retrieval time from nominal historical availability. Test coefficient/design equivalence of H0/F0 against the old implementations, the E01 zero restriction, label mutation isolation, party reversal, unique-cycle standardization, feature/source tampering, 2004 same-day rejection and 2014 future-release exclusion.

After root review authorizes fitting, three mean laws × four target folds require **12 complete joint aggregate fits**, with identical data, cutoff, sampler budget and diagnostics across arms. Fit all coefficients and cycle/race scale parameters jointly; no old-bank slope splice, post-poll offset, independent economic posterior multiplication or saved-column zeroing. An exact existing whole-bank reuse is a separate source/recipe identity decision, not assumed by this plan. Existing aggregate reference settings provide four chains, 1,500 warmup, 4,000 retained draws per chain, target_accept .97, depth 10, rank Rhat <=1.01, bulk/tail ESS >=400, zero divergences and depth hits. Full predictive precision remains separately assessed; do not lower gates because a new coefficient is small.

Start with F outputs for all three arms, then paired FP predictions using the **same** already-admitted poll factor within each fold and unchanged uncertainty ownership. Actual aggregate `condition_polls` is a declared aggregate posterior-mixture comparator, not a new normalized population-cut engine. Its target outcomes enter only after all predictions in the batch are sealed. Reuse `benchmark-sol/runner.py`'s score/diagnostic machinery through a new versioned runner; its old exact-arm validators will not accept a new E01 enum or bank automatically.

Show A2018 and A2022 individually and equally averaged; B2020 and B2024 individually and equally averaged; older C evidence separately. Use share CRPS as the declared aggregate primary score, with WIS, coverage plus width, winner Brier/log score, always-.5 and earlier-only frequency references, fixed common competitive masks, and near-even margin diagnostics. Keep any eligible-seat subtotal explicitly separate from chamber control. Preserve new-feature extrapolation and all negative findings, including gains that vanish after polls. These repeatedly visited development elections do not become untouched holdouts. Aggregate success would support a later full-count E01 design; it would not qualify or promote count fundamentals.

For the full count model, the minimum remains a source-qualified earlier political Problem, population/measurement/event/menu support, new owned economic provider/registry/execution admission, and a **fresh complete conditional fit for each original protected parent**, preserving parent masses and posterior row identity. Historical A/B reference count fits are not yet established by these inspected inputs. Finish the owner's selected reference rather than launching a competing current fit.

## Authenticated local inventory

Paths below are relative to their stated root. SHA-256 values are observations at audit time. The two economic receipts bind the complete 20/32-file inventories; the panel receipt's 13 output hashes all verified. This inventory binds bytes and scope, not historical publication truth or scientific approval.

| Root | Exact path | SHA-256 |
|---|---|---|
| Integration | `AGENTS.md` | `0a27fc2d01b021422362b278f8a7253ff8b9c2d8297a92625290338d45297fbd` |
| Integration | `.agents/skills/refine-model-layer/SKILL.md` | `4c7af4ede4631ad82ede3ab3bc8505dcd98bf5eb875fd2a0444f9f83c3ec2cab` |
| Integration | `.agents/skills/verify-election-model/SKILL.md` | `d8421eff4c49ac360eeab64fcbc4f551cfa5cb5f72da05b5405697e156ba57b2` |
| Integration | `docs/ARCHITECTURE.md` | `c4a1cb0e03058c9ef1d4633361f329edff787816d7f3c18cf0189c137928ee10` |
| Integration | `research/model-stack-operational-specification-20261009.md` | `b471fc6a5ed85ea59847ac00469c99cf18bef276b4bb86e56f2a6dd27d0314f1` |
| Integration | `src/senate_count_stack_v1/inputs.py` | `6f4a1f5a747432c69a12392f1d1c2a1f4c4c3df3c1f16e886c44e1089438cb3e` |
| Integration | `research/senate-stack-integration-20261020/checkpoint-root-v19.json` | `a33e8b9c9b2f5add36c838da41e62918c0386f826c8aa7bd3f1251896d1e173b` |
| Integration | `research/senate-stack-integration-20261020/partial-affine-completion-root-v1/plan.json` | `adef60442a7097dc581323f9e361bb2a75441e9292eb92912c2fa1a0900b61b1` |
| Integration | `research/senate-stack-integration-20261020/actual-pilot-preparation-sol-v1/prepared-v4/problem.json` | `96f890fae93fec40caa0b58fb67b7df264259fc533b5d7a23eab70c8a4dc7034` |
| Integration | `research/senate-stack-integration-20261020/cycle-separation-astra-v1/joint_role_cycle_separated.py` | `06a3351cf7bf05ef15599af4c5224ba1fd5a427b2f3df21814d7a3bfdf439bfa` |
| Integration | `research/senate-stack-integration-20261020/modular-count-graph-astra-v1/components.py` | `7baf3ea8e644c025763e5a63b1721b14ddc748a94be3f1e106adf9f34f6ad2a7` |
| Integration | `research/senate-stack-integration-20261020/modular-count-graph-astra-v1/graph.py` | `f8c9417c371e405cc24c884035cded3598746a043b74f183079e297d72108131` |
| Integration | `research/senate-stack-integration-20261020/modular-execution-sol-v1/provider.py` | `f13864ed0a9aa037fbbe0ad9a5ab1326db264e338223459953d5f9a48d334d08` |
| Integration | `research/senate-stack-integration-20261020/historical-political-assembly-astra-v1/prepared-v3/candidate-problem-NOT-ADMITTED.json` | `d1197d3a8b83912eedc292877d9a90e800ab1a81740d56b798c2ca65fe251744` |
| Integration | `research/senate-stack-integration-20261020/historical-political-assembly-astra-v1/receipt-v2.json` | `670bec7e07276e042ea280abd5fb3196095be05284555aa4972941dbf34b6d3f` |
| Integration | `src/forecasting/fundamentals_v1.py` | `d1437063c84cb9276382b5f96d04577d8ccce04c8aeaf4d1e20420e258d7fac9` |
| Integration | `src/forecasting/aggregate_component_factory_v1.py` | `583f4e204360fc71f980bd70a86aa3f27afaefaee13a27c5f1833b8b54ae5366` |
| Integration | `research/fundamentals-iteration-20261020/iteration-root/fit_runner.py` | `f065e198d834c563dfe876724e69a4178c26301482c7310b004340bdd6e38756` |
| Integration | `research/fundamentals-iteration-20261020/benchmark-sol/runner.py` | `aaaba513f0b2180824aab4c9139f64f1afe6f4c8f08e6905300375896508eed5` |
| Integration | `research/fundamentals-iteration-20261020/panel-sol/receipt.json` | `3beff840bcf37aa348cd9bb8ba5b28535ba752d717d45d6a9a9f97503e9ca7f3` |
| Integration | `research/fundamentals-iteration-20261020/panel-sol/features/all-cycles.csv` | `0a2a1b9c15a83f33406828e7b7e2c1b24230570ed0639c995c410b2bbe9ec786` |
| Integration | `research/fundamentals-iteration-20261020/panel-sol/labels/outcomes-SEPARATE.csv` | `5f5cafa1afc53387331d4e75b3c65f92ed4cab3a91578dcb7cb0815357b029ad` |
| Integration | `research/fundamentals-iteration-20261020/panel-sol/eligibility-ledger.csv` | `72341393dd930f36f30bb59a865aa5da5722c9e6dba01a3e3bf593003fc8a000` |
| Integration | `research/fundamentals-iteration-20261020/final-decision-root-v1/DECISION.md` | `e45cb7fcd13de4776a7904b373eea11bdc23423421de67b4af78b98937c38f16` |
| Lab | `research/improvement-loop-20261024/economic-vintages-v1/receipt.json` | `7635903576d1154634fe96b9fcab9fcadc261f01827af92eb3ddd75810e22f6f` |
| Lab | `research/improvement-loop-20261024/economic-vintages-v1/rows.json` | `02d54e39994fe41500c33c881b0fc70389deea41f66073d35ebb62e3e89099c4` |
| Lab | `research/improvement-loop-20261024/economic-vintages-v1/selections.json` | `18cb3bd821ac036658423e6d1495d2295fd5d9430b65c58bbc7b200804b918cf` |
| Lab | `research/improvement-loop-20261024/economic-vintages-v1/cpi_vintages.py` | `42473ae777f1e17c1704b93894d9cb37bf92ea456a0a6245d7958da29198dafc` |
| Lab | `research/improvement-loop-20261024/gasoline-twip-cutoff-v1/receipt.json` | `329ddc83d6f17b8399363f77dbaacc15c0b7f9fded29b1f2d9f859be88c4b397` |
| Lab | `research/improvement-loop-20261024/gasoline-twip-cutoff-v1/panel.json` | `9f4627170f846bfa6768e2bda43659a6218c4a577a2daf92989b14bd78ed3902` |
| Lab | `research/improvement-loop-20261024/gasoline-twip-cutoff-v1/intake.py` | `c7a2c9f7ebaf25484fd803f3c0d7ba6a2bb4c43a50c73f69b10d97d905cb1272` |


Additional successor bindings at audit closure:

| Root | Exact path | SHA-256 |
|---|---|---|
| Lab | `research/improvement-loop-20261024/economic-cutoff-adapter-v2/receipt.json` | `4646bc2ea152f441a1b34388ce4bd5693306a065947c3f0d17a09c1f781ac733` |
| Lab | `research/improvement-loop-20261024/economic-cutoff-adapter-v2/selections.json` | `208c02593a8f39771ae23248ec2c5608aa493948cda2d90d64a53fdc3fbf0a0b` |
| Lab | `research/improvement-loop-20261024/economic-cutoff-adapter-v2/adapter.py` | `9c15cf3bb100cadc20d10500aee8c4881f5202e706781db98acb366a24880924` |
| Lab | `research/improvement-loop-20261024/economic-experiment-v1/prepare.py` | `06849868834bd177a21bf4030c119394c2b84cae0364f12ac54f9fb8450c6695` |
| Lab | `research/improvement-loop-20261024/economic-experiment-v1/test_prepare.py` | `7267c3e653992a7029410ba708088d584df9ca8ebcf7d7592a031054012ebb27` |
| Lab | `research/improvement-loop-20261024/economic-experiment-v1/prepared-v1/plan.json` | `cfe669f6cef8cec82702b03e9ced060edf022fd434df131483515a054d24aef9` |
