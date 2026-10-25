# Independent 2018 output review: aggregate E01 replay

Review scope is the completed 2018 fold only, read-only. The other fold outputs are outside this review. No process control, producer edits, refit, or forecast qualification was performed.

## Integrity and diagnostics

The reviewed execution plan is `economic-execution-v1/prepared-v2/plan.json`, SHA-256 `e3faefeb89a5bca837bc7f8b01e666112d563b95fe16e909060edf5aa1cbfc50`. All four 2018 arms (H0, F0, E01, E01_sd025) have terminal receipts, four completed raw-chain receipts, prediction seals, and score files tied to that plan. The chain launch seeds match the plan. Raw NetCDF hashes match both the per-chain completion receipts and terminal receipts. Prediction file hashes match each prediction seal.

Each arm has four independent chain files with 1,500 warmup draws and 4,000 retained draws per chain. Ordered coefficient labels and earlier source-cycle coordinates match each arm's plan binding; retained arrays contain finite values. Each full 5,500-row transition journal was checked against its raw NetCDF warmup and posterior state, including beta, cycle z, and exponentiated log scale parameters; the maximum discrepancy was below `1e-12`. Stored cycle effects equal `tau_cycle × cycle_z` exactly.

I independently recalculated rank-normalized R-hat, bulk/tail ESS, divergences, and max-depth counts from the four saved chains. All frozen fit gates pass: max R-hat ranges from 1.00073 to 1.00130, minimum bulk ESS from 3,234 to 4,924, minimum tail ESS from 3,590 to 5,720, and there are zero divergences or max-depth hits. Per-chain E-BFMI ranges from 0.700 to 0.851. These are computational checks, not evidence of predictive validity.

For each arm, I rebuilt the 2018 target design from the frozen preparation and reconstructed all stored logits from their exact posterior rows and saved new-cycle/race shocks. All four reproduce the stored logits exactly; the shared-cycle component residual is at most `1.3e-15`. Each posterior row has five predictive replicates, one cycle shock shared across all 30 races in a joint replicate, and a race-specific Student-t shock. The 80,000 predictions per arm per race are finite shares in [0,1]; D/R win probabilities are in [0,1] and complementary.

## 2018 scoring cross-check

I used the actual 2018 D/(D+R) shares in the separate target-outcome file, SHA-256 recorded in `independent-verification.json`, and verified that each target race and actual share matches the saved score row. There are 30 races. The saved F0 masks independently reproduce: 11 competitive races (inclusive F0 win probability 0.1–0.9) and 1 near-even race (inclusive 0.4–0.6).

Independently recomputed empirical share CRPS, winner Brier score, and share WIS match the saved race-level and stratum-level values to within `8e-15` (CRPS), exact Brier arithmetic, and `2e-17` (WIS). Scores are loss measures, so lower is better.

| Arm | Set | Races | Share CRPS | Winner Brier | Share WIS |
| --- | --- | ---: | ---: | ---: | ---: |
| H0 | All | 30 | 0.0379624 | 0.116420 | 0.0243932 |
| H0 | Competitive | 11 | 0.0442807 | 0.240769 | 0.0275104 |
| H0 | Near-even | 1 | 0.0457310 | 0.008391 | 0.0287330 |
| F0 | All | 30 | 0.0380016 | 0.118246 | 0.0232645 |
| F0 | Competitive | 11 | 0.0390423 | 0.241553 | 0.0242464 |
| F0 | Near-even | 1 | 0.0262909 | 0.173983 | 0.0198611 |
| E01 | All | 30 | 0.0385463 | 0.118419 | 0.0234979 |
| E01 | Competitive | 11 | 0.0392876 | 0.241578 | 0.0243631 |
| E01 | Near-even | 1 | 0.0271500 | 0.183698 | 0.0203403 |
| E01_sd025 | All | 30 | 0.0382147 | 0.118480 | 0.0233732 |
| E01_sd025 | Competitive | 11 | 0.0391045 | 0.242222 | 0.0242667 |
| E01_sd025 | Near-even | 1 | 0.0268282 | 0.181093 | 0.0201979 |

These are descriptive scores for one retrospective election cycle. They do not establish a general improvement, defeat, adoption decision, or performance in remaining folds. The 2018 outcomes have nonzero empirical winner mass in all 30 races, so this fold has no constant winner-tail case. The broader scoring contract still treats constant rare-winner tails as unresolved where they occur; absence in this fold does not resolve that limitation.

## Artifacts and reproduction

- `independent-verification.json`: plan/truth pins, chain/terminal/prediction checks, diagnostics, race scores and mask-stratified summaries.
- `prediction-composition.json`: independently rebuilt 2018 target-design prediction formula checks.
- `review_2018_economic.py` and `recheck-prediction-composition.py`: read-only audit scripts. `review_2018_economic.py` uses the isolated `economic-v1-env` pinned by this execution (`Python 3.12.3`, `NumPy 2.4.4`, `SciPy 1.17.1`, `ArviZ 0.22.0`, `h5py 3.16.0`, `h5netcdf 1.8.1`). The prediction-design reconstruction also uses that environment and imports the frozen preparation module read-only. Neither script calls the execution runner or writes to its output tree.
- `receipt.json`: SHA-256 and byte inventory for this review packet (receipt omits its own hash).

The plan's own status remains authoritative: `forecast_qualified: false` and `promotion_authorized: false` for every reviewed arm.
