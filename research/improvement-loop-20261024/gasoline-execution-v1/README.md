# G01 aggregate execution adapter

`prepared-v2/plan.json` (SHA-256 `63f6de0e57949857a48d02e5084af8cc9d576b4ffa96985b13afc938fe55b5ab`) schedules twelve **fresh** joint fits: H0, F0 and G01 in each of 2018, 2020, 2022 and 2024. It pins `gasoline-experiment-v1/prepared-v2/plan.json` exactly. `prepared-v1` remains preserved but superseded after the fit-gate validator was added; see `SUPERSESSION.md`. No actual posterior fit has been launched.

The execution code is a narrow version of the reviewed E01 aggregate runner. It retains the owner H0/F0 laws, four sequential independent chains per job, 1,500 warmup and 4,000 retained draws per chain, fit/predictive numerical gates, five future replicates per joint row, CRPS and secondary scores, chain/batch MCSE and failure/continuation preservation. The F0 predictive bank freezes common competitive and near-even masks. In each fold, all three prediction files and seals, plus F0 masks, must verify before the target numeric-label parser runs. Earlier target outcomes can train later chronological folds; all target cycles are disclosed development benchmarks.

The existing locked `economic-v1-env` and its E01 runtime certificate are referenced **only for the previously reviewed package/runtime behavior**. The new G01 code, preparation and execution plan require their own source/plan review before a real fit. The runner validates its external plan SHA, exact source snapshots, inherited runtime lock, all twelve job budgets/seeds/source bindings and unchanged fit/predictive/score gates before sampler work. It never reuses E01/F0 posterior banks or scores. The current plan has `fit_launched:false` and `promotion_authorized:false`.

Thirteen synthetic runner tests and Ruff passed at the pinned code hashes. The tests use toy posterior banks and do not call `pymc.sample`; they cover complete four-fold sealing/scoring, score math, one-chain failure preservation, continuation identity, source mutation, fit-gate rejection and target-label filtering. The no-sampler CLI validation of the current plan also passed.

Once the new source/plan review is complete, the root runner can use this command with an exclusive new attempt directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/economic-v1-env/bin/python \
  /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/gasoline-execution-v1/runner.py \
  --plan /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/gasoline-execution-v1/prepared-v2/plan.json \
  --plan-sha256 63f6de0e57949857a48d02e5084af8cc9d576b4ffa96985b13afc938fe55b5ab \
  --runtime-certificate /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/economic-execution-v1/runtime/acceptance.json \
  --runtime-certificate-sha256 956a66375738bb5e2be2e0296e734c8e46ee7c94a90e3e19cc3618b27a63fcac \
  --run /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/gasoline-execution-v1/attempt-v1
```

The command is documented for the root workflow; this preparation task did not execute it.
