# Source-bound scoring action

This packet is a score-only implementation. Author verification used invented forecasts/outcomes only. Actual labels remain a later Root action after exact independent acceptance; this packet runs no forecast, optimizer, retry or model promotion.

Copy `root-admission-template.json` to a **new external checkpoint**, not back into this sealed packet. Fill placeholders from Root's immutable selection `183d3544e68493a811092e5933823a5bbc44fd17f900387aee286c17446e8d3e`, the selected job terminals and their prediction-seal siblings. For each job, `numerical_status` is exactly its `terminal.prelabel_summary` pin. Copy each fold's exact ordered training IDs from `terminal.provenance.data_source_pins.bank_reference`; do not read outcome columns to construct admission or frequencies.

Use the already frozen mask receipt `checkpoints/momentum-score-masks-root-v1.json`, SHA `c0606cd51dbf98fb270ffe6db69e8752ac3806384654ce2b8a2d178e0a855e2f`. Do not regenerate masks. The empty2022 near-even mask remains an unavailable cycle and makes its equal-cycle A summary unavailable, rather than dropping that cycle or substituting zero.

Fill the final scoring manifest SHA and accepted independent scoring-review receipt SHA after review. The template already pins the accepted runner/review, Gaussian math v2, paired tails v2, trace collector, selected design/clarification, numerical runtime and admitted separate outcome source. The reference terminal pin's SHA is fixed; fill its path from the sealed Root selection. All three arms of a fold use the same selected attempt. This actual comparison selected attempt0 throughout; the reader rejects triggered initial attempts presented as successful evidence.

Before any target numeric-label read, the reader verifies its own complete source inventory/review/runtime, then every selected forecast/prediction/status/reference/mask receipt, matched execution group/source/runtime, exact event/orientation, full-bank source-row identity, job/trace/component dimensions and exact earlier-training scope. It returns an internally constructed immutable capability. Only then can the outcome reader parse numeric columns and compute fixed training win frequencies. Foreign/future rows are filtered by identity before numeric conversion.

Root invocation after acceptance and explicit scoring authorization:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/economic-v1-env/bin/python \
/home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/momentum-forecast-scoring-sol-v1/score.py \
--admission /absolute/path/to/new-root-score-admission.json \
--expected-sha256 ROOT_ADMISSION_SHA256 \
--output /absolute/path/to/new-exclusive-score-output \
--allow-actual
```

The output must not exist. The action writes an exclusive launch before label access; success binds scores, normalized cycle-influence NPZ plus log-scale JSON, label-access evidence and terminal. Post-gate failures preserve an authentic failed terminal and any outputs; they do not repair sources, masks, probabilities or invoke sampling. All NPZ files are non-object and read with pickle disabled.

Python API: `score.run_scoring({'path': admission_path, 'sha256': expected_sha}, new_output, allow_actual=True)`. Pure arithmetic entrypoints are `score_math.gaussian_race`, `score_gaussian_forecast`, `score_empirical_forecast`, `paired_difference` and `equal_cycle_summary`. Direct supplied arithmetic does not open files or acquire forecast qualification by itself. Actual orchestration applies prelabel core/bulk and frozen-failure status before any score is qualified.

Output covers all fixed common/competitive/near-even races, each cycle, equal-cycle A/B summaries and primary FS6−FS5 differences. Required per-race statuses are retained. Log-score aggregates never use only successful races. Exact target ties stay in share scores and are excluded identically from binary scores with explicit counts. Conditional MCSE is not an electoral generalization confidence interval; inherited-bank and calibrated-scale uncertainty remain separate. F0 WIS is exact for its empirical distribution but has source-chain dispersion only, without fabricated density or certified nonlinear WIS MCSE. Display status is separate from score precision.

Reproduce the37-case author suite in the accepted interpreter. It lacks pytest, so this command appends the already installed lab pytest harness **after** economic site-packages, disables plugin autoload and installs nothing. Runtime-identity assertions still use the exact accepted packages/interpreter.

```bash
cd /home/hayden/projects/ballotmancy-improvement-lab/research/improvement-loop-20261024/momentum-forecast-scoring-sol-v1
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/hayden/projects/ballotmancy-improvement-lab/.improvement-ledger/runtime/economic-v1-env/bin/python -c \
'import sys; sys.path.append("/home/hayden/projects/ballotmancy-improvement-lab/.venv/lib/python3.12/site-packages"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","test_score_math.py","test_score_reader.py"]))'
```

`verify_seal.py` performs read-only artifact/upstream byte verification without opening actual predictions or outcomes. Actual selection/mask pointers in the template are Root-supplied metadata references; author tests never load the actual selected jobs, reference arrays, masks or target outcomes.
