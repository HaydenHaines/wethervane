# Exact native observation method: supplied-state parity

The owner-recommended narrow check passes: the exact current native
`Measurement.observations` method agrees with the existing M01 one-component
mixture at three invented fixed count-rate states. Maximum absolute log-density
difference is 2.85e-14. Candidate order differs across contests and one pair is
reversed; multiplying both contests' rates by ten leaves the log likelihood
unchanged. This establishes a small numerical/interface bridge, not a fitted
forecast or complete native integration.

`check.py` parses the owner file and compiles the unmodified method AST, then
executes only its poll branch inside a temporary in-memory PyMC model with no
free parameters. It checks the exact ordered three-token factor receipt and
compares the native correlated `MvNormal` to `SharedVectorMixture` with one
component. It does not import/install a provider or execute a sampler. The
source hashes and precise scope are in `result.json`. The owner tree was read
only. Existing math-prototype evidence already covers constructed multi-poll
joint-mixture versus independent-marginal behavior; it was not duplicated.

The initial lab `.venv` import failed before this check executed because its
ArviZ lacks `InferenceData`. No dependencies were changed. The successful run
used the already established `.improvement-ledger/runtime/calibration-v1-env`
Python, one BLAS/OMP thread, bytecode disabled, and PyTensor `cxx=,floatX=float64`.
The initial failure is an environment failure, not a scientific rejection.

Owner coordination: `m01-count-binding-followup-v2.md` says this session has no
M01 binding or fit and explicitly recommends this narrow check. Runtime owner
Nevada/national work remains separate; synthetic predictive-bank validation
does not admit a joint M01 fit. No overlap with their running jobs is claimed
from process inspection; the source-bound written coordination is the basis.

The highest-value next step is a separately versioned, owner-coordinated native
measurement challenger with a frozen complete-poll-vector and evidence ledger,
followed by full-compiler fixed-state parity before any empirical fit. The
existing compiler requires every numerical factor to claim admitted evidence
once, but its set of `poll:observation_id` tokens cannot by itself detect the
same respondents or election labels under different identifiers. A provenance
dependency audit must therefore bind calibration training labels, polls,
surveys, fundamentals and official counts across modules. Replacing each poll
block by an independent mixture would change the shared-vector target whenever
blocks share calibration uncertainty; first specify the intended complete
vector and any justified conditional independence boundaries.

Before historical refits, require earlier-only calibration banks separately for
2022/2018 (A) and 2024/2020 (B), cutoff publication/vintage evidence, common race
and horizon coverage, and identical information/tuning opportunities across
arms. Never transplant the current TX2026 six-poll bank into these replays.
Preserve `population_cut_v1` original parent masses and each normalized
downstream conditional. Marginal calibration weights can update inside that
conditional; they are not automatically a protected calibration cut.

For the later matched experiment, vary only fixed-covariance incumbent versus
the declared marginal-mixture challenger and jointly refit affected downstream
components. Retain fundamentals-only, polls-only and combined references where
supported. Save proper count/share and win scores, coverage with width, useful
precision, cycle-wise A/B results, supported joint seat outcomes, diagnostics
and numerical comparison uncertainty. The present check determines no useful
effect threshold, runtime qualification, empirical winner or weakest layer.
No fit, registration, promotion or publication is authorized by this artifact.

Applied project skills: `.agents/skills/verify-election-model/SKILL.md` and
`.agents/skills/refine-model-layer/SKILL.md`; their distinction between
computational evidence, dated replay and predictive usefulness governs this
report. This work adds a narrow executable check rather than repeating the
existing M01 specification or prototype.
