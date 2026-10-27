# A2018 observation ledger V2: factor closure checks

This separately versioned aggregate successor preserves the V1 packet and its source pins. It adds fail-closed checks for extension data and likelihood paths before reporting the source-algebraic conditional-prior status of eventG.

## Output and privacy boundary

The only ledger output is aggregate-ledger.json. It has source, block, observation, event and contest metadata plus aggregate counts. Survey record tokens are read only into local memory for count and source-scoped uniqueness checks; the builder does not serialize, print, or log them. The 12 tests write into Python TemporaryDirectory paths and scan the output for each in-memory survey token. There is no row-membership CSV in V2. build_ledger.py requires an explicit output directory. No model or prediction job ran, and no posterior or prediction arrays were read.

The builder pins the A2018 plan, context, prepared problem, selected 117-components.py snapshot, ranked/runoff and open-person contributions, external score receipt, and V1 ledger/source pins. It verifies that selected component bytes match both the plan and context source inventory and that rank/open contribution bytes match the context inventory. V1 source files remain unchanged.

## Aggregate reconciliation

The V2 aggregate projection exactly matches the V1 ledger on frozen inputs; the projection SHA is recorded in aggregate-ledger.json. Counts remain 54 events, 37 contests, 65 survey blocks and 7,968 survey record memberships, one 274-row poll covariance block, zero official count observations, and zero participation reports. The Arizona 2018 contest has 20 ordinary paired-poll rows and no target survey, count, or report. All eight protected parent masses remain 0.125.

The 7,968 number counts survey-record memberships, not necessarily unique people or independent likelihood factors. In the model, these group into 65 survey Potentials. The 274 poll rows group into one covariance likelihood. V2 preserves the source-scoped identifier uniqueness check but exports no survey record tokens.

## Event-scale closure gates

Before returning a conditional eventG prior status, the builder requires all of these conditions:

- open_person is null.
- Ranked extension configurations may exist, but every ranked observation list is empty, every runoff observation list is empty, core ranked observations are empty, and the ranked poll-channel map is empty.
- All 274 poll rows therefore take the ordinary paired-logit branch.
- Official count vectors and participation reports are both empty.
- The pinned survey likelihood loop contains no eventG or event_intensity reference.
- The pinned ordinary poll mean is the log difference of the two category rates. The event multiplier and one scalar candidate-response probability per contest are common factors, so they cancel algebraically in that ratio.
- The exact ranked/runoff and open-person contribution sources are hash-pinned and each creates count factors only from observation lists.

For this exact pinned input/source closure, the ledger reports that all 54 eventG coordinates retain their Gamma(kappa,kappa) conditional law in original coordinates given shared kappa and other parameters. This is a source-algebra argument, not a compiled-graph or posterior-array test. It does not claim marginal independence; shared kappa can induce marginal dependence. It also does not certify floating-point cancellation in a compiled graph. A new count/report, open-person observation, ranked/runoff observation, ranked poll channel, survey eventG dependence, source drift, or changed poll formula makes the builder fail closed.

The prior-only status applies to eventG. Survey likelihoods can still inform turnout propensity and profile composition. The selected 2018 plan remains unqualified, incomplete, and unauthorized for publication.

## Reproduction

Run the 12 bounded source/fixture tests:

    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B -m unittest test_ledger -v

Build into an explicitly selected scratch directory:

    PYTHONDONTWRITEBYTECODE=1 nice -n 15 python3 -B build_ledger.py --output-dir /tmp/a2018-ledger-check

The fixture tests add official counts/reports, open-person observations, ranked/runoff/core-ranked rows, and ranked poll channels; each makes the closure check reject. Source fixtures adding eventG to the survey path or changing the ordinary poll equation are also rejected. Tests verify V1 aggregate equality, no row-token export, and pin-drift failure. They do not execute or compile the model.
