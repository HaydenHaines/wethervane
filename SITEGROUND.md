# Ballotmancy on SiteGround

The `deploy/siteground` branch adds the deployable Senate visualization at the
repository root. The earlier Python application and `web/` directory remain in
the repository, but this deployment builds the new root application.

## SiteGround configuration

Use GrowBig's **Websites → Node.js Projects → New Project** flow and import
`HaydenHaines/wethervane` from GitHub.

| Setting | Value |
| --- | --- |
| Branch | `deploy/siteground` |
| Framework | Next.js |
| Application/root directory, if shown | Repository root (`.`), not `web` |
| Node.js | 22 |
| Package manager | npm |
| Install command, if shown | `npm ci` |
| Build command | `npm run build` |
| Build output directory | `.next` (Next.js preset) |
| Start command, if shown | `npm start` |
| Required secrets or environment variables | None |

Keep the platform-provided `PORT` value. The start script listens on all
interfaces and uses Next.js's standard `PORT` behavior.

The app reads `public/data/explorer/registry.json` and loads the selected
checked-in JSON bundle. It does not call the old localhost API or run Python
during deployment. No database, secrets, or model fit is required on SiteGround.

Check the temporary URL first. The home page should show the dated research
preview, the collapsed model-stack controls, a state map, 35 Senate contests, and
three caucus scenarios. The year controls switch between 2026, 2022, and 2018
on the same page. `/api/health` returns a JSON health response with the snapshot date and
race count. An unfamiliar temporary URL is not access control; use SiteGround
Protected URLs if a private preview is wanted.

Once verified, assign `ballotmancy.com` as the primary domain, provision SSL and
enable HTTPS. Configure `ballotmancer.com` and its `www` form to redirect
permanently to `https://ballotmancy.com`, preserving paths. Redirect source
domains also need working DNS and certificates for HTTPS requests. Preserve
existing email DNS records.

SiteGround automatically deploys pushes to the selected branch. Update only
this release branch when a new web release is ready. SiteGround's auto-deploy
does not necessarily wait for GitHub Actions; run the release checks before
pushing, or pause auto-deployment when preparing a change.

## Local verification

Use Node.js 22 and an activated Python virtual environment for the offline tests.

```sh
npm ci
python3 -m pip install -r scripts/requirements-site.txt
npm run test:snapshot
npm run test:tipping
npm run test:explorer-data
npm run test:explorer-state
npm run test:composition
npm run test:scoreboard
npm run test:ribbon-layout
npm run catalog:build
npm run test:explorer-artifacts
npm run build
npm run typecheck
npx playwright install chromium
npm run test:e2e
npm start -- --port 3127
```

The browser tests exercise the production build. The application uses local
fonts and geography; no external map token is needed.

## Data and scope

The initial snapshot contains the October 20, 2026 national Senate polling and
fundamentals forecasts. It preserves named candidate parties, conditional
principal-contender share intervals, and the distinction between party seats
and scenario-specific caucus control. New independent winners are unassigned
in the default control scenario. Third-candidate wins are outside the model.

These are aggregate research forecasts, not the unfinished community/count
model and not live election results. Numerical checks do not establish
calibration or forecast improvement. The page exposes its cutoff and
limitations. Public data downloads include an export schema and source hashes.

The exporter is `scripts/export_site_snapshot.py`; use its `--help` for the
source arguments. Export from the frozen model artifacts, review the changes
and hashes, rerun the release checks, and commit both JSON data files together.
Never replace research artifacts or rerun model fits as a website build step.

## Single-page model builder

The homepage defaults to the 2026 standard polling stack. Year, supported
forecast date, base, polling, and uncertainty controls select an existing
complete run; they never interpolate incompatible artifacts or fit a model
in the browser. Changing a stack updates the ribbon, map, race details,
chamber probabilities, and historical scores together. A failed selection
leaves the previous stack visible. The URL records run, caucus scenario, race,
comparison state and chosen reference, and forecast/observed map view, including Back/Forward support.

All charts are shared components driven by numerical result data, not stored
images or hand-authored graphs for each model. The margin ribbon derives its
race order and projected control markers from expected shares, eligible caucus
candidates, retained/guaranteed seats, and control thresholds. Layout widths and
labels follow the supplied race count and year. Simulation-based tipping
frequencies remain joint-result statistics: they cannot be recovered from
marginal race leans alone. Seat charts render the supplied probability mass
functions and actual totals, with comparison overlays from the default result.

The initial catalog contains 38 configurations:

- **2026:** fundamentals, standard polling, lower/higher polling error, and
  lower/higher cycle error (six saved runs, October 20 evidence).
- **2018 and 2022:** W, F, FP, and FP persistent-error arms at common-shock
  loadings 0, 0.5, 1, and 2 (16 configurations per year). Only the complete
  35-contest forecast at 14 days before election day is supported.

Historical winner probabilities and the 2026 principal-contender share model
use different model laws. The historical reconstructions include retrospective
source/roster proxies; they do not establish strict as-of performance or
validate the current model. The unfinished community/count model is not exposed
as a selectable component.

Historical bundles preserve categorical group outcomes. For example, California
2018's Democratic group contains two candidates; its group probability is not
assigned to either candidate individually. Winner-only artifacts do not contain
candidate vote shares or margins, so their ribbon orders outcome confidence
and does not invent margin-based tipping points. Same-state regular and special
contests remain independently selectable. Actual-map colors use the selected
contest's observed winner group while retaining original party labels.

Historical chamber distributions come from paired D/R/unaligned seat draws,
with the year's retained seats, vice-presidential tie-break, and caucus scenario.
CRPS, categorical control Brier scores, and interval coverage are checked against
fixed observed chamber results. Comparison deltas use the same year, contest,
outcome, and caucus scenario. Two historical elections are a limited descriptive
comparison, not a calibration claim.

To refresh from an existing frozen research checkout (offline):

```sh
python3 scripts/export_current_explorer.py --source-root /path/to/research-worktree
python3 scripts/export_history_explorer.py --source-root /path/to/research-worktree
npm run catalog:build
```

Review the exported JSON and source hashes and run all release checks above.
Commit both indexes, their bundles, and `registry.json` together. Index entries
must exactly match their bundle's run metadata and SHA-256. The catalog builder
rejects duplicate configurations, unsafe paths, missing defaults, or unsupported
historical timing/margin claims. Browser tests exercise shared state, candidate
groups, historical truth, URLs, comparison, failures, and desktop/mobile layout.
Never rerun research fits or modify research artifacts as a website build step.

## Tipping-point ribbon

The ribbon orders all 35 contests by expected first-contender vote margin,
from Democratic/independent advantage to Republican advantage. Colors show
the leading candidate's party and win probability: **Safe** above 90%,
**Likely** above 60% through 90%, and **Tossup** at 60% or below.
For two-outcome races this is the user-selected 40–60% Tossup band.
These October 20 composition-UI cutoffs replace the earlier 65%/98% boundaries;
raw probabilities are never rounded before classification. These categories do
not determine the ribbon order.

The endcaps show 65 retained seats: 34 in the Democratic caucus (including
King and Sanders) and 31 in the Republican caucus. The saved forecast assigns
the vice-presidential tie-break to Republicans, so the thresholds are 51
Democratic caucus seats and 50 Republican caucus seats.

For each caucus and alignment scenario, we first count retained seats and
contests where either winner belongs to that caucus. We then rank the
remaining eligible contests by that caucus's vote share. The contest that
supplies the last seat needed for control is its tipping point. The ribbon
marker uses expected shares. Simulation percentages repeat this ranking
within each original joint draw, preserving the model's correlations.
Equal margins at the threshold share credit equally.

This is an unconditional ranking statistic: a contest can occupy the
threshold even in a simulation where its caucus falls short. It is not the
probability that changing that contest alone changes the actual chamber
winner. Each caucus's tipping frequencies sum to 100%; the two caucuses'
frequencies should not be added together.

Unaligned independents can produce different Democratic and Republican
thresholds. When opposition independents caucus with Republicans, all four
independent-versus-Republican contests count toward Republican control
regardless of their winner. Those contests remain visible in the ribbon,
with their candidates' actual party labels, but cannot be tipping contests
in that scenario.

The separate `public/data/tipping-points.json` is derived offline from the
same frozen forecasts and saved joint draws. Its provenance records the
input hashes. The website never runs simulations or requires Python.

To regenerate the summaries from an existing research checkout:

```sh
python3 scripts/export_tipping_points.py --source-root /path/to/frozen-research-worktree
```

Commit `tipping-points.json` and `tipping-points-provenance.json` together.
When refreshing the forecast snapshot, regenerate both exporters from the
same source artifacts. These legacy endpoints remain available for reproducibility. The current
interface loads the forecast and tipping statistics together from one
SHA-256-checked explorer bundle. If a new selection fails to load or verify,
the previous complete stack stays visible with an error message.

The preserved state outlines come from the existing project's
`web/public/states-us.geojson`. No county or tract forecast is implied by the
state-level display.

SiteGround references:

- [Node.js projects](https://www.siteground.com/kb/what-is-node-js-project)
- [Project settings](https://www.siteground.com/kb/manage-node-js-project)
- [Custom domain](https://www.siteground.com/kb/change-domain-node-js-project)


## Component composition and historical scorecard (October 20, 2026)

“Build your perspective” starts collapsed. The default is the preserved current
best-guess forecast; direct URLs and locally restored views retain their saved
selection. Opening the panel reveals catalog-driven component selectors and the
preserved basic electoral-history and current best-guess shortcuts.

`public/data/explorer/capabilities.json` pins 38 saved configurations to their
exact result SHA256 and source catalog. `POST /api/configurations/resolve`
accepts selection intent only and validates ready bundle bytes and scientific
metadata on the Node server. The three response states are `ready`,
`requires_computation`, and `incompatible`. Unsupported selections never silently
change other components. While an unresolved choice is inspected, the previous
forecast remains explicitly labeled as the displayed result.

The four experimental aggregate bases expose metadata-only computation
requirements, not runnable national forecasts. The source registry supports
ordinary D/R aggregate specifications; authenticated bindings, sampled forecasts,
and a custom chamber adapter are still missing for these selections. There is no
fit executor, queue, or Python service on SiteGround. Fine component switches may
require a joint refit. Whole-forecast blending remains a separate, unavailable
feature requiring explicit weights and coherent joint uncertainty. Numerical fit
evidence, predictive testing evidence, experimental status, and result
availability are presented separately.

Regenerate the capability manifest offline with
`scripts/export_composition_capabilities.py --source-root /path/to/research-checkout`.
It verifies frozen source evidence and preserved forecast fingerprints. Never run
model fitting as a site build step. Current structural and polling cutoffs remain
separately visible (15:21:23Z and 16:21:44Z on October 20, 2026).

Named configurations are stored only in the visitor's browser (up to 30), pinned
to configuration ID, run ID, and result hash. Invalid or stale stored identities
are rejected rather than rebound. Share URLs include the displayed saved run and
view settings; they do not promise that an unresolved research selection exists.

Historical years put a four-part scorecard directly under the controls: unique
favorites that won, expected versus observed seats with 90% marginal intervals,
probability assigned to observed chamber control, and raw multiclass Brier error
relative to a matched reference. Brier is summed across outcomes and averaged
over races (0–2, lower is better). Exact ties, missing observations, grouped
outcomes, and incomplete coverage remain explicit. Matched historical comparisons
require the same cutoff, horizon, target, law, race/outcome membership, organizing
scenario, and observed truth. A winning favorite count is not a calibration test.

The scorecard design follows proper-scoring guidance from
[Metaculus](https://www.metaculus.com/help/scores-faq/), uncertainty guidance from
[ONS](https://service-manual.ons.gov.uk/data-visualisation/guidance/showing-uncertainty-in-charts),
and the multi-metric verification approach described by
[ECMWF](https://confluence.ecmwf.int/spaces/WLS/pages/394992254/Verification).
Historical reconstruction scores are descriptive; they do not validate the new
experimental bases or imply strict as-of testing.
