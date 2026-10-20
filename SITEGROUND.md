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
preview, the model-stack controls, a state map, 35 Senate contests, and
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
comparison state, and forecast/observed map view, including Back/Forward support.

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
the leading candidate's party and win probability: **Safe** at 98% or above,
**Likely** from 65% to below 98%, and **Tossup** below 65%. These categories do
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
