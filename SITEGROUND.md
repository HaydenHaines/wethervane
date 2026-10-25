# Ballotmancy on SiteGround

The `deploy/siteground` branch contains the deployable congressional visualization
at the repository root, including Senate forecasts and historical House research.
The earlier Python application and `web/` directory remain in
the repository, but this deployment builds the new root application.

The [historical House polling release](docs/development/frontend-house-polling-release-20261025.md)
records the October 25 House collection and validation. The dated House planning
notes later in this guide describe its earlier preparation. Current 2026 House
integration and the population/community/count research stack remain separate
workstreams; availability comes from reviewed publication artifacts.

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

The checked-in [Senate registry](public/data/explorer/registry.json) contains 48
configurations at this documentation update:

- **2026:** sixteen saved runs: the preserved history and standard-polling
  presets, four polling-uncertainty sensitivity runs, and five alternative
  base models each with and without polling. Each result records its own cutoffs.
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
exact result SHA256 and source catalog. `GET /api/configurations/catalog` serves
that public metadata through Node because SiteGround blocks the static
`capabilities.json` URL. `POST /api/configurations/resolve`
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

## Cross-session coordination

The user has assigned complementary roles and authorized direct collaboration:

| Responsibility | Local session | Worktree |
| --- | --- | --- |
| UI/UX, frontend adapters, validation and deployment | `01a15a54-c818-75b0-86f3-8797c3d1cfa2` | `/home/hayden/projects/ballotmancy-siteground` |
| Ongoing coordination, frontend stewardship and model handoffs | `01a11695-7d58-7eb2-b2ce-e9c6c14fc007` | `/home/hayden/projects/wethervane-redesign-20261006/integration` |
| Runtime/backend selection and transferred Texas training | `01a173a3-fc75-7233-af2c-69be726131fc` | Shared integration research directory; isolated GPU environment |
| House modeling framework and benchmarks | `01a15ae1-4e5d-7711-bec6-113673b68a1d` | `/home/hayden/projects/ballotmancy-house-benchmark` |

Share interface changes, frozen outputs, validation evidence, blockers and
concrete development requests directly. Keep edits within each worktree's owned
scope; coordinate changes to shared scientific contracts before integration.
The frontend can read both sessions and relay Senate handoffs if that session's
outgoing messaging tool remains blocked by its approval configuration. House
has acknowledged direct communication and is contacting Senate about methodology.

### Historical House planning notes — October 20, 2026

Chamber navigation and historical House results have since been implemented.
The following paragraphs preserve the original planning context; the October 25
release note above governs the historical collection's available features.
Current House development is owned by the frontend session in
`/home/hayden/projects/ballotmancy-house-live-ui-20261025`.

User requirement recorded October 20, 2026: once the House forecast is live, add
a chamber selector at the top of this same page. The proposed UI is two tabs,
**Senate | House**. Each chamber gets its own map and a snake/tipping-point chart,
with chamber-control and historical performance views driven by that chamber's
results. Preserve the single-page design and the default-collapsed model controls.
Charts must derive from race arrays and control metadata rather than requiring
separate visual assets for every configuration. This is a future requirement,
not an implemented House forecast or an instruction to expand the current
benchmark scope.

The House session's first output is an ordinary D/R historical race subset,
with a 435-seat coverage ledger and explicit exclusions. Its proposed metadata
includes `chamber=house`, `chamber_size=435`, `majority_threshold=218`,
`scope=eligible_race_subset`, and `full_chamber=false`. Preserve map identity as
election cycle, published map vintage and district key. Geometry must match the
forecast's districts. A race subset and common cycle residual draws do not by
themselves establish a complete House control distribution or tipping statistic.

Before activating the House view, agree and freeze the result schema, coverage,
map linkage, output availability, exact configuration/result identities, cutoffs,
training/test cycles, observed outcomes and joint-draw interpretation. The UI will
need chamber-aware validation in place of the current 100-seat assumption and a
district geography adapter. Availability of share margins, true tipping
frequencies and complete chamber forecasts must be stated explicitly; do not
infer them from marginal win probabilities. UI/type/adapter development can
begin from a reviewed schema/example while model execution continues.

House initially scores binary Brier on a 0–1 scale. For two outcomes, the site's
raw categorical Brier (sum over outcomes, then mean over races) is twice that
binary score on a 0–2 scale. Keep both definitions explicit and compare only
matched coverage/support, dates and observed truth. Retrospective proxy sources
remain labeled as such.

### House frontend preparation (2026-10-20)

The root page now owns a Senate/House tab selector. Senate remains the default,
including existing `?run=...` links. House uses `?chamber=house`; switching chambers
preserves each chamber's selection within the current page session, and browser
Back/Forward restores the appropriate chamber. No second page or modeling service
is introduced. The public House view is **in preparation**, with no numerical
forecast, synthetic district map, or implied release date.

The preparation is executable beyond that empty state:

- `components/HouseExplorer.tsx` consumes a normalized publication response from
  `GET /api/chambers/house`; the existing Senate artifact format stays unchanged.
- `components/HouseRibbon.tsx` and `lib/house-ribbon.ts` derive a serpentine grid
  from race IDs, district keys, and Democratic win probabilities. Width determines
  the number of columns; fields above 60 races start in a compact overview, with
  an option for larger labeled tiles. Tests exercise 1, 16, and 435 races. No per-configuration
  graphics or inferred vote margins are used.
- `components/DistrictMap.tsx` accepts injected GeoJSON and race results. Its join
  requires exact election cycle, map vintage, and `STATE-XX` district key; at-large
  is explicitly `00`. It supports state focus and keyboard selection. Unmodeled
  districts remain gray; missing or mismatched geography never becomes a neutral
  race forecast. Publication admission and a geometry source fingerprint are
  separate from the publisher's map-vintage label.
- `lib/chamber.ts` describes chamber size and majority separately from race count.
  Its readiness checks distinguish race display, map display, control, and tipping.
  A subset never enables control or tipping, even with 435 ledger rows or shared
  joint draws. Full House control requires 435 distinct modeled seats, no retained
  seats, a complete seat ledger, aligned configuration/cycle/joint-artifact identity,
  and consistent 436-element D/R/U seat PMFs. Tipping additionally requires the
  saved frequency vector bound to that same joint artifact. The UI currently
  recognizes only `caucus_rank_threshold_v1`; unknown statistics remain unavailable.
- A qualified full House result can show the 218th position in the probability
  ordering. That position is labeled separately from simulated threshold-rank
  frequencies and is not described as a causal pivot. House uses 218 for both
  parties; Senate vice-presidential tie assumptions are not reused.
- Historical House results show favorites correct, raw categorical Brier (0–2),
  and tested coverage. Exact favorite ties are excluded from the favorites tally;
  Brier uses every available observed race. Subset metrics are never chamber totals.
  Experimental status, testing evidence, and publication availability are separate.

#### Publication handoff; proposed adapter, not a research exporter claim

The House research session's draft is
`research/house-benchmark-20261020/frontend-contract-draft.json` in its separate
benchmark worktree. It describes eligible regular D/R candidate contests,
retrospective source proxies, and configuration-local joint draw IDs. It does not
provide a current House forecast, reviewed full-chamber control, or admitted legal
geometry. This frontend does not modify or run that research pipeline.

The deployment-owned normalized contract is `HouseResult` in `lib/house-result.ts`.
It is deliberately limited to ordinary D/R candidate outcomes and one selected
published configuration. A future research adapter must explicitly convert the
reviewed outputs to it; it must not cast the draft JSON into an available result.
A multi-configuration House catalog and component resolver will be integrated when
reviewed results and configuration capabilities exist. No dummy selectors are
presented as executable model choices.

To activate a real publication:

1. Produce and review a frozen normalized result with `reviewStatus: "reviewed"`,
   run/configuration IDs, a configuration SHA256, the election cycle/map vintage,
   explicit scope, and a complete 435-row coverage ledger. Each excluded district
   requires a reason; every eligible district requires a valid result. Metadata
   configuration ID must equal the run ID's configuration. Candidate membership
   and observed winner identity must agree. Optional two-party share distributions
   use the exact `HOUSE_SHARE_DENOMINATOR` constant; paired means and quantiles
   must complement each other. Retrospective inputs stay labeled as such.
2. Place the result JSON in `data/house/` and calculate its SHA256 over exact bytes.
   If geography is ready, independently admit a matching election-map GeoJSON
   FeatureCollection with `properties.districtKey`, then hash those source bytes.
   The map-vintage name itself does not certify boundary correctness.
3. Update `data/house/publication.json` from `status: "preparing"` to an admitted
   manifest. Its shape is:

   ```json
   {
     "schemaVersion": 1,
     "status": "available",
     "result": {"file": "reviewed-result.json", "sha256": "<64 lowercase hex>"},
     "geography": {
       "file": "verified-districts.json",
       "sha256": "<64 lowercase hex>",
       "metadata": {
         "cycle": 2022,
         "mapVintage": "house_election_map_2022",
         "admission": "verified"
       }
     }
   }
   ```

   Geography may be omitted for a reviewed race-only publication; the map then
   explicitly stays unavailable. A supplied map must pass all identity/coverage
   checks. Each artifact must be a direct JSON file inside `data/house/`; traversal
   and symlinks outside that directory are rejected. Full-chamber declarations
   additionally must pass the complete control proof. These are structural and
   byte-integrity checks, not scientific certification of the underlying model.
4. Run `npm run test:house`, `npm run build`, `npm run typecheck`, and the browser
   suite. Review actual geography, results, and their provenance before pushing
   `deploy/siteground`. The Node route avoids SiteGround's static JSON filename
   restrictions, as the component catalog already does.

All fixture probabilities and rectangles live under `tests/fixtures/house.mjs`.
They are software-test data only and are neither referenced by the public
publication manifest nor served by a preview route. This records the initial
fixture-only preparation. Historical House results were subsequently integrated;
see the October 25 release note above. No new environment variables, hosting
services, or user credentials were needed for that preparation.

### Race navigation and chamber color refinements

Desktop race-detail panels are capped to the viewport and scroll independently;
selecting a different race returns the panel to its top. On mobile, details remain
part of normal page scrolling. Senate and House race lists show the first ten
sorted/filter-matched entries by default, with an accessible See all / Show fewer
control. Search always considers the complete race set.

The chamber heading and active tab use the selected result's unconditioned chamber
control probabilities: Democratic blue or Republican red above 60%, purple when
neither party nor unresolved alignment exceeds 60%, and gray for a favored
unresolved alignment. Missing control data stays neutral, including House
preparation and eligible-race subsets. Colors follow the displayed saved result
and caucus scenario, never an uncomputed component request or observed winners.
