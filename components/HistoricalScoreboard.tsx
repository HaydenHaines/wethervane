import { forecastLabel } from "../lib/forecast-copy";
import { HISTORICAL_CONTEXT } from "../lib/forecast-copy";
import type { ExplorerBundle, ExplorerScenario } from "../lib/explorer-types";
import { deriveScoreboard, compareScoreboards } from "../lib/scoreboard";
import {
  COLORS,
  STATE_NAMES,
  dateLabel,
  probability,
  ppChange,
} from "../lib/presentation";

type Props = {
  bundle: ExplorerBundle;
  scenario: ExplorerScenario;
  baselineBundle?: ExplorerBundle;
  baselineScenario?: ExplorerScenario;
  onSelectRace: (raceId: string) => void;
};

const caucusName = (party: string) =>
  party === "U" ? "Unaligned" : `${party} caucus`;
const controlName = (party: string) =>
  party === "U" ? "Unresolved control" : `${party} control`;

export default function HistoricalScoreboard({
  bundle,
  scenario,
  baselineBundle,
  baselineScenario,
  onSelectRace,
}: Props) {
  if (!bundle.run.historical) return null;
  const score = deriveScoreboard(bundle, scenario);
  const reference =
    baselineBundle && baselineScenario
      ? deriveScoreboard(baselineBundle, baselineScenario)
      : null;
  const comparison =
    baselineBundle && baselineScenario
      ? compareScoreboards(bundle, scenario, baselineBundle, baselineScenario)
      : null;
  const matched = comparison?.status === "matched";
  const favorites = score.favorites;
  const control = score.control;
  const difference = matched ? comparison.brierDifference : null;
  const relativeError =
    difference !== null && reference?.brier && reference.brier.mean > 0
      ? Math.abs(difference / reference.brier.mean) * 100
      : null;
  const sameError = difference !== null && Math.abs(difference) < 1e-12;
  const controlOptions = [
    {
      id: "D",
      label: "D control",
      p: scenario.control.democraticMajorityProbability,
      color: COLORS.D,
    },
    {
      id: "R",
      label: "R control",
      p: scenario.control.republicanMajorityProbability,
      color: COLORS.R,
    },
    {
      id: "U",
      label: "Unresolved",
      p: scenario.control.unresolvedOrganizingProbability,
      color: COLORS.unresolved,
    },
  ];

  return (
    <section
      className="historical-scoreboard"
      aria-labelledby="scoreboard-title"
    >
      <div className="scoreboard-heading">
        <div>
          <div className="eyebrow">
            {dateLabel(bundle.run.forecastAt)} · Historical reconstruction
            {bundle.run.horizonDays === null
              ? ""
              : ` · ${bundle.run.horizonDays} days before the election`}
          </div>
          <h2 className="section-title" id="scoreboard-title">
            How this model did
          </h2>
          <p>
            {forecastLabel(bundle.run.label)} · {forecastLabel(scenario.label)}
            {baselineBundle ? ` · Compared with ${forecastLabel(baselineBundle.run.label)}` : ""}
          </p>
        </div>
        <a href="#historical-performance">Detailed scores & seat probabilities ↓</a>
      </div>
      <p className="historical-context">{HISTORICAL_CONTEXT}</p>
      {score.status === "unavailable" ? (
        <p className="scoreboard-unavailable">
          {score.reason || "Actual results are unavailable for this setup."}
        </p>
      ) : (
        <div className="scoreboard-cards">
          <article
            className="scoreboard-card"
            aria-labelledby="favorite-score-title"
          >
            <h3 id="favorite-score-title">Favorites that won</h3>
            {favorites ? (
              <>
                <div className="scoreboard-value">
                  {favorites.correct}
                  <span> / {favorites.decisive}</span>
                </div>
                <p className="scoreboard-caption">
                  Races where the sole most likely candidate or group won
                </p>
                {favorites.decisive > 0 && (
                  <p className="scoreboard-context">
                    Its probabilities implied about{" "}
                    <strong>{favorites.expectedCorrect.toFixed(1)}</strong> of
                    these favorites winning.
                  </p>
                )}
                {favorites.tiedRaceIds.length > 0 && (
                  <p className="scoreboard-context">
                    {favorites.tiedRaceIds.length} tied{" "}
                    {favorites.tiedRaceIds.length === 1
                      ? "favorite excluded"
                      : "favorites excluded"}
                    .
                  </p>
                )}
                {matched && reference?.favorites && (
                  <p className="scoreboard-comparison">
                    Comparison: {reference.favorites.correct} /{" "}
                    {reference.favorites.decisive} favorites won.
                  </p>
                )}
                {favorites.missRaceIds.length > 0 ? (
                  <details className="scoreboard-misses">
                    <summary>
                      Explore {favorites.missRaceIds.length}{" "}
                      {favorites.missRaceIds.length === 1 ? "miss" : "misses"}
                    </summary>
                    <div>
                      {favorites.missRaceIds.map((id) => {
                        const race = bundle.races.find((r) => r.id === id);
                        return (
                          <button key={id} onClick={() => onSelectRace(id)}>
                            {race
                              ? `${STATE_NAMES[race.state] || race.state}${/special/i.test(race.electionLabel) ? " special" : ""}`
                              : id}
                          </button>
                        );
                      })}
                    </div>
                  </details>
                ) : (
                  <p className="scoreboard-context">
                    {favorites.decisive > 0
                      ? "All favorites without a tie in their win chances won."
                      : "No sole favorites to evaluate; tied races still count in the probability score."}
                  </p>
                )}
              </>
            ) : (
              <p className="scoreboard-context">
                Favorite scores are unavailable.
              </p>
            )}
          </article>

          <article
            className="scoreboard-card"
            aria-labelledby="seat-score-title"
          >
            <h3 id="seat-score-title">Predicted vs actual seats</h3>
            <div className="scoreboard-seat-head">
              <span>Party group</span>
              <span>Forecast median & 90% range</span>
              <span>Actual</span>
            </div>
            <div className="scoreboard-seat-rows">
              {score.seats.map((seat) => (
                <div className="scoreboard-seat-row" key={seat.party}>
                  <span title={caucusName(seat.party)}>{seat.party}</span>
                  <div>
                    <SeatRange
                      party={seat.party}
                      median={seat.median}
                      lower={seat.p05}
                      upper={seat.p95}
                      observed={seat.observed}
                    />
                    <span className="scoreboard-seat-numbers">
                      {seat.median}{" "}
                      <span>
                        ({seat.p05}–{seat.p95})
                      </span>
                    </span>
                  </div>
                  <strong>
                    {seat.observed === null ? "—" : seat.observed}
                  </strong>
                </div>
              ))}
            </div>
            {score.seats.length === 0 && (
              <p className="scoreboard-context">
                Seat estimates are unavailable.
              </p>
            )}
            <div className="scoreboard-range-key">
              <span>
                <i className="forecast-dot" />
                Forecast median
              </span>
              <span>U = unaligned</span>
              <span>
                <i className="actual-line" />
                Actual seats
              </span>
            </div>
            <p className="scoreboard-context">The median is the middle simulated seat total. The range spans the middle 90% of totals. Party groups include allied independents; U means unaligned.</p>
            {matched && reference && (
              <p className="scoreboard-comparison">
                Expected-seat change vs comparison:{" "}
                {score.seats
                  .map((seat) => {
                    const previous = reference.seats.find(
                      (s) => s.party === seat.party,
                    );
                    const delta = previous
                      ? seat.expected - previous.expected
                      : null;
                    return delta === null
                      ? null
                      : `${seat.party} ${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(1)}`;
                  })
                  .filter(Boolean)
                  .join(" · ")}
                .
              </p>
            )}
          </article>

          <article
            className="scoreboard-card"
            aria-labelledby="control-score-title"
          >
            <h3 id="control-score-title">Chamber control</h3>
            {control ? (
              <>
                <div className="scoreboard-value">
                  {probability(control.observedProbability)}
                </div>
                <p className="scoreboard-caption">
                  Chance assigned to the actual outcome:{" "}
                  <strong>{controlName(control.observed)}</strong>
                </p>
                <div className="scoreboard-control-options">
                  {controlOptions.map((option) => (
                    <div
                      className={
                        option.id === control.observed ? "observed" : ""
                      }
                      key={option.id}
                    >
                      <span>
                        <i style={{ background: option.color }} />
                        {forecastLabel(option.label)}
                        {option.id === control.observed ? " · Actual" : ""}
                      </span>
                      <strong>{probability(option.p)}</strong>
                    </div>
                  ))}
                </div>
                {matched && reference?.control && (
                  <p className="scoreboard-comparison">
                    {ppChange(
                      control.observedProbability -
                        reference.control.observedProbability,
                    )}{" "}
                    vs comparison for the actual outcome.
                  </p>
                )}
              </>
            ) : (
              <p className="scoreboard-context">
                Observed chamber control is unavailable.
              </p>
            )}
          </article>

          <article
            className="scoreboard-card probability-quality"
            aria-labelledby="probability-score-title"
          >
            <h3 id="probability-score-title">Probability quality</h3>
            {score.brier ? (
              <>
                <div className="scoreboard-value">{score.brier.mean.toFixed(3)}</div>
                <p className="scoreboard-caption">Average probability error (Brier score). Lower is better; 0 is perfect on this 0–2 scale.</p>
                <p className="scoreboard-context">Scores every outcome’s chance against what happened. Confident wrong predictions receive larger penalties. This is not the percentage of races called correctly.</p>
                {difference !== null ? (
                  <>
                    <div
                      className="scoreboard-quality"
                      style={{
                        color: sameError
                          ? "var(--ink)"
                          : difference < 0
                            ? COLORS.I
                            : COLORS.R,
                      }}
                    >
                      {sameError
                        ? "Same error"
                        : relativeError === null
                          ? difference < 0
                            ? "Lower error"
                            : "Higher error"
                          : `${relativeError.toFixed(1)}% ${difference < 0 ? "lower" : "higher"} error`}
                    </div>
                    <p className="scoreboard-caption">
                      {sameError ? "As" : "Than"} the matched {bundle.run.year} comparison
                    </p>
                    <p className="scoreboard-context">
                      Compared with the same races and possible winners in the selected reference forecast.
                    </p>
                  </>
                ) : (
                  <>
                    <div className="scoreboard-quality">
                      Comparison unavailable
                    </div>
                    <p className="scoreboard-context">
                      {comparison?.reason ||
                        "A matched reference is needed to compare probability error."}
                    </p>
                  </>
                )}
                <details className="scoreboard-raw">
                  <summary>Probability score details</summary>
                  <p>
                    Average Brier score across races:{" "}
                    <strong>{score.brier.mean.toFixed(4)}</strong>
                    {matched && reference?.brier
                      ? `; comparison ${reference.brier.mean.toFixed(4)}`
                      : ""}
                    . Lower is better, on a 0–2 scale. This is probability
                    error, not the percentage of races won.
                  </p>
                  <p>
                    For each outcome, subtract 1 from its probability if it won, or 0 if it lost; square each difference and add them. Average those totals across races. Candidates in a grouped outcome are scored together. {score.brier.count} contests scored.
                  </p>
                </details>
              </>
            ) : (
              <p className="scoreboard-context">
                No probability score is available.
              </p>
            )}
          </article>
        </div>
      )}
      <div className="scoreboard-scope">
        <span>
          {score.coverage.scored} of {score.coverage.total} contests scored.
          Grouped candidates count as one model outcome; favorites are not
          individual picks within a group.
        </span>
        {score.issues.length > 0 && (
          <details>
            <summary>Coverage notes</summary>
            {score.issues.map((issue, i) => (
              <p key={i}>{issue}</p>
            ))}
          </details>
        )}
      </div>
    </section>
  );
}

function SeatRange({
  party,
  median,
  lower,
  upper,
  observed,
}: {
  party: string;
  median: number;
  lower: number;
  upper: number;
  observed: number | null;
}) {
  const minimum = Math.max(0, Math.min(lower, observed ?? lower) - 1);
  const maximum = Math.max(minimum + 1, Math.max(upper, observed ?? upper) + 1);
  const x = (value: number) =>
    7 + ((value - minimum) / (maximum - minimum)) * 126;
  const color =
    party === "D" ? COLORS.D : party === "R" ? COLORS.R : COLORS.unresolved;
  return (
    <svg
      viewBox="0 0 140 27"
      role="img"
      aria-label={`${caucusName(party)}: forecast median ${median}, 90% range ${lower} to ${upper} seats; actual ${observed === null ? "unavailable" : observed}. Axis ${minimum} to ${maximum} seats.`}
    >
      <line x1="7" x2="133" y1="12" y2="12" stroke="var(--line)" />
      <line
        x1={x(lower)}
        x2={x(upper)}
        y1="12"
        y2="12"
        stroke={color}
        strokeWidth="6"
        strokeLinecap="round"
        opacity=".4"
      />
      <circle cx={x(median)} cy="12" r="3.5" fill={color} />
      {observed !== null && (
        <line
          x1={x(observed)}
          x2={x(observed)}
          y1="3"
          y2="21"
          stroke="var(--ink)"
          strokeWidth="2"
        />
      )}
    </svg>
  );
}
