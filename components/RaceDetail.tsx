import { forecastLabel } from "../lib/forecast-copy";
import { useEffect, useRef } from "react";
import { explainResearchNote } from "../lib/forecast-copy";
import type { ExplorerBundle, ExplorerRace } from "../lib/explorer-types";
import {
  STATE_NAMES,
  candidateShortName,
  dateLabel,
  partyColor,
  outcomeColor,
  partyName,
  percent,
  probability,
  raceColor,
  raceRating,
  raceTextColor,
  ppChange,
} from "../lib/presentation";

export default function RaceDetail({
  bundle,
  race,
  baseline,
  onSelect,
}: {
  bundle: ExplorerBundle;
  race: ExplorerRace;
  baseline?: ExplorerRace;
  onSelect: (id: string) => void;
}) {
  const detail = useRef<HTMLElement>(null);
  useEffect(() => { detail.current?.scrollTo({ top: 0 }); }, [race.id]);
  const siblings = bundle.races.filter((r) => r.state === race.state);
  const shareOutcomes = race.outcomes.filter((o) => o.share);
  return (
    <aside
      ref={detail}
      tabIndex={0}
      id="race-detail"
      className="detail"
      aria-label="Selected race detail"
      aria-live="polite"
    >
      <div className="detail-kicker">
        <span className="eyebrow" style={{ margin: 0 }}>
          Race in focus
        </span>
        <span
          className="race-tag"
          style={{
            color: raceTextColor(race),
            background: `${raceColor(race)}22`,
          }}
        >
          {raceRating(race)}
        </span>
      </div>
      <h2 id="race-detail-title" tabIndex={-1}>
        {STATE_NAMES[race.state] || race.state}
      </h2>
      <div className="small">U.S. Senate · {race.electionLabel}</div>
      {siblings.length > 1 && (
        <>
          <label className="sr-only" htmlFor="seat-select">
            Choose election in {STATE_NAMES[race.state]}
          </label>
          <select
            id="seat-select"
            className="race-selector"
            value={race.id}
            onChange={(e) => onSelect(e.target.value)}
          >
            {siblings.map((r) => (
              <option key={r.id} value={r.id}>
                {r.electionLabel} ·{" "}
                {r.outcomes
                  .map((o) =>
                    o.kind === "group" ? o.name : candidateShortName(o.name),
                  )
                  .join(" / ")}
              </option>
            ))}
          </select>
        </>
      )}
      {race.observed && (
        <p className="observed-result winner">
          Actual winner: <strong>{race.observed.winnerName}</strong> ·{" "}
          {partyName(race.observed.winnerParty)}
        </p>
      )}
      <hr className="detail-divider" />
      <h3 className="detail-label">Chance of winning</h3>
      <p className="detail-note">A 70% chance means winning about 70 out of 100 elections simulated by this model. It does not mean winning 70% of the votes.</p>
      <div className="candidate-grid">
        {race.outcomes.map((outcome) => {
          const reference = baseline?.outcomes.find((o) => o.id === outcome.id);
          return (
            <div className="candidate" key={outcome.id}>
              <div className="candidate-top">
                <div>
                  <div className="candidate-name">{outcome.name}</div>
                  <div className="party-label">
                    {partyName(outcome.party)}
                    {outcome.kind === "group"
                      ? " · Group outcome"
                      : outcome.members[0]?.incumbent
                        ? " · Currently holds this seat"
                        : ""}
                  </div>
                </div>
                <strong
                  className="candidate-prob"
                  style={{ color: outcomeColor(outcome) }}
                >
                  {probability(outcome.winProbability)}
                </strong>
              </div>
              <div className="candidate-track">
                <span
                  style={{
                    width: `${outcome.winProbability * 100}%`,
                    background: outcomeColor(outcome),
                  }}
                />
              </div>
              {outcome.kind === "group" && (
                <p className="detail-note">
                  Group probability for{" "}
                  {outcome.members.map((m) => m.name).join("; ")}. This combined
                  probability is not allocated between individual members.
                </p>
              )}
              {reference && (
                <span className="comparison-delta">
                  {ppChange(outcome.winProbability - reference.winProbability)}{" "}
                  vs comparison
                </span>
              )}
              {race.observed?.outcomeId === outcome.id && (
                <p className="observed-result">Includes the actual winner</p>
              )}
            </div>
          );
        })}
      </div>
      {shareOutcomes.length > 0 ? (
        <>
          <hr className="detail-divider" />
          <h3 className="detail-label">Vote share among these two contenders</h3>
          <div className="intervals-grid">
            {shareOutcomes.map((o) => {
              const share = o.share!;
              return (
                <div className="interval-row" key={o.id}>
                  <div className="interval-label">
                    <span>{candidateShortName(o.name)}</span>
                    <strong>{percent(share.mean)}</strong>
                  </div>
                  <div
                    className="interval-line"
                    aria-label={`${o.name}: mean ${percent(share.mean)}, Modeled 90% range ${percent(share.quantiles.p05)} to ${percent(share.quantiles.p95)}`}
                  >
                    <span
                      className="interval-range"
                      style={{
                        left: `${share.quantiles.p05 * 100}%`,
                        width: `${(share.quantiles.p95 - share.quantiles.p05) * 100}%`,
                        background: outcomeColor(o),
                      }}
                    />
                    <span
                      className="interval-mean"
                      style={{
                        left: `${share.mean * 100}%`,
                        background: outcomeColor(o),
                      }}
                    />
                  </div>
                  <p className="detail-note">
                    Modeled 90% range: {percent(share.quantiles.p05)}–
                    {percent(share.quantiles.p95)}
                    <br />
                    {share.denominator}
                  </p>
                </div>
              );
            })}
          </div>
          <p className="detail-note">
            The shares divide votes between these two contenders only; votes for other candidates are excluded. The range runs from the 5th to the 95th percentile of simulated shares—the middle 90%. It is a model estimate, not a poll’s margin of error or a guarantee.
          </p>
        </>
      ) : (
        <>
          <hr className="detail-divider" />
          <p className="detail-note">
            This historical reconstruction provides outcome probabilities.
            Candidate vote-share estimates are unavailable.
          </p>
        </>
      )}
      <p className="detail-note">
        Probabilities of 0% or 100% reflect the saved simulation, not certainty.
      </p>
      <hr className="detail-divider" />
      <div className="metrics">
        <div className="metric">
          <span className="small">Selected model setup</span>
          <strong style={{ fontSize: 16 }}>{forecastLabel(bundle.run.label)}</strong>
        </div>
        {race.pollCount !== undefined && (
          <div className="metric">
            <span className="small">Poll records included</span>
            <strong>{race.pollCount}</strong>
          </div>
        )}
      </div>
      {race.pollCount !== undefined && (
        <p className="detail-note">
          {race.pollCount === 0 ? "No poll records for this race were included in this saved forecast. " : "This counts poll records used for this race, not people interviewed or every poll available. "}
          <a href="#poll-selection">Why some polls are included—and others are not ↓</a>
        </p>
      )}
      {bundle.run.components.polling === "none" && <p className="detail-note">This setup uses electoral history without polls.</p>}
      {race.notes.length > 0 && (
        <details className="assumptions">
          <summary>Race assumptions</summary>
          {race.notes.map((note, i) => (
            <p className="detail-note" key={i}>
              {explainResearchNote(note)}
            </p>
          ))}
        </details>
      )}
      <hr className="detail-divider" />
      <h3 className="detail-label">Candidate and result sources</h3>
      <div className="official-links">
        {race.outcomes
          .flatMap((o) => o.members)
          .map((m) => (
            <span key={m.id}>
              {m.sourceUrls[0] ? (
                <a href={m.sourceUrls[0]} target="_blank" rel="noreferrer">
                  {m.name} ↗
                </a>
              ) : (
                <span>{m.name}: source recorded in the saved data</span>
              )}
            </span>
          ))}
        {race.observed?.sourceUrls[0] && (
          <a
            href={race.observed.sourceUrls[0]}
            target="_blank"
            rel="noreferrer"
          >
            Actual result ↗
          </a>
        )}
      </div>
      <p className="source-date">
        Forecast date: {dateLabel(bundle.run.forecastAt)}
        <br />
        Inputs dated through: {dateLabel(bundle.run.cutoffAt)}
        <br />
        {bundle.run.historical
          ? "Recreated after the election; some historical inputs were assembled later."
          : "Saved research forecast; new polls are not added automatically."}
      </p>
    </aside>
  );
}
