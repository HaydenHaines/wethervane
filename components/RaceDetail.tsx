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
  const siblings = bundle.races.filter((r) => r.state === race.state);
  const shareOutcomes = race.outcomes.filter((o) => o.share);
  return (
    <aside
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
          Observed winner: <strong>{race.observed.winnerName}</strong> ·{" "}
          {partyName(race.observed.winnerParty)}
        </p>
      )}
      <hr className="detail-divider" />
      <h3 className="detail-label">Chance of winning</h3>
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
                        ? " · Incumbent"
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
                <p className="observed-result">Contains the observed winner</p>
              )}
            </div>
          );
        })}
      </div>
      {shareOutcomes.length > 0 ? (
        <>
          <hr className="detail-divider" />
          <h3 className="detail-label">Principal contender vote share</h3>
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
                    aria-label={`${o.name}: mean ${percent(share.mean)}, 90% interval ${percent(share.quantiles.p05)} to ${percent(share.quantiles.p95)}`}
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
                    90% interval: {percent(share.quantiles.p05)}–
                    {percent(share.quantiles.p95)}
                    <br />
                    {share.denominator}
                  </p>
                </div>
              );
            })}
          </div>
          <p className="detail-note">
            Share intervals describe modeled vote share, not win probability.
            These shares are conditional on the two principal contenders.
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
          <span className="small">Selected stack</span>
          <strong style={{ fontSize: 16 }}>{bundle.run.label}</strong>
        </div>
        {race.pollCount !== undefined && (
          <div className="metric">
            <span className="small">Admitted observations</span>
            <strong>{race.pollCount}</strong>
          </div>
        )}
      </div>
      {race.notes.length > 0 && (
        <details className="assumptions">
          <summary>Race assumptions</summary>
          {race.notes.map((note, i) => (
            <p className="detail-note" key={i}>
              {note}
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
                <span>{m.name}: source in saved artifact</span>
              )}
            </span>
          ))}
        {race.observed?.sourceUrls[0] && (
          <a
            href={race.observed.sourceUrls[0]}
            target="_blank"
            rel="noreferrer"
          >
            Observed result ↗
          </a>
        )}
      </div>
      <p className="source-date">
        Forecast date: {dateLabel(bundle.run.forecastAt)}
        <br />
        Evidence cutoff: {dateLabel(bundle.run.cutoffAt)}
        <br />
        {bundle.run.historical
          ? "Historical research reconstruction; retrospective proxies apply."
          : "Saved research output; not a live forecast."}
      </p>
    </aside>
  );
}
