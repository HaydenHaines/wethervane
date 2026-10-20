"use client";

import { useEffect, useRef } from "react";
import type { ComponentStudyResult } from "../lib/component-study";
import { studySourceCopy } from "../lib/component-study-view";
import {
  dateLabel,
  partyColor,
  percent,
  probability,
  STATE_NAMES,
} from "../lib/presentation";

export default function StudyRaceDetail({
  result,
  race,
}: {
  result: ComponentStudyResult;
  race: ComponentStudyResult["races"][number];
}) {
  const detail = useRef<HTMLElement>(null);
  useEffect(() => {
    detail.current?.scrollTo({ top: 0 });
  }, [race.id]);
  const observed = race.observed;
  const aheadParty =
    observed.democraticVotes === observed.republicanVotes
      ? null
      : observed.democraticVotes > observed.republicanVotes
        ? "D"
        : "R";
  const ahead = race.candidates.find(
    (candidate) => candidate.party === aheadParty,
  );
  return (
    <aside
      ref={detail}
      id="race-detail"
      className="detail"
      tabIndex={0}
      aria-label="Selected study race detail"
      aria-live="polite"
    >
      <div className="eyebrow">{result.year} race in focus</div>
      <h2 id="race-detail-title" tabIndex={-1}>
        {STATE_NAMES[race.state] || race.state}
      </h2>
      <p className="small">U.S. Senate · Regular election</p>
      <h3 className="detail-label">Chance of winning this modeled contest</h3>
      <p className="detail-note">
        These chances concern the Democratic–Republican contest modeled here. A
        70% chance means winning about 70 of 100 simulated contests, not
        receiving 70% of the votes.
      </p>
      <div className="candidate-grid">
        {race.candidates.map((candidate) => (
          <div className="candidate" key={candidate.id}>
            <div className="candidate-top">
              <div>
                <div className="candidate-name">{candidate.name}</div>
                <div className="party-label">
                  {candidate.party === "D" ? "Democratic" : "Republican"}
                  {candidate.incumbent &&
                    ` · Held this seat at the ${result.year} forecast cutoff`}
                </div>
              </div>
              <strong
                className="candidate-prob"
                style={{ color: partyColor(candidate.party) }}
              >
                {probability(candidate.winProbability)}
              </strong>
            </div>
            <div className="candidate-track" aria-hidden="true">
              <span
                style={{
                  width: `${candidate.winProbability * 100}%`,
                  background: partyColor(candidate.party),
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <hr className="detail-divider" />
      <h3 className="detail-label">Share of Democratic–Republican votes</h3>
      <p className="detail-note">
        D/R means Democratic and Republican. These shares divide their votes
        between the two named candidates; votes for other candidates are
        excluded.
      </p>
      <div className="intervals-grid">
        {race.candidates.map((candidate) => {
          const quantile = (level: number) =>
            candidate.share.quantiles.find((point) => point.level === level)!
              .value;
          const lower95 = quantile(0.025),
            upper95 = quantile(0.975);
          const lower80 = quantile(0.1),
            upper80 = quantile(0.9);
          const color = partyColor(candidate.party);
          return (
            <div className="interval-row" key={candidate.id}>
              <div className="interval-label">
                <span>{candidate.name}</span>
                <strong>{percent(candidate.share.mean)} average</strong>
              </div>
              <div
                className="interval-line"
                role="img"
                aria-label={`${candidate.name}: average conditional D/R share ${percent(candidate.share.mean)}; modeled 80% range ${percent(lower80)} to ${percent(upper80)}; modeled 95% range ${percent(lower95)} to ${percent(upper95)}`}
              >
                <span
                  className="interval-range"
                  aria-hidden="true"
                  style={{
                    left: `${lower95 * 100}%`,
                    width: `${(upper95 - lower95) * 100}%`,
                    background: color,
                    opacity: 0.25,
                  }}
                />
                <span
                  className="interval-range"
                  aria-hidden="true"
                  style={{
                    left: `${lower80 * 100}%`,
                    width: `${(upper80 - lower80) * 100}%`,
                    background: color,
                    opacity: 0.8,
                  }}
                />
                <span
                  className="interval-mean"
                  aria-hidden="true"
                  style={{
                    left: `${candidate.share.mean * 100}%`,
                    background: color,
                  }}
                />
              </div>
              <p className="detail-note">
                Modeled 80% range: {percent(lower80)}–{percent(upper80)}
                <br />
                Modeled 95% range: {percent(lower95)}–{percent(upper95)}
              </p>
            </div>
          );
        })}
      </div>
      <p className="detail-note">
        The darker range contains the middle 80% of simulated shares (10th–90th
        percentiles); the lighter range contains the middle 95% (2.5th–97.5th).
        The mark shows the average. These are model ranges, not poll margins of
        error or guarantees.
      </p>
      <hr className="detail-divider" />
      <h3 className="detail-label">Actual result among D/R votes</h3>
      <div className="metrics">
        {race.candidates.map((candidate) => (
          <div className="metric" key={candidate.id}>
            <span className="small">
              {candidate.party === "D" ? "Democratic" : "Republican"}
            </span>
            <strong>
              {percent(
                candidate.party === "D"
                  ? observed.democraticTwoPartyShare
                  : 1 - observed.democraticTwoPartyShare,
              )}
            </strong>
            <span className="small">
              {(candidate.party === "D"
                ? observed.democraticVotes
                : observed.republicanVotes
              ).toLocaleString("en-US")}{" "}
              votes
            </span>
          </div>
        ))}
      </div>
      <p className="detail-note">
        {ahead
          ? `${ahead.name} received more votes in this D/R comparison.`
          : "The D/R vote totals were tied."}{" "}
        Other candidates' votes are excluded from this comparison.
      </p>
      <hr className="detail-divider" />
      <div className="metric">
        <span className="small">Selected model setup</span>
        <strong style={{ fontSize: 16 }}>{result.label}</strong>
      </div>
      <p className="source-date">
        Study reference date: {dateLabel(result.informationCutoff)}
        <br />
        Reconstructed after the election using some inputs assembled later.
        Candidate names come from final election records; their availability on
        the reference date has not been verified.
      </p>
      <details className="assumptions">
        <summary>Race and source qualifications</summary>
        <p className="detail-note">The model's settings were estimated from earlier elections. Some historical inputs were assembled later. Poll availability is approximated from dates; original publication dates and revisions have not been verified.</p>
        <p className="detail-note">D and R identify the two possible outcomes within each race. They do not identify a person across elections.</p>
      </details>
      <h3 className="detail-label">Candidate and result sources</h3>
      <div className="official-links">
        {result.sourceAttribution.map(studySourceCopy).map((source) => (
          <div key={source.label}>
            <strong>{source.label}</strong>
            <p className="detail-note">{source.description}</p>
            {source.urls.map((url, index) => (
              <a key={url} href={url} target="_blank" rel="noreferrer">
                {source.label}
                {source.urls.length > 1 ? ` · Reference ${index + 1}` : ""} ↗
              </a>
            ))}
          </div>
        ))}
      </div>
    </aside>
  );
}
