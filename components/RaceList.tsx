import { useEffect, useMemo, useState } from "react";
import RaceListToggle from "./RaceListToggle";
import type { ExplorerRace } from "../lib/explorer-types";
import {
  STATE_NAMES,
  competitiveOrder,
  leadingOutcome,
  partyColor,
  outcomeColor,
  partyName,
  ppChange,
  probability,
  raceRating,
  raceTextColor,
} from "../lib/presentation";

export default function RaceList({
  races,
  selectedId,
  baseline,
  onSelect,
}: {
  races: ExplorerRace[];
  selectedId: string;
  baseline?: ExplorerRace[];
  onSelect: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("competitive");
  const [expanded, setExpanded] = useState(false);
  useEffect(() => { setExpanded(false); }, [races]);
  const shown = useMemo(
    () =>
      [...races]
        .filter((r) =>
          `${STATE_NAMES[r.state]} ${r.state} ${r.electionLabel} ${r.outcomes.flatMap((o) => [o.name, ...o.members.map((m) => m.name)]).join(" ")}`
            .toLowerCase()
            .includes(query.trim().toLowerCase()),
        )
        .sort(
          sort === "state"
            ? (a, b) =>
                (STATE_NAMES[a.state] || a.state).localeCompare(
                  STATE_NAMES[b.state] || b.state,
                ) || a.id.localeCompare(b.id)
            : competitiveOrder,
        ),
    [races, query, sort],
  );
  const visible = expanded ? shown : shown.slice(0, 10);
  return (
    <section className="race-section" id="races" aria-labelledby="races-title">
      <div className="section-top">
        <h2 className="section-title" id="races-title">
          The races
        </h2>
        <span className="small">
          {visible.length} of {shown.length} {query.trim() ? "matching elections" : "elections"}
        </span>
      </div>
      <p className="small">The favorite is the candidate or group with the highest modeled chance of winning. “Most competitive” puts the least certain favorites first.</p>
      <div className="race-tools">
        <div className="search-wrap">
          <span className="search-icon" aria-hidden="true">
            ⌕
          </span>
          <input
            aria-label="Search states or candidates"
            placeholder="Find a state or candidate"
            value={query}
            onChange={(e) => { setQuery(e.target.value); setExpanded(false); }}
          />
        </div>
        <select
          className="sort-select"
          aria-label="Sort races"
          value={sort}
          onChange={(e) => { setSort(e.target.value); setExpanded(false); }}
        >
          <option value="competitive">Most competitive</option>
          <option value="state">State A–Z</option>
        </select>
      </div>
      <table className="race-table" id="senate-race-list">
        <thead>
          <tr>
            <th scope="col">State / election</th>
            <th scope="col">Favorite</th>
            <th scope="col">Chance to win</th>
            <th scope="col">Outlook</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((r) => {
            const lead = leadingOutcome(r);
            const reference = baseline
              ?.find((b) => b.id === r.id)
              ?.outcomes.find((o) => o.id === lead.id);
            return (
              <tr key={r.id} className={r.id === selectedId ? "selected" : ""}>
                <td>
                  <button
                    className="race-name"
                    aria-pressed={r.id === selectedId}
                    onClick={() => onSelect(r.id)}
                  >
                    {STATE_NAMES[r.state] || r.state}
                    <span className="race-sub">{r.electionLabel}</span>
                  </button>
                </td>
                <td>
                  <span style={{ color: outcomeColor(lead) }}>{lead.name}</span>
                  <span className="race-sub">
                    {partyName(lead.party)}
                    {lead.kind === "group" ? " · Group" : ""}
                  </span>
                </td>
                <td className="chance-cell">
                  <span className="win-label">
                    {probability(lead.winProbability)}
                  </span>
                  <div className="mini-prob" aria-hidden="true">
                    {r.outcomes.map((o) => (
                      <span
                        key={o.id}
                        style={{
                          width: `${o.winProbability * 100}%`,
                          background: outcomeColor(o),
                        }}
                      />
                    ))}
                  </div>
                  {reference && (
                    <span className="comparison-delta">
                      {ppChange(lead.winProbability - reference.winProbability)}{" "}
                      vs comparison
                    </span>
                  )}
                </td>
                <td>
                  <span className="small" style={{ color: raceTextColor(r) }}>
                    {raceRating(r)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <RaceListToggle expanded={expanded} total={shown.length} controls="senate-race-list" onToggle={() => setExpanded(!expanded)} />
      {shown.length === 0 && (
        <p className="empty">
          No matching races. Try another state or candidate.
        </p>
      )}
    </section>
  );
}
