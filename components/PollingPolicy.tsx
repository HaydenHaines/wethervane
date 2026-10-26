import { useEffect, useState } from "react";
import type { ExplorerBundle, ExplorerRace } from "../lib/explorer-types";
import { dateLabel, STATE_NAMES } from "../lib/presentation";
import oldReceipt from "../data/polling-selection-20261020.json";
import refreshedReceipt from "../data/polling-selection-20261026.json";
import { senatePollReceiptFor } from "../lib/senate-poll-policy";

interface PollReceipt {
  schemaVersion: number; cutoffAt: string; applicableRuns: Record<string, string>; unknownPublicationDates?: number;
  records: {
    id: string; raceId: string; state: string; pollster: string; status: string; reason: string;
    fieldStart: string | null; fieldEnd: string | null; publicationDate: string | null;
    publicationDateVerified?: boolean; availableBy?: string | null;
    sourceUrl: string; questionUniverse: string; questionKind: string;
    actualFieldworkProvider?: string | null; providerIdentityStatus?: string;
    questionSampleSize?: number | null; nominalSampleSize?: number | null; nominalSampleUniverse?: string | null;
    preferredSurveyQuestions?: { id: string; questionKind: string; questionUniverse: string; included: boolean }[];
  }[];
}
const receipts: PollReceipt[] = [refreshedReceipt, oldReceipt];

export default function PollingPolicy({ bundle, selected, resultSha256 }: {
  bundle: ExplorerBundle;
  selected: ExplorerRace;
  resultSha256: string;
}) {
  const [state, setState] = useState(selected.state);
  useEffect(() => { setState(selected.state); }, [selected.state]);
  const found = senatePollReceiptFor(receipts, bundle, resultSha256);
  const receipt = found ?? refreshedReceipt as PollReceipt;
  const matched = !!found;
  const historicalPolicy = bundle.run.historical && bundle.run.horizonDays === 14 && bundle.provenance.sources.historicalProtocolSha256 === "470fe6f48aca4105338a17b5290f00fa9a3567bc7fdb8d91df14528c31486a28";
  const included = receipt.records.filter((record) => record.status === "included");
  const records = receipt.records.filter((record) => !state || record.state === state);
  const raceCount = new Set(included.map((record) => record.raceId)).size;
  return (
    <section id="poll-selection" className="polling-policy" aria-labelledby="poll-selection-title" tabIndex={-1}>
      <div className="eyebrow">The evidence behind this view</div>
      <h2 id="poll-selection-title" className="section-title">Why some polls are included—and others are not</h2>
      {bundle.run.components.polling === "none" ? (
        <p>This model setup uses electoral history without polling. No polls are included. Choose a polling component in “Build your perspective” to explore a saved forecast that uses polls.</p>
      ) : matched ? (
        <>
          <p>This saved forecast uses <strong>{included.length} poll records across {raceCount} of {bundle.races.length} races</strong>, from sources collected by {dateLabel(receipt.cutoffAt)}. A record is one selected survey question for one race. It is not a count of people interviewed, and this collection does not include every published poll.</p>
          <div className="poll-rule-grid">
            <div><h3>Match the contest and the date</h3><p>The race and named contenders must match the forecast. Sources must document usable percentages and show that the evidence was available by the saved cutoff. Fieldwork must end within 100 days of election day. An availability check does not establish the exact day a poll was published.</p></div>
            <div><h3>Avoid counting the same evidence twice</h3><p>Only one question is chosen from a survey, so alternative ballot questions do not become extra polls. For each race, the latest eligible sample in each polling-provider grouping is retained. Where the actual survey contractor is unknown, the named survey organization supplies that grouping; this is an approximation, not a verified contractor identity.</p></div>
            <div><h3>Keep source gaps visible</h3><p>Missing or uncertain source documents, dates or suitable questions can keep a record out. An absent poll may simply be outside the collected sources. Its absence is not, by itself, a judgment that the pollster is unreliable.</p></div>
          </div>
          <details className="assumptions">
            <summary>How sample sizes, voter types and missing polls are handled</summary>
            <p>The saved records include likely voters (people screened as likely to vote), registered voters, and a source described only as “voters.” These groups are not identical. The model uses their published responses as an approximation to election support.</p>
            <p>The poll count is not a measure of sample size or evidence quality. This version does not weight each poll by its respondent count. It estimates overall polling error from historical data; question-specific sample sizes are often unavailable.</p>
            <p>A race with no included poll still has an electoral-history estimate. Polling in other races can affect shared model estimates. Overlap between respondents in different surveys is unknown.</p>
            <p>{receipt.unknownPublicationDates !== undefined
              ? `${receipt.unknownPublicationDates} included records have no verified exact publication date. Their sources were confirmed available by the recorded upper-bound time; that time is not presented as their release date. `
              : "Exact publication times were unavailable; the earlier edition used recorded publication dates and its collection cutoff. "}
              This collection closes at {new Date(receipt.cutoffAt).toUTCString()}. Publisher evidence was preserved as readable web extracts; original HTML or PDF files were not retained.</p>
          </details>
          <details className="poll-records">
            <summary>Check included records and recorded exclusions by state</summary>
            <p>This list covers the reviewed source collection, including alternative questions. It is not a list of all polls conducted. “Not included” gives the recorded reason; it does not score a pollster’s quality.</p>
            <p>“Dependent” questions use the same survey respondents. A head-to-head question compares two candidates; a multicandidate question lists more. In ranked-choice voting (RCV), lower-ranked candidates are eliminated and ballots transfer to the next remaining choice. A proxy date is an estimated substitute for an unverified publication date.</p>
            <label htmlFor="poll-record-state">Poll records for</label>
            <select id="poll-record-state" value={state} onChange={(event) => setState(event.target.value)}>
              <option value="">All states</option>
              {[...new Set(bundle.races.map((race) => race.state))].sort((a, b) => (STATE_NAMES[a] || a).localeCompare(STATE_NAMES[b] || b)).map((abbreviation) => <option key={abbreviation} value={abbreviation}>{STATE_NAMES[abbreviation] || abbreviation}</option>)}
            </select>
            {records.length ? <ul className="poll-record-list">{records.map((record) => (
              <li key={record.id}>
                <div><strong>{record.pollster}</strong> · {STATE_NAMES[record.state] || record.state} <span className={`poll-record-status ${record.status}`}>{record.status === "included" ? "Included" : "Not included"}</span></div>
                <p className="small">{record.fieldStart && record.fieldEnd ? `Interview dates: ${record.fieldStart} to ${record.fieldEnd}. ` : "Interview dates not verified. "}{record.publicationDate
                  ? `${record.publicationDateVerified === false ? "Unverified recorded date" : "Recorded publication date"}: ${record.publicationDate}. `
                  : "Exact publication date not verified. "}{record.availableBy && `Source confirmed available by ${new Date(record.availableBy).toUTCString()} (not a publication date). `}{record.questionUniverse}. {record.questionKind}</p>
                <p>{record.reason}</p>
                {!!record.preferredSurveyQuestions?.length && <p className="small">Preferred question in this same survey: {record.preferredSurveyQuestions.map(question => `${question.questionKind}; ${question.questionUniverse}${question.included ? " (included)" : " (not retained in this edition)"}`).join(". ")}.</p>}
                {receipt.schemaVersion === 2 && <p className="small">{record.questionSampleSize != null ? `Reported sample size for this question: ${record.questionSampleSize}. ` : "Sample size for this exact question is not verified. "}
                  {record.nominalSampleSize != null && `Reported overall survey sample: ${record.nominalSampleSize}${record.nominalSampleUniverse ? ` · ${record.nominalSampleUniverse}` : ""}. `}These are publisher sample labels, not model weights or reconstructed vote counts.</p>}
                {record.providerIdentityStatus?.includes("proxy") && <p className="small">The polling-provider grouping uses the named survey organization because its actual fieldwork contractor is unverified.</p>}
                {record.sourceUrl && <a href={record.sourceUrl} target="_blank" rel="noreferrer">Publisher or source record ↗</a>}
              </li>
            ))}</ul> : <p>No poll records for this state appear in the reviewed collection. That does not establish that no polls were conducted.</p>}
          </details>
        </>
      ) : historicalPolicy ? (
        <>
          <p>This historical view uses a cutoff 14 days before the election. It keeps eligible polls ending within 100 days of election day, then selects the latest eligible record per race and polling provider. The questions must match the contest’s modeled candidates.</p>
          <p>Publication availability is reconstructed from the later of the fieldwork end date and the day after the archive’s creation date. This is an approximation, not proof of when the original public release was available. Some inputs were assembled after the election.</p>
          <p>The historical source checks also exclude unsupported candidate matchups, hypothetical questions, unusable figures, and duplicate or overlapping records. These are different source checks from the 2026 collection. A poll-by-poll inclusion and exclusion list for this historical view is not yet published here.</p>
        </>
      ) : (
        <p>{bundle.run.historical ? "Historical views use a separate polling collection and reconstructed publication dates. The current 2026 source rules and counts do not apply to these past elections. A poll-by-poll inclusion record for this historical view is not yet published here." : "Poll selection details have not been published for this saved result. A missing count does not establish that no polls were conducted."}</p>
      )}
    </section>
  );
}
