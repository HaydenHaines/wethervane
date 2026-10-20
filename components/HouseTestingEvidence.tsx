"use client";

import type { HouseComparisonResult } from "../lib/house-comparison";
import useHouseScoreEvidence from "./useHouseScoreEvidence";

const points = (value: number, digits = 1) => (value * 100).toFixed(digits);

export default function HouseTestingEvidence({ publication, reference }: {
  publication: HouseComparisonResult;
  reference?: HouseComparisonResult;
}) {
  const selected = useHouseScoreEvidence(publication);
  const compared = useHouseScoreEvidence(reference);
  const evidence = selected?.status === "verified" ? selected.summary : null;
  const other = compared?.status === "verified" ? compared.summary : null;
  return <details className="house-score-evidence">
    <summary>More about vote-share tests</summary>
    {selected?.status === "loading" && <p role="status">Checking the saved vote-share tests…</p>}
    {selected?.status === "unavailable" && <p>{selected.reason} The winner and probability scores above are still available.</p>}
    {evidence && <>
      <p><strong>{publication.result.run.label} · {evidence.year}</strong><br />
        These tests cover the same {evidence.total} included districts. Vote share counts Democratic and Republican votes only; votes for other candidates are excluded.</p>
      {reference && <p className="house-evidence-reference">Compared model: <strong>{reference.result.run.label}</strong>.
        {compared?.status === "loading" && <span role="status"> Checking its vote-share tests…</span>}
        {compared?.status === "unavailable" && <span> Its vote-share tests are unavailable: {compared.reason}</span>}
      </p>}
      <dl className="house-evidence-metrics">
        <div className="house-evidence-mae">
          <dt>Average Democratic vote-share miss<p>Average size of the difference between predicted and actual Democratic share of Democratic + Republican votes. A prediction of 55% and a result of 52% differ by 3 percentage points. Lower is closer.</p></dt>
          <dd><strong>{points(evidence.meanAbsoluteError)} percentage points</strong>
            {other && <span>Compared: {points(other.meanAbsoluteError)} percentage points</span>}
          </dd>
        </div>
        <div className="house-evidence-coverage">
          <dt>Actual results inside modeled 90% ranges<p>Each range contains the middle 90% of simulated vote shares. Wider ranges can contain more results, so read this percentage alongside their width.</p></dt>
          <dd><strong>{(100 * evidence.covered / evidence.total).toFixed(1)}% · {evidence.covered} / {evidence.total} districts</strong>
            <span>Average range width: {points(evidence.meanIntervalWidth)} percentage points</span>
            {other && <span>Compared: {(100 * other.covered / other.total).toFixed(1)}% · {other.covered} / {other.total} districts<br />Average range width: {points(other.meanIntervalWidth)} percentage points</span>}
          </dd>
        </div>
        <div className="house-evidence-crps">
          <dt>Full-distribution error (CRPS)<p>The continuous ranked probability score assesses the possible vote shares and probabilities the model assigned to them against the actual result. Lower is better; 0 is perfect.</p></dt>
          <dd><strong>{points(evidence.crps, 2)} points</strong>
            {other && <span>Compared: {points(other.crps, 2)} points</span>}
          </dd>
        </div>
      </dl>
      <p><strong>Research stage:</strong> Experimental. <strong>Saved calculation checks:</strong> Passed for individual races. These describe different things: successful calculation checks do not establish future forecasting accuracy.</p>
      <p>These are historical development tests. Districts in the same election can move together, and the measures can favor different models. The saved test records are matched to this exact model and election year.</p>
    </>}
  </details>;
}
