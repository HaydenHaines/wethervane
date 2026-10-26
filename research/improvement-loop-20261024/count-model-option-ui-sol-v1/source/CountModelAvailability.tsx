import type { ExplorerRun } from "../lib/explorer-types";

/** Availability preparation only: no count-result contract or forecast is connected. */
export default function CountModelAvailability({ displayedRun }: { displayedRun: ExplorerRun }) {
  return <section className="count-model-availability" aria-labelledby="count-model-option-title">
    <div>
      <h3 id="count-model-option-title">Count-model option</h3>
      <button type="button" disabled aria-describedby="count-model-option-status">Count model · Not available</button>
    </div>
    <p id="count-model-option-status">No count-model result is connected to this view. The displayed forecast remains <strong>{displayedRun.label}</strong>.</p>
    <dl>
      <div><dt>Result</dt><dd>No published result connected</dd></div>
      <div><dt>Diagnostics</dt><dd>No count-model diagnostics connected</dd></div>
      <div><dt>Coverage</dt><dd>No supported race or chamber coverage connected</dd></div>
    </dl>
    <details>
      <summary>How this differs from the saved choices</summary>
      <p>The current choices change saved race-level forecasts. A count model would describe participation and votes for named candidates, with its own election rules, quantities and coverage. It cannot be substituted for the electoral-history setting or added as a weight.</p>
      <p>An available count-model forecast could appear as an optional result without becoming the default. Its diagnostics, supported races and limitations would be shown separately from evidence about predictive performance.</p>
      <p>No new calculation has been requested or scheduled.</p>
    </details>
  </section>;
}
