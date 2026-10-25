import { forecastLabel } from "../lib/forecast-copy";
import type { ModelSliceView, SliceControl } from "../lib/model-slices";

const fieldId = (id: string) => `model-choice-${id}`;

/** One responsive renderer for available, optional, required and future slices. */
export default function ModelSliceControls({ slices, busy, explain, onChange }: {
  slices: ModelSliceView[];
  busy: boolean;
  explain: (slot: string, id: string, fallback: string) => string;
  onChange: (id: string, value: string) => void;
}) {
  function controlView(control: SliceControl) {
    const { toggle } = control;
    const included = !toggle || control.value !== toggle.offValue;
    const options = control.options.filter((option) => option.id !== toggle?.offValue);
    const displayedValue = included ? control.value : toggle!.onValue;
    const selected = control.options.find((option) => option.id === control.value);
    const hintId = `${fieldId(control.id)}-hint`;
    return <div className="slice-control stack-field" key={control.id}>
      {toggle && <label className="slice-include">
        <input type="checkbox" checked={included} disabled={busy} aria-describedby={hintId}
          onChange={(event) => onChange(control.id, event.target.checked ? toggle.onValue : toggle.offValue)} />
        {toggle.label}
      </label>}
      {options.length > 1 || !options.some((option) => option.id === displayedValue) ? <>
        <label htmlFor={fieldId(control.id)}>{control.label}</label>
        <select id={fieldId(control.id)} value={displayedValue} disabled={busy || !included} aria-describedby={hintId}
          onChange={(event) => onChange(control.id, event.target.value)}>
          {!options.some((option) => option.id === displayedValue) && <option value={displayedValue} disabled>{displayedValue || "Choose a component"} · Unsupported</option>}
          {options.map((option) => <option key={option.id} value={option.id} disabled={option.availability === "unavailable"}>
            {forecastLabel(option.label)}{option.availability === "requires_computation" ? " · Needs a new calculation" : option.availability === "unavailable" ? " · Unavailable" : ""}
          </option>)}
        </select>
      </> : <p className="slice-fixed-choice">{included ? forecastLabel(options[0]?.label || control.label) : "Polling is off"}</p>}
      <p id={hintId}>
        {included && selected?.experimental && <span className="component-flag">Experimental</span>}
        {explain(control.id, selected?.id || "", selected?.description || control.description)}
      </p>
    </div>;
  }
  return <section className="model-slices" aria-labelledby="model-slices-title">
    <div className="model-slices-intro">
      <h3 id="model-slices-title">The full model, slice by slice</h3>
      <p>Choose the ingredients for your requested setup. Results change when a matching saved forecast loads. Unfinished slices stay visible as the model develops.</p>
    </div>
    <div className="model-slice-list">
      {slices.map((slice) => {
        const unavailable = slice.controls.length === 0;
        const fixed = unavailable && slice.builtIn === true;
        const off = !unavailable && slice.controls.every((control) => control.toggle && control.value === control.toggle.offValue);
        const alwaysIncluded = !unavailable && slice.controls.every((control) => control.required);
        const hintId = `slice-${slice.id}-availability`;
        return <section key={slice.id} className={`model-slice${unavailable ? (fixed ? " is-fixed" : " is-unavailable") : off ? " is-off" : " is-included"}`} data-slice={slice.id} aria-labelledby={`slice-${slice.id}-title`}>
          <div className="model-slice-description">
            <div className="model-slice-heading">
              {unavailable && !fixed ? <input type="checkbox" checked={false} disabled readOnly aria-label={`Include ${slice.label.toLowerCase()}`} aria-describedby={hintId} /> : <span className="slice-state-mark" aria-hidden="true">{off ? "−" : "✓"}</span>}
              <h4 id={`slice-${slice.id}-title`}>{slice.label}</h4>
              <span className="slice-state">{fixed ? "Built into this model" : unavailable ? "Not connected yet" : alwaysIncluded ? "Always included" : off ? "Off" : "Included"}</span>
            </div>
            <p>{slice.purpose}</p>
          </div>
          {unavailable ? <p id={hintId} className={`slice-unavailable-reason${fixed ? " slice-fixed-reason" : ""}`}>{fixed ? slice.builtInReason : slice.unavailableReason}</p>
            : <div className="model-slice-options">{slice.controls.map(controlView)}</div>}
        </section>;
      })}
    </div>
    <p className="model-slices-footnote">“Always included” means this part is required by these saved models. “Built into this model” means it is present but has no separate control. Disabled choices cannot change this result. These choices belong to one model with shared assumptions; they are not percentages to blend.</p>
  </section>;
}
