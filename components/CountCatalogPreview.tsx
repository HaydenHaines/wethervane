import { useId } from "react";
import type { CountCatalogPresentation } from "../lib/count-catalog-view.ts";

const styles = `
.count-catalog-preview{max-width:1100px;margin:auto;color:var(--ink,#24332b);font:14px/1.7 Arial,sans-serif}
.count-catalog-preview *{box-sizing:border-box}
.count-catalog-preview h2{font:32px/1.2 Georgia,serif;margin:12px 0}
.count-catalog-preview h3{font:24px/1.3 Georgia,serif;margin:0 0 12px}
.count-catalog-preview p{max-width:820px}
.count-catalog-preview .catalog-note{padding:16px 20px;background:#fff4d8;border:1px solid #c7b477;border-radius:6px;margin-bottom:24px}
.count-catalog-preview .catalog-note p{margin:4px 0 0}
.count-catalog-preview .catalog-scope{display:grid;grid-template-columns:minmax(150px,.5fr) minmax(0,1fr);gap:6px 20px;padding:18px 0}
.count-catalog-preview dt{font-weight:600}
.count-catalog-preview dd{margin:0;min-width:0;overflow-wrap:anywhere}
.count-catalog-preview .catalog-states{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:12px;margin:20px 0 32px}
.count-catalog-preview .catalog-state{padding:18px;border:1px solid var(--line,#d8ddd5);border-radius:8px;background:var(--paper,#fafbf7);min-width:0}
.count-catalog-preview .catalog-state h3{font:600 13px/1.5 Arial,sans-serif}
.count-catalog-preview .catalog-state strong{display:block;font-size:16px}
.count-catalog-preview .catalog-state p{font-size:13px;margin:8px 0 0}
.count-catalog-preview .catalog-roles{list-style:none;padding:0;border:1px solid var(--line,#d8ddd5);border-radius:8px;overflow:hidden}
.count-catalog-preview .model-slice{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;padding:20px;background:var(--paper,#fafbf7)}
.count-catalog-preview .model-slice+.model-slice{border-top:1px solid var(--line,#d8ddd5)}
.count-catalog-preview .model-slice h4{font-size:16px;margin:0 0 8px}
.count-catalog-preview .model-slice p{margin:6px 0 0}
.count-catalog-preview .catalog-required{display:inline-block;font-size:12px;padding:3px 7px;border-radius:4px;background:#e7eee7}
.count-catalog-preview details{min-width:0}
.count-catalog-preview summary{min-height:44px;display:list-item;padding:10px 0;cursor:pointer;font-weight:600}
.count-catalog-preview summary:focus-visible{outline:3px solid #245c46;outline-offset:3px;border-radius:3px}
.count-catalog-preview details dl{font-size:12px;margin:4px 0}
.count-catalog-preview details dd{margin-bottom:8px}
.count-catalog-preview code{font-size:12px;overflow-wrap:anywhere;white-space:normal}
.count-catalog-preview .catalog-details{border-top:1px solid var(--line,#d8ddd5);padding-top:14px;margin-top:26px}
@media(max-width:700px){.count-catalog-preview h2{font-size:27px}.count-catalog-preview .model-slice{grid-template-columns:minmax(0,1fr);gap:8px;padding:16px}.count-catalog-preview .catalog-scope{display:block}.count-catalog-preview .catalog-scope dd{margin-bottom:12px}}
`;

/** Private offline metadata preview; no route, execution control or result chart. */
export default function CountCatalogPreview({ view }: { view: CountCatalogPresentation | null }) {
  const id = useId();
  if (!view) return <section className="count-catalog-preview" aria-label="Private count research catalog unavailable">
    <style>{styles}</style><h2>Count research catalog unavailable</h2>
    <p>The recorded research metadata did not pass validation. A verified catalog is required to show this setup.</p>
  </section>;
  return <section className="count-catalog-preview" aria-labelledby={`${id}-title`}>
    <style>{styles}</style>
    <div className="catalog-note" role="note"><strong>Private research setup · no public count forecast</strong>
      <p>This records a model setup. It does not report current job activity.</p></div>
    <header><h2 id={`${id}-title`}>{view.title}</h2><p>{view.scopeLabel}</p></header>
    <dl className="catalog-scope">
      <dt>Prediction election</dt><dd><time dateTime={view.electionDate}>{view.electionDateLabel}</time></dd>
      <dt>Earlier elections specified for this setup</dt><dd>{view.trainingLabel}</dd>
      <dt>Information cutoff</dt><dd><time dateTime={view.informationCutoff}>{view.informationCutoffLabel}</time></dd>
    </dl>
    <div className="catalog-states" aria-label="Preparation, result, evidence and site status">
      {view.states.map(state => <section className="catalog-state" data-catalog-status={state.id} key={state.id} aria-labelledby={`${id}-${state.id}`}>
        <h3 id={`${id}-${state.id}`}>{state.label}</h3><strong>{state.value}</strong><p>{state.description}</p>
      </section>)}
    </div>
    <section aria-labelledby={`${id}-roles`}>
      <h3 id={`${id}-roles`}>Eight components, one required setup</h3>
      <p>Every component below is implemented in research and required by this recorded setup. These components form one linked setup. The catalog does not support changing them independently.</p>
      <ol className="catalog-roles">
        {view.roles.map(role => <li className="model-slice is-fixed" data-count-role={role.id} key={role.id}>
          <div><h4 id={`${id}-role-${role.id}`}>{role.label}</h4><span className="catalog-required">{role.requiredLabel}</span><p>{role.purpose}</p></div>
          <details aria-labelledby={`${id}-role-${role.id}`}><summary>Recorded implementation details</summary>
            {role.providers.map(provider => <div key={`${provider.providerId}:${provider.instanceId}`}>
              <dl><dt>Provider</dt><dd><code>{provider.providerId}</code></dd>
                <dt>Version</dt><dd>{provider.version}</dd><dt>Instance</dt><dd><code>{provider.instanceId}</code></dd>
                <dt>Recorded settings</dt><dd>{provider.options.length ? provider.options.map(([key, value]) => <div key={key}><code>{key}: {value}</code></div>) : "No adjustable settings in this recorded version."}</dd>
              </dl>
              <ul>{provider.limitations.map((limitation, index) => <li key={index}>{limitation}</li>)}</ul>
            </div>)}
          </details>
        </li>)}
      </ol>
      <p>Alternative implementations will need to work with the whole setup, be estimated together again and have their own reviewed evidence.</p>
    </section>
    <details className="catalog-details"><summary>Recorded setup details</summary>
      <dl><dt>Target</dt><dd><code>{view.technical.targetId}</code></dd>
        <dt>Prepared configuration SHA256</dt><dd><code>{view.technical.configurationSha256}</code></dd>
        <dt>Prepared request SHA256</dt><dd><code>{view.technical.requestSha256}</code></dd>
        <dt>Output contract</dt><dd><code>{view.technical.outputContractId}</code></dd>
        <dt>Recorded resolution</dt><dd><code>{view.technical.resolutionStatus}</code></dd>
      </dl>
      <p>A prepared setup does not establish that a calculation can be started now. A complete count result must be checked, with evidence reviewed separately, before it can be connected to the site.</p>
    </details>
  </section>;
}
