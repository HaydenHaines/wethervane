/** Private invented-input experiment. Pin consistency is not owner authentication.
 * No imports, I/O, scheduling, scientific parsing or production integration.
 */
const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
const text = x => typeof x === 'string' && x.trim().length > 0;
const strings = x => Array.isArray(x) && x.length > 0 && x.every(text);
const sha = x => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const ref = x => object(x) && text(x.id) && sha(x.sha256);
const keys = (x, expected) => object(x) && Object.keys(x).length === expected.length && expected.every(k => Object.hasOwn(x, k));
const same = (a, b, slots) => keys(a, slots) && keys(b, slots) && slots.every(k => a[k] === b[k]);
const tuple = (b, slots) => JSON.stringify([b.year, b.dateId, ...slots.map(k => b.components[k])]);
const result = (state, reason) => ({state, reason, executionAvailable: false, configurationId: null, resultId: null, run: null});

/** All inputs are untrusted runtime values. expectedOwnerPin is caller-supplied context,
 * not a credential. Catalog options must already exist; metadata cannot add options.
 * Missing metadata returns none; malformed metadata returns invalid without fallback.
 */
export function matchPreflight(metadata, catalog, request, expectedOwnerPin) {
  if (metadata === undefined) return result('none', 'No private metadata supplied.');
  const invalid = reason => result('invalid', reason);
  if (!object(catalog) || !text(catalog.id) || !object(catalog.sourceCatalog) || !sha(catalog.sourceCatalog.sha256) || !Array.isArray(catalog.slots) || !catalog.slots.length) return invalid('Invalid catalog context.');
  const slots = catalog.slots.map(s => s?.id);
  if (!slots.every(text) || new Set(slots).size !== slots.length) return invalid('Invalid catalog slots.');
  const options = new Map();
  for (const s of catalog.slots) {
    if (!Array.isArray(s.options) || !s.options.length) return invalid('Invalid catalog options.');
    const ids = s.options.map(o => o?.id);
    if (!ids.every(text) || new Set(ids).size !== ids.length || s.options.some(o => !Array.isArray(o.years) || !o.years.length || !o.years.every(Number.isInteger) || !['saved','requires_computation','unavailable'].includes(o.availability))) return invalid('Invalid catalog option declarations.');
    options.set(s.id, new Map(s.options.map(o => [o.id, o])));
  }
  const selectionValid = b => object(b) && Number.isInteger(b.year) && text(b.dateId) && keys(b.components, slots) && slots.every(k => {
    const o = options.get(k).get(b.components[k]);
    return text(b.components[k]) && o && o.availability !== 'unavailable' && o.years.includes(b.year);
  });
  if (!object(request) || request.catalogId !== catalog.id || Object.hasOwn(request, 'presetId') || !selectionValid(request)) return invalid('Invalid request; presets require the saved resolver.');
  if (!ref(expectedOwnerPin)) return invalid('An explicit expected owner pin is required.');
  if (!object(metadata) || metadata.schema !== 'ComputationPreflightMetadataV1' || metadata.executionAvailable !== false || !object(metadata.catalog) || metadata.catalog.id !== catalog.id || metadata.catalog.sourceSha256 !== catalog.sourceCatalog.sha256 || !Array.isArray(metadata.entries) || !metadata.entries.length) return invalid('Invalid metadata version, execution flag or catalog binding.');
  const ids = new Set(), tuples = new Set();
  let duplicate = false;
  const matches = [];
  for (const entry of metadata.entries) {
    if (!object(entry) || !text(entry.id) || ids.has(entry.id)) return invalid('Invalid or duplicate entry identity.');
    ids.add(entry.id);
    if (!ref(entry.ownerHandoff) || entry.ownerHandoff.id !== expectedOwnerPin.id || entry.ownerHandoff.sha256 !== expectedOwnerPin.sha256) return invalid('Owner handoff pin mismatch.');
    if (!object(entry.computation) || entry.computation.kind !== 'joint_refit' || !text(entry.computation.reason) || !strings(entry.computation.requiredSteps) || !strings(entry.outputs)) return invalid('Explicit joint refit and outputs required.');
    const e = entry.evidence;
    if (!object(e) || !['preserved','experimental','historical_reconstruction'].includes(e.status) || !['implementation','numerical','predictive','scope'].every(k => text(e[k]))) return invalid('Evidence descriptions required.');
    if (!Array.isArray(entry.bindings) || !entry.bindings.length) return invalid('Exact bindings required.');
    for (const b of entry.bindings) {
      if (!selectionValid(b) || !keys(b.componentInstances, slots) || !slots.every(k => {
        const i = b.componentInstances[k];
        return object(i) && ['providerId','version','instanceId'].every(f => text(i[f]));
      }) || !text(b.targetId) || !ref(b.law) || !ref(b.sourceManifest) || !ref(b.outputContract)) return invalid('Invalid source-bound exact binding.');
      const key = tuple(b, slots);
      if (tuples.has(key)) duplicate = true;
      tuples.add(key);
      if (b.year === request.year && b.dateId === request.dateId && same(b.components, request.components, slots)) matches.push({entry, binding: b});
    }
  }
  // Global rejection prevents an unrelated ambiguous declaration from being silently ignored.
  if (duplicate || matches.length > 1) return result('ambiguous', 'Duplicate exact bindings; no declaration selected.');
  if (!matches.length) return result('none', 'No exact year/date/component binding.');
  // Copy output so callers cannot mutate the input through a returned reference.
  const {entry, binding} = structuredClone(matches[0]);
  return {...result('match', entry.computation.reason), lifecycle: 'requires_computation', metadataEntryId: entry.id, binding, ownerHandoff: entry.ownerHandoff, computation: entry.computation, outputs: entry.outputs, evidence: entry.evidence, consistencyOnly: true};
}
