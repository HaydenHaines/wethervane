/** INVENTED mixed-edition metadata around real software fixtures; never a scientific forecast export. */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
const read = path => JSON.parse(readFileSync(path, "utf8"));
export function mixedSenateEditionFixture() {
  const registry = read("public/data/explorer/registry.json"), catalog = read("public/data/explorer/capabilities.json");
  const bundles = new Map(), bytes = new Map();
  catalog.id += ":INVENTED-mixed-edition";
  for (const entry of registry.runs) {
    const bundle = read(`public${entry.href}`);
    if (entry.year === 2026 && entry.components.polling === "standard") {
      const dated = { dateId: "2026-10-26", dateLabel: "October 26, 2026", forecastAt: "2026-10-26T12:00:00Z",
        issuedAt: "2026-10-26T13:00:00Z", cutoffAt: "2026-10-26T12:00:00Z" };
      Object.assign(entry, dated); Object.assign(bundle.run, dated);
    }
    const raw = JSON.stringify(bundle); entry.sha256 = createHash("sha256").update(raw).digest("hex");
    bytes.set(entry.href, raw); bundles.set(entry.id, bundle);
    const saved = catalog.saved.find(c => c.runId === entry.id);
    if (saved) {
      saved.dateId = entry.dateId; saved.resultSha256 = entry.sha256;
      // These are test-only saved identities consumed by the pure resolver.
      saved.id = `cfg-${createHash("sha256").update(`INVENTED:${entry.id}:${entry.sha256}`).digest("hex")}`;
    }
  }
  return { registry, catalog, bundles, bytes };
}
