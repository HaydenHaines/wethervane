/** Offline, selected-case preparation only. No website publication or source permission inference. */
import { createHash } from "node:crypto";
import { readFile, realpath, mkdir, writeFile, stat } from "node:fs/promises";
import { resolve, relative, isAbsolute, sep, dirname } from "node:path";
import { parseHouseLiveCatalog } from "../lib/house-live-catalog.ts";
import { parseHouseLiveResult } from "../lib/house-live-result.ts";
import { projectHouseContextComparisonCase } from "../lib/house-context-comparison.ts";

const need = (v, message) => { if (!v) throw Error(message); };
const object = v => !!v && typeof v === "object" && !Array.isArray(v);
const hash = v => typeof v === "string" && /^[a-f0-9]{64}$/.test(v);
const text = v => typeof v === "string" && v.length > 0 && v.length <= 4000 && v.trim() === v;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const bytesOf = value => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const inside = (root, path) => { const part = relative(root, path); return !isAbsolute(part) && part !== ".." && !part.startsWith(`..${sep}`); };
const pin = p => object(p) && typeof p.path === "string" &&
  /^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(p.path) && p.path.split("/").every(s => s && s !== "." && s !== "..") && hash(p.sha256);
const eqPin = (a, b) => pin(a) && pin(b) && a.path === b.path && a.sha256 === b.sha256;
const hasHash = (items, sha) => Array.isArray(items) && items.some(p => pin(p) && p.sha256 === sha);

async function reader(root) {
  const realRoot = await realpath(root);
  const fileBytes = async p => {
    need(pin(p), "Invalid relative artifact pin");
    const path = await realpath(resolve(realRoot, p.path));
    need(path !== realRoot && inside(realRoot, path), "Artifact escapes its declared trust root");
    const file = await stat(path);
    need(file.isFile() && file.size <= 64 * 1024 * 1024, "Artifact is not a bounded file");
    const bytes = await readFile(path);
    need(digest(bytes) === p.sha256, `Artifact SHA differs: ${p.path}`);
    return bytes;
  };
  const fileJson = async p => JSON.parse((await fileBytes(p)).toString("utf8"));
  fileJson.bytes = fileBytes;
  return fileJson;
}

function validateApproval(approval, policy, options, handoff, review) {
  need(approval.schema === "HouseContextExcerptOperatorApprovalV1" &&
    approval.status === "approved_for_offline_normalization" &&
    approval.scope === "private_selected_excerpts_only" &&
    approval.handoffSha256 === options.handoff.sha256 &&
    approval.handoffReviewSha256 === options.handoffReview.sha256 &&
    approval.comparisonReceiptSha256 === handoff.comparison_receipt.sha256 &&
    same(approval.currentReadyPacket, handoff.current_ready_packet) &&
    same(approval.baselineReadyPacket, handoff.baseline_ready_packet) &&
    object(approval.currentEdition) && object(approval.baselineEdition),
  "Independent operator approval does not bind this comparison/edition");
  need(policy.schema === "HouseContextExcerptSourcePolicyV1" &&
    policy.status === "approved_for_private_excerpts" &&
    policy.distribution === "private_research_only" &&
    policy.handoffSha256 === options.handoff.sha256 &&
    policy.sourceUrl === handoff.source_url && policy.cookSnapshot === handoff.cook_snapshot &&
    Array.isArray(policy.permittedDistricts) && policy.permittedDistricts.length > 0 &&
    policy.permittedDistricts.length <= 7 && new Set(policy.permittedDistricts).size === policy.permittedDistricts.length &&
    policy.permittedDistricts.every(k => handoff.cases.some(c => c.district_key === k)) &&
    Array.isArray(approval.selectedDistricts) && same(approval.selectedDistricts, policy.permittedDistricts) &&
    review.handoff.sha256 === options.handoff.sha256,
  "Independent source-policy permission is absent or includes unapproved excerpts");
  need(approval.selectedDistricts.length < 435 && approval.publicationQualified === false &&
    policy.publicDisplayApproved === false, "Public use is outside this importer");
  if (!handoff.self_comparison) {
    need(object(approval.baselineModelByDistrict) &&
      same(Object.keys(approval.baselineModelByDistrict).sort(), [...approval.selectedDistricts].sort()) &&
      Object.values(approval.baselineModelByDistrict).every(text),
    "Non-self baseline run mapping requires exact independent operator approval");
  }
}

async function edition({ ownerRead, siteRead, approval, readyPin, handoff, role }) {
  need(pin(approval.publication) && pin(approval.catalog) && pin(approval.result) &&
    text(approval.runId) && eqPin(approval.readyPacket, readyPin), `${role} operator edition pins missing`);
  const [ready, manifest, rawCatalog, rawResult] = await Promise.all([
    ownerRead(readyPin), siteRead(approval.publication), siteRead(approval.catalog), siteRead(approval.result),
  ]);
  need(ready.schema === "HouseLiveReadyPacketV1" && ready.status === "scientific_packet_ready_for_website_review" &&
    ready.publication_qualified === false && pin(ready.catalog) && ready.catalog.sha256 === approval.catalog.sha256,
  `${role} native ready packet/catalog differs`);
  need(manifest.schema === "HouseLiveSiteManifestV1" && manifest.status === "available" &&
    manifest.cycle === 2026 && pin(manifest.catalog) && manifest.catalog.sha256 === approval.catalog.sha256 &&
    Array.isArray(manifest.results) && manifest.results.some(row => row.id === approval.runId &&
      pin(row.artifact) && row.artifact.sha256 === approval.result.sha256),
  `${role} publication does not carry exact catalog/result`);
  const catalog = parseHouseLiveCatalog(rawCatalog), result = parseHouseLiveResult(rawResult);
  const entry = catalog?.entries.find(row => row.id === approval.runId);
  need(catalog && entry && entry.result.sha256 === approval.result.sha256 &&
    result && result.id === approval.runId && result.configuration.base === entry.configuration.base &&
    result.configuration.polling === entry.configuration.polling &&
    result.configuration.uncertainty === entry.configuration.uncertainty &&
    result.map_vintage === handoff.map_vintage,
  `${role} native catalog/result identity differs`);
  return { result, identity: { publicationSha256: approval.publication.sha256,
    catalogSha256: approval.catalog.sha256, readyPacketSha256: readyPin.sha256,
    runId: result.id, resultSha256: approval.result.sha256, issuedAt: result.issued_at,
    cutoffAt: result.forecast_cutoff, mapVintage: result.map_vintage,
    geographySha256: result.geography.sha256 } };
}

/** All trust pins are caller supplied. Owner handoff/review cannot approve themselves. */
export async function importHouseContextComparison(options) {
  need(object(options) && [options.ownerRoot, options.currentSiteRoot, options.baselineSiteRoot,
    options.operatorRoot, options.sourcePolicyRoot, options.outputDir].every(text) &&
    [options.handoff, options.handoffReview, options.operatorApproval, options.sourcePolicyApproval].every(pin),
  "Explicit roots and trusted pins required");
  const roots = await Promise.all([options.ownerRoot, options.currentSiteRoot, options.baselineSiteRoot,
    options.operatorRoot, options.sourcePolicyRoot].map(path => realpath(path)));
  const [ownerRoot, currentRoot, baselineRoot, operatorRoot, policyRoot] = roots;
  const projectRoot = await realpath(new URL("../", import.meta.url).pathname);
  need(!inside(ownerRoot, operatorRoot) && !inside(operatorRoot, ownerRoot) &&
    !inside(ownerRoot, policyRoot) && !inside(policyRoot, ownerRoot) &&
    !inside(currentRoot, operatorRoot) && !inside(operatorRoot, currentRoot) &&
    !inside(currentRoot, policyRoot) && !inside(policyRoot, currentRoot) &&
    !inside(baselineRoot, operatorRoot) && !inside(operatorRoot, baselineRoot) &&
    !inside(baselineRoot, policyRoot) && !inside(policyRoot, baselineRoot) &&
    !inside(projectRoot, operatorRoot) && !inside(projectRoot, policyRoot) &&
    !inside(operatorRoot, policyRoot) && !inside(policyRoot, operatorRoot),
  "Approval trust roots must be separate from producer, website and each other");
  const [ownerRead, currentRead, baselineRead, operatorRead, policyRead] = await Promise.all(roots.map(reader));
  const [handoff, review, approval, policy] = await Promise.all([
    ownerRead(options.handoff), ownerRead(options.handoffReview),
    operatorRead(options.operatorApproval), policyRead(options.sourcePolicyApproval),
  ]);
  need(handoff.schema === "HouseCookPriorityCasesHandoffV1" &&
    handoff.status === "private_research_preparation_not_website_admission" &&
    handoff.publication_qualified === false && handoff.cycle === 2026 &&
    handoff.map_vintage === "house_election_map_2026" &&
    Array.isArray(handoff.cases) && handoff.cases.length > 0 && handoff.cases.length <= 7 &&
    new Set(handoff.cases.map(c => c.district_key)).size === handoff.cases.length &&
    handoff.contract?.cook_scale === "ordinal_categories_without_assigned_probabilities" &&
    handoff.contract?.missing === "null_is_inapplicable_not_zero" &&
    pin(handoff.current_ready_packet) && pin(handoff.baseline_ready_packet) &&
    pin(handoff.comparison_receipt) && Array.isArray(handoff.input_pins),
  "Invalid bounded owner handoff");
  need(review.schema === "HouseCookPriorityCasesHandoffReviewV1" &&
    review.status === "passed_for_private_frontend_preparation" &&
    review.publication_qualified === false && review.handoff?.sha256 === options.handoff.sha256,
  "Owner handoff review differs");
  validateApproval(approval, policy, options, handoff, review);
  for (const p of handoff.input_pins) await ownerRead.bytes(p);
  const receipt = await ownerRead(handoff.comparison_receipt);
  need(receipt.schema === "HouseReviewedCookComparisonV2" &&
    receipt.status === "research_triage_not_forecast_evaluation" && receipt.publication_qualified === false &&
    eqPin(receipt.current_ready_packet, handoff.current_ready_packet) &&
    eqPin(receipt.baseline_ready_packet, handoff.baseline_ready_packet) &&
    receipt.cook_snapshot === handoff.cook_snapshot &&
    Array.isArray(receipt.input_pins), "Comparison receipt differs from handoff");
  const [current, baseline] = await Promise.all([
    edition({ ownerRead, siteRead: currentRead, approval: approval.currentEdition,
      readyPin: handoff.current_ready_packet, handoff, role: "Current" }),
    edition({ ownerRead, siteRead: baselineRead, approval: approval.baselineEdition,
      readyPin: handoff.baseline_ready_packet, handoff, role: "Baseline" }),
  ]);
  need(hasHash(receipt.input_pins, current.identity.catalogSha256) &&
    hasHash(receipt.input_pins, baseline.identity.catalogSha256) &&
    hasHash(receipt.input_pins, current.identity.resultSha256) &&
    hasHash(receipt.input_pins, baseline.identity.resultSha256),
  "Reviewed comparison receipt omits native edition artifacts");
  need(current.identity.mapVintage === baseline.identity.mapVintage &&
    current.identity.geographySha256 === baseline.identity.geographySha256,
  "Comparison editions use different district maps");
  need(handoff.self_comparison === eqPin(handoff.current_ready_packet, handoff.baseline_ready_packet),
  "Self-comparison marker differs from ready-packet identities");
  if (handoff.self_comparison) need(same(current.identity, baseline.identity),
    "Self-comparison must bind one exact published result");
  const cases = policy.permittedDistricts.map(key => {
    const raw = handoff.cases.find(c => c.district_key === key);
    need(raw && (handoff.self_comparison ||
      ((raw.baseline_model_id === undefined || raw.baseline_model_id === approval.baselineModelByDistrict[key]) &&
        approval.baselineModelByDistrict[key] === baseline.result.id)),
    "Non-self comparison requires an explicit reviewed baseline run ID");
    return projectHouseContextComparisonCase(raw, current.result, baseline.result);
  });
  const catalog = { schema: "HouseContextComparisonCatalogV1", publicationQualified: false,
    scope: "private_selected_excerpts_only", selfComparison: handoff.self_comparison,
    cookSnapshot: handoff.cook_snapshot, sourceUrl: handoff.source_url,
    sourcePins: { handoffSha256: options.handoff.sha256, handoffReviewSha256: options.handoffReview.sha256,
      comparisonReceiptSha256: handoff.comparison_receipt.sha256,
      operatorApprovalSha256: options.operatorApproval.sha256,
      sourcePolicyApprovalSha256: options.sourcePolicyApproval.sha256 },
    currentEdition: current.identity, baselineEdition: baseline.identity, cases,
    interpretation: "Private selected old-release context only. An ordinal external rating is not a probability, observed error, or causal explanation; this file is not approved for public display." };
  const outputParent = await realpath(dirname(resolve(options.outputDir)));
  const output = resolve(outputParent, options.outputDir.split(/[\\/]/).at(-1));
  need(!roots.some(root => inside(root, output)) && !inside(projectRoot, output),
    "Output must be outside producer, site, approval and website roots");
  await mkdir(output); // exclusive: never overwrite a dated preparation artifact
  const catalogBytes = bytesOf(catalog);
  const manifest = { schema: "HouseContextComparisonPrivateManifestV1", publicationQualified: false,
    externalActivation: false, catalog: { path: "catalog.json", sha256: digest(catalogBytes) },
    currentResultSha256: current.identity.resultSha256, baselineResultSha256: baseline.identity.resultSha256,
    approvedExcerptCount: cases.length, operatorApprovalSha256: options.operatorApproval.sha256,
    sourcePolicyApprovalSha256: options.sourcePolicyApproval.sha256 };
  await writeFile(resolve(output, "catalog.json"), catalogBytes, { flag: "wx" });
  await writeFile(resolve(output, "manifest.json"), bytesOf(manifest), { flag: "wx" });
  return { output, catalog, manifest };
}

if (process.argv[1] && resolve(process.argv[1]) === new URL(import.meta.url).pathname) {
  const file = process.argv[2];
  need(file, "Usage: node scripts/import_house_context_comparison.mjs <absolute-options.json>");
  importHouseContextComparison(JSON.parse(await readFile(file, "utf8")))
    .then(({ output, manifest }) => process.stdout.write(`${JSON.stringify({ output, manifest })}\n`))
    .catch(error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
}
