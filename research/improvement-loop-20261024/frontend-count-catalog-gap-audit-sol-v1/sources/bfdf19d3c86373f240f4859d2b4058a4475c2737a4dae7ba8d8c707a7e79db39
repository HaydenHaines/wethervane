import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { validateExplorerBundle } from "../../../../lib/explorer-state";
import { resolveComposition } from "../../../../lib/composition-resolver";
import type { CompositionCatalog, CompositionRequest } from "../../../../lib/composition-types";
import type { ExplorerBundle, ExplorerRegistry } from "../../../../lib/explorer-types";

export const runtime = "nodejs";

const MAX_BODY_BYTES = 8 * 1024;
const MAX_COMPONENTS = 12;
const BODY_KEYS = new Set(["catalogId", "year", "dateId", "components", "presetId"]);

class BadRequest extends Error {}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new Error("Cannot canonicalize configuration identity.");
  return encoded;
}

async function boundedJson(request: Request): Promise<unknown> {
  const lengthHeader = request.headers.get("content-length");
  if (lengthHeader && Number(lengthHeader) > MAX_BODY_BYTES) throw new BadRequest("Request body is too large.");
  const contentType = request.headers.get("content-type");
  if (contentType && !contentType.toLowerCase().includes("application/json")) {
    throw new BadRequest("Send a JSON composition request.");
  }
  if (!request.body) throw new BadRequest("A JSON composition request is required.");

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > MAX_BODY_BYTES) {
        await reader.cancel();
        throw new BadRequest("Request body is too large.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new BadRequest("Request body is not valid JSON.");
  }
}

function parseRequest(value: unknown): CompositionRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new BadRequest("Request must be a JSON object.");
  }
  const body = value as Record<string, unknown>;
  if (Object.keys(body).some((key) => !BODY_KEYS.has(key))) {
    throw new BadRequest("Client-supplied model bindings and unknown fields are not accepted.");
  }
  if (typeof body.catalogId !== "string" || body.catalogId.length < 1 || body.catalogId.length > 128 ||
      !Number.isInteger(body.year) || (body.year as number) < 1900 || (body.year as number) > 2200 ||
      typeof body.dateId !== "string" || body.dateId.length < 1 || body.dateId.length > 80 ||
      (body.presetId !== undefined && (typeof body.presetId !== "string" || body.presetId.length > 160))) {
    throw new BadRequest("Request needs a catalog ID, election year, date ID and component choices.");
  }
  if (!body.components || typeof body.components !== "object" || Array.isArray(body.components)) {
    throw new BadRequest("Component choices must be an object.");
  }
  const components = body.components as Record<string, unknown>;
  if (Object.keys(components).length > MAX_COMPONENTS ||
      Object.values(components).some((choice) => typeof choice !== "string" || choice.length > 160)) {
    throw new BadRequest("Component choices are malformed.");
  }
  return {
    catalogId: body.catalogId,
    year: body.year as number,
    dateId: body.dateId,
    components: components as Record<string, string>,
    ...(typeof body.presetId === "string" ? { presetId: body.presetId } : {}),
  };
}

async function verifyReadyArtifact(
  root: string,
  catalog: CompositionCatalog,
  resolution: ReturnType<typeof resolveComposition>,
): Promise<void> {
  if (resolution.state !== "ready" || !resolution.run || !resolution.configuration) return;
  const href = resolution.run.href;
  const prefix = "/data/explorer/";
  if (!href.startsWith(prefix) || href.includes("..") || href.includes("\\")) {
    throw new Error("Saved forecast path is invalid.");
  }
  const publicRoot = resolve(root, "public");
  const artifactPath = resolve(publicRoot, `.${href}`);
  if (artifactPath !== publicRoot && !artifactPath.startsWith(publicRoot + sep)) {
    throw new Error("Saved forecast path escaped the public data directory.");
  }
  const bytes = await readFile(artifactPath);
  const actualHash = createHash("sha256").update(bytes).digest("hex");
  if (actualHash !== resolution.run.sha256 || actualHash !== resolution.configuration.resultSha256) {
    throw new Error("Saved forecast result hash does not match the manifest.");
  }
  const expectedConfigurationId = `cfg-${createHash("sha256").update(canonicalJson({
    catalogSha256: catalog.sourceCatalog.sha256,
    run: resolution.run,
    resultSha256: resolution.configuration.resultSha256,
  })).digest("hex")}`;
  if (resolution.configuration.id !== expectedConfigurationId) {
    throw new Error("Saved configuration identity does not match its source catalog and result.");
  }
  const bundle = JSON.parse(bytes.toString("utf8")) as ExplorerBundle;
  if (!validateExplorerBundle(bundle, resolution.run) ||
      bundle.run.year !== resolution.configuration.year ||
      bundle.run.dateId !== resolution.configuration.dateId ||
      bundle.run.lawId !== resolution.configuration.lawId ||
      bundle.run.targetId !== resolution.configuration.targetId ||
      bundle.run.components.base !== resolution.configuration.components.base ||
      bundle.run.components.polling !== resolution.configuration.components.polling ||
      bundle.run.components.uncertainty !== resolution.configuration.components.uncertainty) {
    throw new Error("Saved forecast scientific metadata does not match the resolved configuration.");
  }
  if (resolution.request.presetId) {
    const preset = catalog.presets.find((item) => item.id === resolution.request.presetId);
    if (!preset || bundle.provenance.sources.forecastSha256 !== preset.forecastSha256) {
      throw new Error("Preserved preset source fingerprint does not match its saved forecast bundle.");
    }
  }
}

export async function POST(request: Request): Promise<Response> {
  let compositionRequest: CompositionRequest;
  try {
    compositionRequest = parseRequest(await boundedJson(request));
  } catch (error) {
    const message = error instanceof BadRequest ? error.message : "Invalid request.";
    return Response.json({ error: message }, { status: 400 });
  }

  try {
    const root = process.cwd();
    const [catalogText, registryText] = await Promise.all([
      readFile(resolve(root, "public/data/explorer/capabilities.json"), "utf8"),
      readFile(resolve(root, "public/data/explorer/registry.json"), "utf8"),
    ]);
    const catalog = JSON.parse(catalogText) as CompositionCatalog;
    const registry = JSON.parse(registryText) as ExplorerRegistry;
    if (catalog.schemaVersion !== 1 || registry.schemaVersion !== 1 ||
        !Array.isArray(catalog.saved) || !Array.isArray(registry.runs)) {
      throw new Error("Deployment catalog schema is invalid.");
    }
    const resolution = resolveComposition(catalog, registry, compositionRequest);
    await verifyReadyArtifact(root, catalog, resolution);
    return Response.json(resolution);
  } catch (error) {
    console.error("Composition resolver deployment data failed validation:", error);
    return Response.json({ error: "The saved composition catalog is temporarily unavailable." }, { status: 500 });
  }
}
