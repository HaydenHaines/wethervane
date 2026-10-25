/** Server-only artifact integrity. A matching digest does not establish forecast qualification. */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, realpath, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";

export interface HouseLiveArtifactPin {
  path: string;
  sha256: string;
}

function artifactPin(value: unknown): HouseLiveArtifactPin {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw Error("Missing live House artifact reference.");
  const pin = value as Record<string, unknown>;
  if (typeof pin.path !== "string" || typeof pin.sha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(pin.sha256) ||
    !/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(pin.path) ||
    pin.path.split("/").some(part => !part || part === "." || part === ".."))
    throw Error("Invalid live House artifact reference.");
  return { path: pin.path, sha256: pin.sha256 };
}

async function artifactPath(directory: string, pin: HouseLiveArtifactPin): Promise<string> {
  const root = await realpath(directory);
  const path = await realpath(resolve(root, pin.path));
  const within = relative(root, path);
  if (!within || isAbsolute(within) || within === ".." || within.startsWith(`..${sep}`))
    throw Error("Live House artifacts must stay inside the publication directory.");
  if (!(await stat(path)).isFile()) throw Error("Live House artifact must be a file.");
  return path;
}

/** Hash binary draws without loading the entire matrix into memory. */
export async function verifyPinnedHouseLiveArtifact(directory: string, reference: unknown): Promise<void> {
  const pin = artifactPin(reference);
  const path = await artifactPath(directory, pin);
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  if (hash.digest("hex") !== pin.sha256) throw Error("Live House artifact fingerprint mismatch.");
}

/** Parse the same bytes that were authenticated; never trust an embedded status as admission. */
export async function readPinnedHouseLiveJson(directory: string, reference: unknown): Promise<unknown> {
  const pin = artifactPin(reference);
  if (!/\.(?:geo)?json$/.test(pin.path)) throw Error("Live House JSON artifact must have a JSON or GeoJSON filename.");
  const bytes = await readFile(await artifactPath(directory, pin));
  if (createHash("sha256").update(bytes).digest("hex") !== pin.sha256)
    throw Error("Live House artifact fingerprint mismatch.");
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
}
