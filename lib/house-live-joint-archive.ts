/** Offline archive verification only. Never import this module into a web route or browser bundle. */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { parseHouseLiveResult, type HouseLiveResult } from "./house-live-result.ts";
import { checkHouseLiveJointRows, type HouseLiveJointCheck, type HouseLiveJointCheckOptions } from "./house-live-joint-check.ts";

export interface HouseLiveJointArchiveOptions extends HouseLiveJointCheckOptions {
  /** Offline tool executable; no Python or raw matrix is required on SiteGround. */
  pythonExecutable?: string;
  signal?: AbortSignal;
}
export interface HouseLiveJointArchiveCheck extends HouseLiveJointCheck {
  artifactSha256: string;
  archiveFormat: "house_live_npz_chain_major_v1";
  seatQuantileConvention: "inverse_cdf" | "linear";
}

/**
 * Authenticate and decode the actual exporter format, then verify every winner
 * and chamber summary. The helper's successful exit is part of acceptance:
 * emitted rows alone do not establish a valid archive or completed hash check.
 */
export async function checkHouseLiveJointArchive(directory: string, input: HouseLiveResult,
  options: HouseLiveJointArchiveOptions): Promise<HouseLiveJointArchiveCheck> {
  const result = parseHouseLiveResult(input);
  if (!result) throw Error("A valid native House result is required before checking its archive.");
  if (!options || !["inverse_cdf", "linear"].includes(options.seatQuantileConvention))
    throw Error("A reviewed seat-quantile convention is required for archive checking.");
  if (typeof directory !== "string" || !directory.length) throw Error("An explicit archive source directory is required.");
  const decoder = fileURLToPath(new URL("../scripts/decode_house_live_joint.py", import.meta.url));
  const child = spawn(options.pythonExecutable ?? "python3", [decoder], {
    stdio: ["pipe", "pipe", "pipe"], signal: options.signal,
  });
  let stderr = "", stdinError: Error | null = null;
  child.stderr.on("data", (chunk: Buffer) => { stderr = (stderr + chunk.toString("utf8")).slice(-8192); });
  child.stdin.on("error", error => { stdinError = error; });
  const closed = new Promise<{ code: number | null; signal?: string | null; error?: Error }>(resolve => {
    child.once("error", error => resolve({ code: null, error }));
    child.once("close", (code, signal) => resolve({ code, signal }));
  });
  child.stdin.end(JSON.stringify({ schema: "HouseLiveJointDecodeRequestV1", directory,
    artifact: { path: result.joint_draws.path, sha256: result.joint_draws.sha256 },
    count: result.joint_draws.count, chainCount: result.joint_draws.chain_count,
    orderedRaceIds: result.joint_draws.ordered_race_ids }));

  async function* rows(): AsyncGenerator<number[]> {
    let remainder = Buffer.alloc(0);
    for await (const chunk of child.stdout) {
      const bytes = remainder.length ? Buffer.concat([remainder, chunk]) : chunk as Buffer;
      let offset = 0;
      while (offset + 435 <= bytes.length) {
        yield Array.from(bytes.subarray(offset, offset + 435));
        offset += 435;
      }
      remainder = Buffer.from(bytes.subarray(offset));
    }
    const outcome = await closed;
    if (outcome.error || outcome.code !== 0 || stdinError)
      throw Error(`House joint archive decoder failed: ${outcome.error?.message || stderr.trim() || stdinError?.message || outcome.signal || `exit ${outcome.code}`}`);
    if (remainder.length) throw Error("House joint archive decoder returned an incomplete winner row.");
  }
  try {
    const checked = await checkHouseLiveJointRows(result, rows(), options);
    return { ...checked, artifactSha256: result.joint_draws.sha256,
      archiveFormat: "house_live_npz_chain_major_v1", seatQuantileConvention: options.seatQuantileConvention };
  } finally {
    // On a numerical mismatch the reader stops early; terminate only our own helper.
    if (child.exitCode === null && child.signalCode === null) child.kill();
    await closed;
  }
}
