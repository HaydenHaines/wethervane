/** Private offline weighted-stage verification; never expose raw draws on the site. */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { houseLivePin, type HouseLivePin } from "./house-live-result.ts";
import { checkWeightedJointRows, type WeightedCheck, type WeightedResearchResult, type WeightedRow } from "./house-weighted-research-check.ts";

export interface WeightedStageOptions {
  receipt: HouseLivePin;
  configuration: { id: string; base: string; polling: string; uncertainty: string };
  candidateEdition: string;
  pythonExecutable?: string;
  signal?: AbortSignal;
}

/** Authenticate a stage receipt and both files, then independently check all weighted summaries.
 * This research verification does not admit its inputs or authorize publication. */
export async function checkWeightedResearchStage(directory: string, result: WeightedResearchResult,
  options: WeightedStageOptions): Promise<WeightedCheck & { receiptSha256: string; publicationAdmission: false }> {
  if (typeof directory !== "string" || !directory.length || !options || !houseLivePin(options.receipt))
    throw Error("Explicit directory and stage receipt pin are required.");
  if (!options.configuration || typeof options.candidateEdition !== "string" || !options.candidateEdition.length)
    throw Error("Explicit configuration and candidate edition are required.");
  const joint = result.joint_draws;
  const decoder = fileURLToPath(new URL("../scripts/decode_house_weighted_stage.py", import.meta.url));
  const child = spawn(options.pythonExecutable ?? "python3", [decoder], { stdio: ["pipe", "pipe", "pipe"], signal: options.signal });
  let stderr = "", stdinError: Error | null = null;
  child.stderr.on("data", (chunk: Buffer) => { stderr = (stderr + chunk.toString("utf8")).slice(-8192); });
  child.stdin.on("error", error => { stdinError = error; });
  const closed = new Promise<{ code: number | null; signal?: string | null; error?: Error }>(resolve => {
    child.once("error", error => resolve({ code: null, error }));
    child.once("close", (code, signal) => resolve({ code, signal }));
  });
  child.stdin.end(JSON.stringify({ schema: "HouseWeightedStageDecodeRequestV1", directory,
    receipt: options.receipt, count: joint.count, chainCount: joint.chain_count,
    orderedRaceIds: joint.ordered_race_ids, configuration: options.configuration, candidateEdition: options.candidateEdition }));
  async function* rows(): AsyncGenerator<WeightedRow> {
    let remainder = Buffer.alloc(0), index = 0;
    const perChain = joint.count / joint.chain_count, frame = 443;
    for await (const chunk of child.stdout) {
      const bytes = remainder.length ? Buffer.concat([remainder, chunk]) : chunk as Buffer;
      let offset = 0;
      while (offset + frame <= bytes.length) {
        yield { jointDrawId: index, predictiveChain: Math.floor(index / perChain), predictiveDraw: index % perChain,
          winnerCodes: Array.from(bytes.subarray(offset, offset + 435)), logFactor: bytes.readDoubleLE(offset + 435) };
        offset += frame; index++;
      }
      remainder = Buffer.from(bytes.subarray(offset));
    }
    const outcome = await closed;
    if (outcome.error || outcome.code !== 0 || stdinError)
      throw Error(`Weighted stage decoder failed: ${outcome.error?.message || stderr.trim() || stdinError?.message || outcome.signal || `exit ${outcome.code}`}`);
    if (remainder.length) throw Error("Weighted stage decoder returned an incomplete row.");
  }
  try {
    const checked = await checkWeightedJointRows(result, rows());
    return { ...checked, receiptSha256: options.receipt.sha256, publicationAdmission: false };
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill();
    await closed;
  }
}
