/** Offline authenticated local evidence. Never follow receipt-internal research paths. */
import { readPinnedHouseLiveJson } from "../lib/house-live-artifacts.ts";
import { houseLivePin } from "../lib/house-live-result.ts";

const require = (condition, message) => { if (!condition) throw Error(message); };
const positiveInteger = value => Number.isSafeInteger(value) && value > 0;
const nonnegativeInteger = value => Number.isSafeInteger(value) && value >= 0;
const configurationMatches = (value, result) => value && typeof value === "object" &&
  Object.keys(value).length === 4 && ["id", "base", "polling", "uncertainty"].every(key =>
    value[key] === (key === "id" ? result.id : result.configuration[key]));
const localPin = pin => ({ path: pin.path, sha256: pin.sha256 });

/** Retain every evidence occurrence so duplicate aliases cannot hide ambiguous mappings. */
export async function readHouseLivePredictionEvidence(directory, result, cache = new Map()) {
  const documents = [];
  for (const pin of result.qualification.evidence) {
    const key = JSON.stringify([directory, pin.path, pin.sha256]);
    if (!cache.has(key)) cache.set(key, await readPinnedHouseLiveJson(directory, pin));
    documents.push({ pin, value: cache.get(key) });
  }
  return documents;
}

/** Caller authenticates these local documents first. No mapping is permitted for legacy generic evidence. */
export function resolveHouseLivePredictionPlan(result, documents) {
  const receipts = documents.filter(document => document.value?.schema === "HouseLivePredictionReceiptV1");
  const plans = documents.filter(document => ["HouseLivePredictionPlanV1", "HouseLiveCombinedPredictionPlanV1"]
    .includes(document.value?.schema));
  if (!receipts.length && !plans.length) return undefined;
  require(receipts.length === 1 && plans.length === 1,
    `Result ${result.id} has a missing or ambiguous local prediction receipt/plan mapping.`);
  const receipt = receipts[0].value;
  require(receipt.status === "passed" && configurationMatches(receipt.configuration, result) &&
    houseLivePin(receipt.plan) && Array.isArray(receipt.outputs) && receipt.outputs.every(houseLivePin) &&
    receipt.outputs.filter(pin => pin.sha256 === result.joint_draws.sha256).length === 1,
  `Result ${result.id} prediction receipt does not bind its reviewed configuration and joint archive.`);
  const matches = plans.filter(document => document.pin.sha256 === receipt.plan.sha256);
  require(matches.length === 1, `Result ${result.id} prediction receipt has no unique authenticated local plan.`);
  const { pin, value: plan } = matches[0];
  require(configurationMatches(plan.configuration, result) && plan.pilot === false &&
    positiveInteger(plan.settings?.chains) && positiveInteger(plan.settings?.draws) &&
    plan.settings.chains === result.joint_draws.chain_count &&
    plan.settings.chains * plan.settings.draws === result.joint_draws.count,
  `Result ${result.id} prediction plan configuration or chain/draw counts disagree with its native result.`);
  const binding = { kind: plan.schema === "HouseLivePredictionPlanV1" ? "original" : "combined",
    plan: localPin(pin), receipt: localPin(receipts[0].pin) };
  if (binding.kind === "original") return binding;
  require(Array.isArray(plan.source_runs) && plan.source_runs.length >= 2 &&
    nonnegativeInteger(plan.settings.warmup_per_source), `Result ${result.id} has invalid combined prediction lineage.`);
  let rows = 0, chains = 0;
  const seeds = new Set();
  for (const source of plan.source_runs) {
    const settings = source?.settings;
    require(houseLivePin(source?.receipt) && houseLivePin(source?.plan) &&
      positiveInteger(settings?.chains) && settings.draws === plan.settings.draws &&
      settings.warmup === plan.settings.warmup_per_source && nonnegativeInteger(settings.seed) &&
      !seeds.has(settings.seed) && source.joint_draw_offset === rows && source.predictive_chain_offset === chains,
    `Result ${result.id} has inconsistent combined source chain/draw lineage.`);
    seeds.add(settings.seed);
    rows += settings.chains * settings.draws;
    chains += settings.chains;
  }
  require(rows === result.joint_draws.count && chains === result.joint_draws.chain_count,
    `Result ${result.id} combined source lengths disagree with its native result.`);
  return binding;
}
