/** Offline authenticated evidence. V2 research paths require a separately explicit operator storage root. */
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
  const receipts = documents.filter(document => ["HouseLivePredictionReceiptV1", "HouseLivePredictionReceiptV2"].includes(document.value?.schema));
  const plans = documents.filter(document => ["HouseLivePredictionPlanV1", "HouseLiveCombinedPredictionPlanV1", "HouseLiveCombinedPredictionPlanV2"]
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
  const segmented = plan.schema === "HouseLiveCombinedPredictionPlanV2";
  require(receipt.schema === (segmented ? "HouseLivePredictionReceiptV2" : "HouseLivePredictionReceiptV1"),
    `Result ${result.id} prediction receipt/plan storage versions disagree.`);
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
  if (segmented) {
    const storage = plan.conditional_storage;
    require(storage && Object.keys(storage).length === 3 && storage.schema === "ConditionalSegmentsManifestV1" &&
      houseLivePin(storage.manifest) && storage.manifest.path.endsWith(".json") && storage.weights === "equal_weight_per_original_joint_row" &&
      receipt.outputs.some(pin => pin.path === storage.manifest.path && pin.sha256 === storage.manifest.sha256),
      `Result ${result.id} V2 conditional manifest or original-row weights are unbound.`);
    const reviews = documents.filter(document => document.value?.schema === "HouseLivePredictionBindingReviewV2");
    require(reviews.length === 1 && reviews[0].value.status === "exact_receipt_plan_and_outputs_bound" &&
      configurationMatches(reviews[0].value.configuration, result) && houseLivePin(reviews[0].value.prediction_receipt) &&
      reviews[0].value.prediction_receipt.sha256 === receipts[0].pin.sha256 && Array.isArray(reviews[0].value.input_pins) &&
      reviews[0].value.input_pins.every(houseLivePin) && reviews[0].value.input_pins.some(pin => pin.path === storage.manifest.path && pin.sha256 === storage.manifest.sha256),
      `Result ${result.id} needs one authenticated exact V2 storage binding review.`);
    binding.conditionalStorage = { manifest: localPin(storage.manifest), plan: localPin(receipt.plan),
      receipt: localPin(reviews[0].value.prediction_receipt) };
  }
  return binding;
}
