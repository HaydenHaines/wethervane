import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadConfiguredElectionNightEvaluation } from "../../lib/election-night-evaluation-server.ts";
import { parseElectionNightEvaluationView } from "../../lib/election-night-evaluation-view.ts";
import { makeElectionNightEvaluationFixture } from "./election-night-evaluation-server.mjs";

/** Invented browser/API payloads only. No production result, source or score is included. */
export async function makeElectionNightEvaluationViewFixtures() {
  const root = mkdtempSync(join(tmpdir(), "election-night-view-fixtures-"));
  const dataRoot = join(root, "data"), evaluationRoot = join(dataRoot, "election-night", "evaluation-artifacts");
  const configPath = join(dataRoot, "election-night", "evaluation-publication.json");
  const noLock = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  const fixture = makeElectionNightEvaluationFixture(evaluationRoot);
  const write = ledger => {
    mkdirSync(join(dataRoot, "election-night"), { recursive: true });
    writeFileSync(configPath, JSON.stringify({ schema: "ElectionNightEvaluationPublicationConfigV1", chambers: { senate: {
      lock: { subject: fixture.lockPin, publication: fixture.lockRequest.trustedPublication, admission: fixture.lockRequest.trustedAdmission },
      ledger,
    } } }));
  };
  const ledger = () => {
    const request = fixture.makeLedger();
    return { request, pins: { subject: request.ledger, publication: request.trustedPublication, admission: request.trustedAdmission } };
  };
  const emptyLedger = ledger(); write(emptyLedger.pins);
  const lockedNoResults = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  fixture.addRecord({ index: 0, winnerStatus: "called", marginStatus: "called", marginPp: -7 });
  const calledLedger = ledger(); write(calledLedger.pins);
  const called = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  fixture.addRecord({ index: 0, revision: "2", winnerStatus: "certified", marginStatus: "certified", marginPp: -7 });
  const correctedLedger = ledger(); write(correctedLedger.pins);
  const corrected = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  fixture.addRecord({ index: 1, winnerStatus: "certified", marginStatus: "certified", marginPp: 3 });
  const completeLedger = ledger(); write(completeLedger.pins);
  const graded = await loadConfiguredElectionNightEvaluation("senate", dataRoot);
  const views = { noLock, lockedNoResults, called, corrected, graded };
  for (const [name, view] of Object.entries(views)) {
    if (!parseElectionNightEvaluationView(view, "senate")) {
      rmSync(root, { recursive: true, force: true });
      throw Error(`Invented ${name} API fixture failed its shared browser contract.`);
    }
  }
  return { views: structuredClone(views), cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
