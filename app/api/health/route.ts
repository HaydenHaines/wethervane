import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { createHash } from "node:crypto";
import type { ExplorerRegistry } from "../../../lib/explorer-types";

export async function GET() {
  const registry: ExplorerRegistry = JSON.parse(await readFile(join(process.cwd(), "public/data/explorer/registry.json"), "utf8"));
  const entry = registry.runs.find(run => run.id === registry.defaults["2026"]);
  if (!entry || !/^[a-z0-9_-]+$/.test(entry.id) || entry.href !== `/data/explorer/${entry.id}.json`)
    return Response.json({ status: "unavailable", application: "ballotmancy" }, { status: 503 });
  const bytes = await readFile(join(process.cwd(), "public/data/explorer", `${entry.id}.json`));
  if (createHash("sha256").update(bytes).digest("hex") !== entry.sha256)
    return Response.json({ status: "unavailable", application: "ballotmancy" }, { status: 503 });
  const snapshot = JSON.parse(bytes.toString("utf8"));
  return Response.json({
    status: "ok",
    application: "ballotmancy",
    issuedAt: entry.issuedAt,
    cutoffAt: entry.cutoffAt,
    races: snapshot.races.length,
    runId: entry.id,
    resultSha256: entry.sha256,
  });
}
