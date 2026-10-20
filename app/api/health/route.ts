import { readFile } from "node:fs/promises";
import { join } from "node:path";

export async function GET() {
  const snapshot = JSON.parse(
    await readFile(join(process.cwd(), "public/data/forecast.json"), "utf8"),
  );
  return Response.json({
    status: "ok",
    application: "ballotmancy",
    issuedAt: snapshot.issuedAt,
    cutoffAt: snapshot.cutoffAt,
    races: snapshot.models.polling.races.length,
  });
}
