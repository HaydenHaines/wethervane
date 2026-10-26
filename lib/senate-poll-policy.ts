import type { ExplorerBundle } from "./explorer-types.ts";

interface PollReceiptIdentity {
  cutoffAt: string; applicableRuns: Record<string, string>;
  records: { raceId: string; status: string }[];
}

/** A historical receipt must never explain another edition that reused its run ID. */
export function senatePollReceiptFor<T extends PollReceiptIdentity>(receipts: readonly T[], bundle: ExplorerBundle, digest: string): T | null {
  if (bundle.run.historical || bundle.run.components.polling === "none") return null;
  return receipts.find(receipt => {
    if (receipt.applicableRuns[bundle.run.id] !== digest || receipt.cutoffAt !== bundle.run.cutoffAt) return false;
    const counts = new Map<string, number>();
    for (const row of receipt.records) if (row.status === "included") counts.set(row.raceId, (counts.get(row.raceId) ?? 0) + 1);
    return [...counts.keys()].every(id => bundle.races.some(race => race.id === id)) &&
      bundle.races.every(race => race.pollCount === (counts.get(race.id) ?? 0));
  }) ?? null;
}
