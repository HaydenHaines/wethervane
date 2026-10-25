import { parseHouseLivePublication, type HouseLivePublication } from "./house-live-publication.ts";
import { houseLiveRunId } from "./house-live-navigation.ts";

export interface HouseLiveRequest {
  runId: string | null;
  /** Guard a choice made in the displayed catalog if publication changes during loading. */
  expectedResultSha256?: string;
}

export async function loadHouseLive(request: HouseLiveRequest, signal: AbortSignal): Promise<HouseLivePublication> {
  if (request.runId !== null && !houseLiveRunId(request.runId))
    throw Error("This House model selection is invalid.");
  if (request.expectedResultSha256 !== undefined &&
    (!request.runId || !/^[a-f0-9]{64}$/.test(request.expectedResultSha256)))
    throw Error("The selected House result reference is invalid.");
  const query = request.runId ? `?${new URLSearchParams({ run: request.runId })}` : "";
  const response = await fetch(`/api/chambers/house/live${query}`, { signal, cache: "no-store" });
  if (!response.ok) throw Error(response.status === 400
    ? "This House model is not available in the published collection."
    : "The current House forecast could not be loaded. Try again shortly.");
  const publication = parseHouseLivePublication(await response.json());
  if (!publication) throw Error("The current House response could not be verified.");
  const expectedId = request.runId ?? publication.catalog.defaultId;
  if (publication.view.run.id !== expectedId)
    throw Error("The returned House forecast does not match the selected model.");
  if (request.expectedResultSha256 && publication.view.run.resultSha256 !== request.expectedResultSha256)
    throw Error("The published forecast changed while loading. Reload the current edition before choosing another model.");
  return publication;
}
