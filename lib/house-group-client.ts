import { groupRunId, parseHouseGroupPublication, type HouseGroupPublication } from "./house-group-catalog.ts";

export async function loadHouseGroup(id: string, signal: AbortSignal): Promise<HouseGroupPublication> {
  if (!groupRunId(id)) throw Error("This group forecast is not in the published collection.");
  const response = await fetch(`/api/chambers/house/groups?${new URLSearchParams({ run: id })}`, { signal, cache: "no-store" });
  if (!response.ok) throw Error("The House group forecast could not be loaded. Try again shortly.");
  const value = await parseHouseGroupPublication(await response.json());
  if (!value || value.result.id !== id) throw Error("The returned group forecast does not match the requested result.");
  return value;
}
