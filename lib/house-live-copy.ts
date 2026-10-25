/** Keep computation, source review, and predictive evidence distinct in both displays. */
export function houseLiveStatusLabel(value: string, kind: "input" | "numerical" | "publication" | "skill"): string {
  const normalized = value.toLowerCase().replaceAll("-", "_");
  if (kind === "input" && ["strict_asof", "strict_as_of"].includes(normalized))
    return "Inputs checked against the stated cutoff dates";
  if (normalized === "reviewed")
    return kind === "publication" ? "Reviewed for publication" : "Reviewed";
  if (normalized === "passed")
    return kind === "numerical" ? "Passed the stated calculation checks" : "Passed the stated review checks";
  const known: Record<string, string> = {
    not_for_publication: "Not approved for publication",
    not_established: "Not established",
    unobserved_2026: "The 2026 election result is not yet known, so its accuracy cannot yet be measured",
    unavailable: "Not available",
  };
  return known[normalized] ?? normalized.replaceAll("_", " ").replace(/^./, letter => letter.toUpperCase());
}

export function houseLiveCutoffRole(value: string): string {
  return ({ structural: "Electoral history and candidates", nonpoll: "Inputs other than polls",
    poll: "Polling", polling: "Polling" } as Record<string, string>)[value] ?? value.replaceAll("_", " ");
}
