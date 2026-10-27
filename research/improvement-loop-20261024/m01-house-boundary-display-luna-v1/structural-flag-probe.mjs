import { layoutGroupRibbon } from "/home/hayden/projects/ballotmancy-house-boundary-display-20261026/lib/group-ribbon-layout.ts";

// Invented input only: a two-group modeled row mislabeled structural.
const layout = layoutGroupRibbon([{
  id: "invented",
  districtKey: "AA-01",
  probabilities: { D: 0.5, R: 0.5, O: 0 },
  supportedBlocs: ["D", "R"],
  structural: true,
}], 1);
console.log(JSON.stringify({
  accepted: layout.main.rows.length === 1,
  probabilitySource: layout.main.rows[0]?.probabilitySource,
  supportedBlocs: layout.main.rows[0]?.supportedBlocs,
}));
