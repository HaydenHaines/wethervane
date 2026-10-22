import { COLORS } from "../lib/presentation";

export default function HouseGroupLegend({ mapView = "forecast", unavailableLabel }: {
  mapView?: "forecast" | "actual"; unavailableLabel?: string;
}) {
  const entries = mapView === "actual" ? [
    { label: "Democratic winner", color: COLORS.D },
    { label: "Republican winner", color: COLORS.R },
    { label: "Other winner", color: COLORS.O },
  ] : [
    { label: "Safe Democratic", color: COLORS.D },
    { label: "Likely Democratic", color: COLORS.likelyD },
    { label: "Tossup", color: COLORS.competitive },
    { label: "Likely Republican", color: COLORS.likelyR },
    { label: "Safe Republican", color: COLORS.R },
    { label: "Likely Other", color: COLORS.likelyO },
    { label: "Safe Other", color: COLORS.O },
  ];
  if (unavailableLabel) entries.push({ label: unavailableLabel, color: COLORS.inactive });
  return <div className="map-legend house-group-legend" role="list" aria-label="Group forecast color key">
    {entries.map(entry => <span role="listitem" key={entry.label}><i aria-hidden="true" className="dot" style={{ color: entry.color }} />{entry.label}</span>)}
  </div>;
}
