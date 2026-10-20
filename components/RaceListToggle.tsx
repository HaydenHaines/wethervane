export default function RaceListToggle({ expanded, total, controls, onToggle }: {
  expanded: boolean; total: number; controls: string; onToggle: () => void;
}) {
  if (total <= 10) return null;
  return <button type="button" className="race-list-toggle" aria-expanded={expanded} aria-controls={controls} onClick={onToggle}>
    {expanded ? "Show fewer races" : `See all ${total} races`}
    <span aria-hidden="true">⌄</span>
  </button>;
}
