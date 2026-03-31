// TypeNarrative renders the narrative/description text for an electoral type.
// This is a deliberate block-quote style — the border-left makes it visually
// distinct from the surrounding body copy without needing a separate heading.

interface TypeNarrativeProps {
  narrative: string;
}

export function TypeNarrative({ narrative }: TypeNarrativeProps) {
  return (
    <p
      style={{
        fontSize: 16,
        lineHeight: 1.7,
        color: "var(--color-text)",
        marginBottom: 32,
        borderLeft: "3px solid var(--color-border)",
        paddingLeft: 16,
      }}
    >
      {narrative}
    </p>
  );
}
