import { RaceCard } from "./RaceCard";
import type { SenateRaceData } from "@/lib/api";

interface RaceCardGridProps {
  races: SenateRaceData[];
  title: string;
}

/**
 * Grid of race cards, sorted by competitiveness (smallest absolute margin first).
 */
export function RaceCardGrid({ races, title }: RaceCardGridProps) {
  const sorted = [...races].sort(
    (a, b) => Math.abs(a.margin) - Math.abs(b.margin),
  );

  return (
    <section className="mb-8">
      <h2 className="font-serif text-lg font-semibold mb-3">{title}</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3 gap-3">
        {sorted.map((race) => (
          <RaceCard key={race.slug} race={race} />
        ))}
      </div>
    </section>
  );
}
