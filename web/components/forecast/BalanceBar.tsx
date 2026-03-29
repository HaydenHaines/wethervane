"use client";

import { useRouter } from "next/navigation";
import { RATING_COLORS, RATING_LABELS } from "@/lib/config/palette";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { SenateRaceData } from "@/lib/api";

interface BalanceBarProps {
  races: SenateRaceData[];
  demSeats: number;
  gopSeats: number;
}

/** Rating sort order: most D first → tossup → most R last */
const RATING_ORDER: Record<string, number> = {
  safe_d: 0,
  likely_d: 1,
  lean_d: 2,
  tossup: 3,
  lean_r: 4,
  likely_r: 5,
  safe_r: 6,
};

/**
 * Format a margin centered at 0 (positive = Dem) as a partisan string.
 * Unlike formatMargin(), which expects a 0-1 dem share, this works directly
 * on the API's margin field where positive = Dem advantage.
 */
function formatRaceMargin(margin: number): string {
  if (Math.abs(margin) < 0.005) return "EVEN";
  const pct = (Math.abs(margin) * 100).toFixed(1);
  return margin > 0 ? `D+${pct}` : `R+${pct}`;
}

/**
 * Senate balance bar — horizontal stacked segments, one per competitive race,
 * colored by rating. Each segment is a button with a tooltip and navigates
 * to the race detail page on click.
 */
export function BalanceBar({ races, demSeats, gopSeats }: BalanceBarProps) {
  const router = useRouter();

  const sorted = [...races].sort(
    (a, b) => (RATING_ORDER[a.rating] ?? 3) - (RATING_ORDER[b.rating] ?? 3),
  );

  if (sorted.length === 0) return null;

  return (
    <TooltipProvider delay={100}>
      <div className="mb-6">
        {/* Seat counts above bar */}
        <div className="flex justify-between mb-2 text-sm font-semibold">
          <span style={{ color: RATING_COLORS.safe_d }}>{demSeats}D</span>
          <span className="text-muted-foreground">51 needed for control</span>
          <span style={{ color: RATING_COLORS.safe_r }}>{gopSeats}R</span>
        </div>

        {/* Stacked bar */}
        <div className="flex h-8 rounded-md overflow-hidden border border-[rgb(var(--color-border))]">
          {sorted.map((race) => (
            <Tooltip key={race.slug}>
              <TooltipTrigger
                render={
                  <button
                    className="h-full transition-opacity hover:opacity-80 focus:outline-none focus:ring-2 focus:ring-offset-1 min-w-[8px]"
                    style={{
                      flex: 1,
                      backgroundColor:
                        RATING_COLORS[race.rating as keyof typeof RATING_COLORS] ??
                        RATING_COLORS.tossup,
                    }}
                    onClick={() => router.push(`/forecast/${race.slug}`)}
                    aria-label={`${race.state}: ${formatRaceMargin(race.margin)}`}
                  />
                }
              />
              <TooltipContent>
                <p className="font-semibold">{race.state}</p>
                <p>
                  {formatRaceMargin(race.margin)} ·{" "}
                  {RATING_LABELS[race.rating as keyof typeof RATING_LABELS] ?? race.rating}
                </p>
                <p className="text-xs text-muted-foreground">
                  {race.n_polls} poll{race.n_polls === 1 ? "" : "s"}
                </p>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>

        {/* 50-seat midline indicator */}
        <div className="relative h-0">
          <div
            className="absolute top-[-32px] h-8 w-px bg-foreground opacity-30"
            style={{ left: "50%" }}
          />
        </div>
      </div>
    </TooltipProvider>
  );
}
