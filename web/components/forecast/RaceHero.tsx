"use client";

import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { RatingBadge } from "@/components/shared/RatingBadge";
import { marginToRating } from "@/lib/config/palette";
import { formatMargin } from "@/lib/format";

interface RaceHeroProps {
  raceName: string;
  stateName: string;
  raceType: string;
  year: number;
  /** 0-1 dem share (API `prediction` field) */
  prediction: number | null;
  nCounties: number;
  /** Optional 90% confidence interval bounds (0-1 dem share) */
  lo90?: number | null;
  hi90?: number | null;
}

/**
 * Hero section for a race detail page.
 *
 * Displays the large margin number, rating badge, and a 90% confidence
 * interval text line when bounds are available.
 */
export function RaceHero({
  raceName,
  stateName,
  raceType,
  year,
  prediction,
  nCounties,
  lo90,
  hi90,
}: RaceHeroProps) {
  const rating = prediction !== null ? marginToRating(prediction) : "tossup";

  const hasBounds =
    lo90 !== null && lo90 !== undefined && hi90 !== null && hi90 !== undefined;

  return (
    <div className="mb-8">
      {/* Race title */}
      <h1
        className="font-serif text-3xl lg:text-4xl mb-1"
        style={{ fontFamily: "var(--font-serif)", lineHeight: 1.2 }}
      >
        {year} {stateName} {raceType}
      </h1>
      <p className="text-sm text-muted-foreground mb-4">
        {nCounties} {nCounties === 1 ? "county" : "counties"} in model
      </p>

      {/* Margin + badge row */}
      <div className="flex items-center gap-4 flex-wrap mb-3">
        <MarginDisplay demShare={prediction} size="xl" />
        <RatingBadge rating={rating} />
      </div>

      {/* 90% confidence interval */}
      {hasBounds && (
        <p className="text-sm text-muted-foreground">
          90% interval:{" "}
          <span className="font-mono">
            {formatMargin(lo90 as number)}
          </span>{" "}
          to{" "}
          <span className="font-mono">
            {formatMargin(hi90 as number)}
          </span>
        </p>
      )}

      {/* Sub-label when no prediction */}
      {prediction === null && (
        <p className="text-sm text-muted-foreground italic">
          No model prediction available for this race yet.
        </p>
      )}

      {/* Race identifier chips */}
      <div className="flex gap-2 mt-4 flex-wrap">
        <span
          className="text-xs px-3 py-1 rounded-full border font-semibold"
          style={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text-muted)",
          }}
        >
          {raceName}
        </span>
      </div>
    </div>
  );
}
