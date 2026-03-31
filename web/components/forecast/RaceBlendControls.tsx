"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { SectionWeightSliders, SectionWeights } from "./SectionWeightSliders";
import { RaceHero } from "./RaceHero";
import dynamic from "next/dynamic";
import { Skeleton } from "@/components/ui/skeleton";

const QuantileDotplot = dynamic(
  () =>
    import("@/components/forecast/QuantileDotplot").then(
      (m) => m.QuantileDotplot,
    ),
  {
    ssr: false,
    loading: () => <Skeleton className="w-full h-[160px]" />,
  },
);

// How long to wait after the last slider move before firing the API call.
const DEBOUNCE_MS = 400;

// The default std to fall back on when the API doesn't provide one.
const FALLBACK_STD = 0.065;

interface BlendResult {
  prediction: number | null;
  pred_std: number | null;
  pred_lo90: number | null;
  pred_hi90: number | null;
}

interface RaceBlendControlsProps {
  /** URL slug for the race — used to call POST /forecast/race/{slug}/blend */
  slug: string;
  /** API base URL (from NEXT_PUBLIC_API_URL env) */
  apiBase: string;
  /** Initial forecast values from SSR — displayed before any slider interaction */
  initialPrediction: number | null;
  initialPredStd: number | null;
  initialLo90: number | null;
  initialHi90: number | null;
  /** Whether polls exist; determines default weights and whether sliders appear */
  hasPolls: boolean;
  /** State-level prior labels shown in the blend section description */
  statePredLocal?: number | null;
  statePredNational?: number | null;
  /** Passed through to RaceHero for static display */
  stateName: string;
  raceType: string;
  year: number;
  nCounties: number;
  nPolls: number;
}

/**
 * Client component that owns the blend-slider state for a race detail page.
 *
 * Renders three sections in order:
 *   1. RaceHero — live margin, rating badge, CI bounds
 *   2. Outcome Distribution — dotplot derived from current blend values
 *   3. Forecast Blend — section weight sliders (only when polls exist)
 *
 * The SSR page renders the static sections (polls, types, about) after this.
 *
 * Loading state: a subtle opacity fade on the hero and dotplot while an
 * API call is in-flight.  On error, the previous good values are retained.
 */
export function RaceBlendControls({
  slug,
  apiBase,
  initialPrediction,
  initialPredStd,
  initialLo90,
  initialHi90,
  hasPolls,
  statePredLocal,
  statePredNational,
  stateName,
  raceType,
  year,
  nCounties,
  nPolls,
}: RaceBlendControlsProps) {
  // Default weights vary based on poll availability (spec requirement)
  const defaultWeights: SectionWeights = hasPolls
    ? { model_prior: 60, state_polls: 30, national_polls: 10 }
    : { model_prior: 80, state_polls: 15, national_polls: 5 };

  const [blend, setBlend] = useState<BlendResult>({
    prediction: initialPrediction,
    pred_std: initialPredStd,
    pred_lo90: initialLo90,
    pred_hi90: initialHi90,
  });
  const [isLoading, setIsLoading] = useState(false);

  // Keep the previous good values so we can fall back on API error
  const prevBlend = useRef<BlendResult>(blend);
  // Debounce timer ref
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync initial props into state when they change (e.g. Next.js soft navigation)
  useEffect(() => {
    const next: BlendResult = {
      prediction: initialPrediction,
      pred_std: initialPredStd,
      pred_lo90: initialLo90,
      pred_hi90: initialHi90,
    };
    setBlend(next);
    prevBlend.current = next;
  }, [initialPrediction, initialPredStd, initialLo90, initialHi90]);

  const handleWeightsChange = useCallback(
    (weights: SectionWeights) => {
      // Cancel any in-flight debounce
      if (debounceTimer.current !== null) {
        clearTimeout(debounceTimer.current);
      }

      debounceTimer.current = setTimeout(async () => {
        setIsLoading(true);
        try {
          const res = await fetch(
            `${apiBase}/api/v1/forecast/race/${slug}/blend`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(weights),
            },
          );

          if (!res.ok) {
            // Retain previous values on non-2xx response
            return;
          }

          const data: BlendResult = await res.json();
          prevBlend.current = data;
          setBlend(data);
        } catch {
          // Network error — retain previous values silently
          setBlend(prevBlend.current);
        } finally {
          setIsLoading(false);
        }
      }, DEBOUNCE_MS);
    },
    [slug, apiBase],
  );

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimer.current !== null) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  const predStd = blend.pred_std ?? FALLBACK_STD;

  return (
    <>
      {/* Hero — large margin, rating badge, CI bounds.
          Fades slightly while a recalculation is in-flight. */}
      <div
        style={{ transition: "opacity 150ms ease", opacity: isLoading ? 0.5 : 1 }}
        aria-busy={isLoading}
      >
        <RaceHero
          stateName={stateName}
          raceType={raceType}
          year={year}
          prediction={blend.prediction}
          nCounties={nCounties}
          lo90={blend.pred_lo90}
          hi90={blend.pred_hi90}
          nPolls={nPolls}
        />
      </div>

      {/* Outcome distribution dotplot */}
      {blend.prediction !== null && (
        <section className="mb-10">
          <h2
            className="font-serif text-xl mb-4"
            style={{ fontFamily: "var(--font-serif)" }}
          >
            Outcome Distribution
          </h2>
          <p className="text-sm mb-3" style={{ color: "var(--color-text-muted)" }}>
            Each dot represents one possible scenario. The distribution is derived
            from the model&apos;s prediction and estimated uncertainty
            (±{(predStd * 100).toFixed(0)}pp std).
          </p>
          <div
            style={{ transition: "opacity 150ms ease", opacity: isLoading ? 0.5 : 1 }}
            aria-busy={isLoading}
          >
            <QuantileDotplot
              predDemShare={blend.prediction}
              predStd={predStd}
              nDots={100}
              width={480}
              height={160}
            />
          </div>
        </section>
      )}

      {/* Forecast blend controls */}
      <section className="mb-10">
        <h2
          className="font-serif text-xl mb-4"
          style={{ fontFamily: "var(--font-serif)" }}
        >
          Forecast Blend
        </h2>
        {hasPolls ? (
          <>
            <p className="text-sm mb-3" style={{ color: "var(--color-text-muted)" }}>
              Adjust how the forecast weights the structural model prior against
              available polling data.
              {statePredLocal !== null && statePredLocal !== undefined && (
                <>
                  {" "}State-level model prior:{" "}
                  <span className="font-mono">
                    {(statePredLocal * 100).toFixed(1)}% D
                  </span>
                  .
                </>
              )}
              {statePredNational !== null && statePredNational !== undefined && (
                <>
                  {" "}National-adjusted:{" "}
                  <span className="font-mono">
                    {(statePredNational * 100).toFixed(1)}% D
                  </span>
                  .
                </>
              )}
            </p>
            <SectionWeightSliders
              initial={defaultWeights}
              onChange={handleWeightsChange}
            />
          </>
        ) : (
          <p
            className="text-sm rounded-md px-4 py-3"
            style={{
              color: "var(--color-text-muted)",
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
            }}
          >
            Blend controls will appear here once polling data is available for this race.
          </p>
        )}
      </section>
    </>
  );
}
