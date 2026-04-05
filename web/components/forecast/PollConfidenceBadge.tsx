"use client"

/**
 * PollConfidenceBadge — small badge indicating poll source diversity for a race.
 *
 * Renders a colored label ("High" / "Medium" / "Low") with a tooltip that
 * shows the breakdown: number of pollsters, methodologies, and total polls.
 *
 * This is a client component because it uses the @base-ui Tooltip, which
 * requires client-side interactivity.
 */

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

export interface PollConfidence {
  n_polls: number
  n_pollsters: number
  n_methodologies: number
  label: "High" | "Medium" | "Low"
  tooltip: string
}

interface PollConfidenceBadgeProps {
  confidence: PollConfidence
}

/** Map confidence label to a stable color pair (background / text). */
const LABEL_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  High: {
    bg: "rgba(34, 197, 94, 0.12)",
    text: "rgb(20, 120, 60)",
    border: "rgba(34, 197, 94, 0.35)",
  },
  Medium: {
    bg: "rgba(234, 179, 8, 0.12)",
    text: "rgb(130, 100, 0)",
    border: "rgba(234, 179, 8, 0.35)",
  },
  Low: {
    bg: "rgba(148, 163, 184, 0.12)",
    text: "rgb(90, 100, 115)",
    border: "rgba(148, 163, 184, 0.35)",
  },
}

export function PollConfidenceBadge({ confidence }: PollConfidenceBadgeProps) {
  const style = LABEL_STYLES[confidence.label] ?? LABEL_STYLES.Low

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          render={
            <span
              aria-label={`Poll confidence: ${confidence.label}. ${confidence.tooltip}`}
              role="status"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "1px 7px",
                borderRadius: "9999px",
                fontSize: "0.7rem",
                fontWeight: 600,
                letterSpacing: "0.03em",
                cursor: "default",
                userSelect: "none",
                background: style.bg,
                color: style.text,
                border: `1px solid ${style.border}`,
              }}
            >
              <span aria-hidden="true">●</span>
              {confidence.label}
            </span>
          }
        />
        <TooltipContent side="top">
          {confidence.tooltip}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
