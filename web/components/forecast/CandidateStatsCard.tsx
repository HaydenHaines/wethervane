"use client";

/**
 * CandidateStatsCard — campaign finance and legislative stats for a candidate profile.
 *
 * Renders two optional sections:
 *   1. Campaign Finance (SDR, FER, burn rate) — shown only when has_fec_record is true
 *   2. Legislative Profile (DW-NOMINATE ideology, LES effectiveness, congresses served)
 *      — shown only when has_legislative_record is true
 *
 * Both sections are absent when the candidate has no congressional/FEC record
 * (e.g., pure gubernatorial candidates).
 *
 * Design: Dusty Ink palette, terse labels, shadcn-style containers consistent
 * with CandidateBadges.tsx and the candidate profile page Section component.
 */

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { CampaignStats, LegislativeStats } from "@/lib/api";

// ── Shared style tokens ───────────────────────────────────────────────────────

/** Muted label style: uppercase, tracking-wide, tiny */
const LABEL_STYLE: React.CSSProperties = {
  fontSize: "0.65rem",
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "var(--color-text-muted)",
  marginBottom: "2px",
};

/** Card section header — matches the Section component on the profile page */
const SECTION_TITLE_STYLE: React.CSSProperties = {
  fontSize: "0.68rem",
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: "var(--color-text-muted)",
  marginBottom: "12px",
};

// ── StatBar ───────────────────────────────────────────────────────────────────

interface StatBarProps {
  /** Label shown above the bar */
  label: string;
  /** Tooltip text explaining what the stat means */
  tooltipText: string;
  /** Value to fill the bar (0–1 fraction of maxValue) */
  value: number;
  /** Maximum value on the scale (default: 1.0) */
  maxValue?: number;
  /** Display string shown next to the bar */
  displayValue: string;
  /** Accent color for the fill */
  fillColor?: string;
  /** Optional: percentile text shown in muted style after the value */
  percentileLabel?: string;
}

function StatBar({
  label,
  tooltipText,
  value,
  maxValue = 1.0,
  displayValue,
  fillColor = "var(--color-dem)",
  percentileLabel,
}: StatBarProps) {
  const pct = Math.min(100, Math.max(0, (value / maxValue) * 100));

  return (
    <div style={{ marginBottom: "10px" }}>
      {/* Label row */}
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "baseline",
                  marginBottom: "3px",
                  cursor: "help",
                }}
              >
                <span style={LABEL_STYLE}>{label}</span>
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: "var(--color-text)",
                  }}
                >
                  {displayValue}
                  {percentileLabel && (
                    <span
                      style={{
                        fontSize: "0.65rem",
                        fontWeight: 400,
                        color: "var(--color-text-muted)",
                        marginLeft: "4px",
                      }}
                    >
                      {percentileLabel}
                    </span>
                  )}
                </span>
              </div>
            }
          />
          <TooltipContent side="top">
            <span style={{ fontSize: "0.75rem" }}>{tooltipText}</span>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {/* Bar track */}
      <div
        style={{
          height: "5px",
          borderRadius: "3px",
          background: "var(--color-border)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: fillColor,
            borderRadius: "3px",
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

// ── IdeologySlider ────────────────────────────────────────────────────────────

interface IdeologySliderProps {
  /** DW-NOMINATE dim1 value in [-1, +1] */
  nominate: number;
  /** Party of the candidate — used for marker color */
  party: string;
}

function IdeologySlider({ nominate, party }: IdeologySliderProps) {
  // Map [-1, +1] to [0%, 100%] left position
  const pct = ((nominate + 1) / 2) * 100;
  const markerColor = party === "R" ? "var(--color-rep)" : "var(--color-dem)";

  return (
    <div style={{ marginBottom: "10px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "3px",
        }}
      >
        <span style={LABEL_STYLE}>Ideology (DW-NOMINATE)</span>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger
              render={
                <span
                  style={{
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    color: markerColor,
                    cursor: "help",
                  }}
                >
                  {nominate >= 0 ? "+" : ""}
                  {nominate.toFixed(3)}
                </span>
              }
            />
            <TooltipContent side="top">
              <span style={{ fontSize: "0.75rem" }}>
                DW-NOMINATE first dimension: −1 = most liberal, +1 = most conservative.
                Career average across all recorded Congresses.
              </span>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>

      {/* Track with gradient */}
      <div
        style={{
          position: "relative",
          height: "6px",
          borderRadius: "3px",
          background:
            "linear-gradient(to right, var(--color-dem), rgba(148,163,184,0.3), var(--color-rep))",
        }}
      >
        {/* Candidate marker */}
        <div
          style={{
            position: "absolute",
            left: `${pct}%`,
            top: "50%",
            transform: "translate(-50%, -50%)",
            width: "10px",
            height: "10px",
            borderRadius: "50%",
            background: markerColor,
            border: "2px solid var(--color-surface)",
            boxShadow: "0 0 0 1px rgba(0,0,0,0.15)",
          }}
        />
      </div>

      {/* Axis labels */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "3px",
        }}
      >
        <span
          style={{
            fontSize: "0.6rem",
            color: "var(--color-dem)",
            fontWeight: 500,
          }}
        >
          ← Liberal
        </span>
        <span
          style={{
            fontSize: "0.6rem",
            color: "var(--color-rep)",
            fontWeight: 500,
          }}
        >
          Conservative →
        </span>
      </div>
    </div>
  );
}

// ── CampaignFinanceSection ────────────────────────────────────────────────────

interface CampaignFinanceSectionProps {
  stats: CampaignStats;
  /** All candidates' stats for the same party — used to compute percentiles */
  partyStats?: CampaignStats[];
}

/** Compute the percentile of `value` within `values` (0–100). */
function computePercentile(value: number, values: number[]): number {
  const valid = values.filter((v) => v !== null && !isNaN(v));
  if (valid.length === 0) return 50;
  const below = valid.filter((v) => v < value).length;
  return Math.round((below / valid.length) * 100);
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function CampaignFinanceSection({
  stats,
  partyStats = [],
}: CampaignFinanceSectionProps) {
  // Percentiles within party peers (omit null values)
  const partySDRs = partyStats
    .filter((s) => s.has_fec_record && s.sdr !== null)
    .map((s) => s.sdr as number);
  const partyFERs = partyStats
    .filter((s) => s.has_fec_record && s.fer !== null)
    .map((s) => s.fer as number);

  const sdrPctLabel =
    partySDRs.length > 1 && stats.sdr !== null
      ? `${computePercentile(stats.sdr, partySDRs)}th %ile (party)`
      : undefined;
  const ferPctLabel =
    partyFERs.length > 1 && stats.fer !== null
      ? `${computePercentile(stats.fer, partyFERs)}th %ile (party)`
      : undefined;

  return (
    <div>
      {stats.cycle && (
        <p
          style={{
            fontSize: "0.7rem",
            color: "var(--color-text-muted)",
            marginBottom: "10px",
          }}
        >
          Most recent cycle: {stats.cycle}
        </p>
      )}

      {stats.sdr !== null && (
        <StatBar
          label="Small-Dollar Ratio"
          tooltipText="% of individual donations under $200 — higher means more grassroots fundraising"
          value={stats.sdr}
          maxValue={1.0}
          displayValue={formatPct(stats.sdr)}
          fillColor="var(--color-dem)"
          percentileLabel={sdrPctLabel}
        />
      )}

      {stats.fer !== null && (
        <StatBar
          label="Fundraising Efficiency"
          tooltipText="Receipts ÷ disbursements — above 1.0 means the campaign raised more than it spent"
          value={Math.min(stats.fer, 1.5)}
          maxValue={1.5}
          displayValue={stats.fer.toFixed(3)}
          fillColor="rgba(20, 130, 65, 0.75)"
          percentileLabel={ferPctLabel}
        />
      )}

      {stats.burn_rate !== null && (
        <div style={{ marginTop: "8px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <span style={{ ...LABEL_STYLE, cursor: "help" }}>
                      Burn Rate (spending/raised)
                    </span>
                  }
                />
                <TooltipContent side="top">
                  <span style={{ fontSize: "0.75rem" }}>
                    Disbursements ÷ receipts. Below 1.0 means the campaign spent less than it raised
                    (cash accumulating). Above 1.0 means the campaign ran a spending deficit.
                  </span>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <span
              style={{
                fontSize: "0.82rem",
                fontWeight: 700,
                color:
                  stats.burn_rate > 1.05
                    ? "rgb(200, 60, 40)"
                    : stats.burn_rate < 0.85
                    ? "rgb(20, 130, 65)"
                    : "var(--color-text)",
              }}
            >
              {stats.burn_rate.toFixed(3)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── LegislativeSection ────────────────────────────────────────────────────────

interface LegislativeSectionProps {
  stats: LegislativeStats;
  party: string;
}

function LegislativeSection({ stats, party }: LegislativeSectionProps) {
  return (
    <div>
      {/* Ideology slider */}
      {stats.nominate_dim1 !== null && (
        <IdeologySlider nominate={stats.nominate_dim1} party={party} />
      )}

      {/* LES bar — scale 0–5+ */}
      {stats.les_score !== null && (
        <StatBar
          label="Legislative Effectiveness"
          tooltipText="Volden–Wiseman LES: measures bill sponsorship, committee advancement, and floor passage. Median ≈ 1.0 for serving members."
          value={stats.les_score}
          maxValue={5.0}
          displayValue={stats.les_score.toFixed(2)}
          fillColor="rgba(161, 98, 7, 0.75)"
          percentileLabel="Median = 1.0"
        />
      )}

      {/* Congresses served */}
      {stats.congresses_served !== null && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: "6px",
          }}
        >
          <span style={LABEL_STYLE}>Congresses Served</span>
          <span style={{ fontSize: "0.82rem", fontWeight: 700, color: "var(--color-text)" }}>
            {stats.congresses_served}
            <span
              style={{
                fontSize: "0.65rem",
                fontWeight: 400,
                color: "var(--color-text-muted)",
                marginLeft: "4px",
              }}
            >
              ({Math.round(stats.congresses_served * 2)} years)
            </span>
          </span>
        </div>
      )}
    </div>
  );
}

// ── CandidateStatsCard (public export) ────────────────────────────────────────

interface CandidateStatsCardProps {
  campaignStats: CampaignStats | null | undefined;
  legislativeStats: LegislativeStats | null | undefined;
  /** Candidate's party, used to color the ideology slider marker */
  party: string;
}

/**
 * Renders campaign finance and legislative profile cards for a candidate.
 *
 * Each section is independently gated on its `has_*_record` flag so governors
 * without FEC data or legislative records don't see empty cards.
 *
 * Returns null if both sections have nothing to show.
 */
export function CandidateStatsCard({
  campaignStats,
  legislativeStats,
  party,
}: CandidateStatsCardProps) {
  const showFinance = campaignStats?.has_fec_record === true;
  const showLegislative = legislativeStats?.has_legislative_record === true;

  if (!showFinance && !showLegislative) return null;

  const cardStyle: React.CSSProperties = {
    background: "var(--color-surface)",
    border: "1px solid var(--color-border)",
    borderRadius: "6px",
    padding: "16px",
    marginBottom: "12px",
  };

  return (
    <>
      {/* Campaign Finance section */}
      {showFinance && campaignStats && (
        <section style={cardStyle}>
          <h2 style={SECTION_TITLE_STYLE}>Campaign Finance</h2>
          <CampaignFinanceSection stats={campaignStats} />
        </section>
      )}

      {/* Legislative Profile section */}
      {showLegislative && legislativeStats && (
        <section style={cardStyle}>
          <h2 style={SECTION_TITLE_STYLE}>Legislative Profile</h2>
          <LegislativeSection stats={legislativeStats} party={party} />
        </section>
      )}
    </>
  );
}
