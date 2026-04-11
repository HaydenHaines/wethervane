"use client";

import { useFundamentals } from "@/lib/hooks/use-fundamentals";

// ── Helpers ───────────────────────────────────────────────────────────────

/**
 * Format a signed percentage-point shift as "D+X.X" or "R+X.X".
 *
 * A positive shift means Democrats gain relative to the national baseline;
 * negative means Republicans gain.  The in_party determines the direction
 * of each contributor, but the combined_shift_pp is already signed correctly
 * (positive = Dem-favored environment).
 */
function formatShiftPp(pp: number): { label: string; color: string } {
  const abs = Math.abs(pp);
  if (abs < 0.05) {
    return { label: "EVEN", color: "var(--forecast-tossup)" };
  }
  const fmt = abs.toFixed(1);
  if (pp > 0) {
    return { label: `D+${fmt}`, color: "var(--forecast-safe-d)" };
  }
  return { label: `R+${fmt}`, color: "var(--forecast-safe-r)" };
}

/** Format a signed contribution value as "+X.Xpp" or "-X.Xpp" with a color. */
function formatContribution(pp: number): { text: string; color: string } {
  const abs = Math.abs(pp);
  if (abs < 0.005) {
    return { text: "0.0pp", color: "var(--color-text-muted)" };
  }
  const fmt = abs.toFixed(1) + "pp";
  if (pp > 0) {
    return { text: `+${fmt}`, color: "var(--forecast-lean-d)" };
  }
  return { text: `\u2212${fmt}`, color: "var(--forecast-lean-r)" };
}

// ── Loading skeleton ──────────────────────────────────────────────────────

function FundamentalsCardSkeleton() {
  return (
    <section
      className="mb-8 rounded-md p-4 text-sm animate-pulse"
      aria-label="National Environment loading"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      <div
        className="h-5 w-40 rounded mb-3"
        style={{ background: "var(--color-border)" }}
      />
      <div
        className="h-4 w-64 rounded mb-4"
        style={{ background: "var(--color-border-subtle)" }}
      />
      <div className="space-y-2">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex justify-between">
            <div
              className="h-3 w-32 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
            <div
              className="h-3 w-16 rounded"
              style={{ background: "var(--color-border-subtle)" }}
            />
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Main component ────────────────────────────────────────────────────────

/**
 * FundamentalsCard — National Environment section.
 *
 * Displays the output of the fundamentals model: a structural forecast
 * based on presidential approval and economic indicators.  This is a
 * "use client" component that fetches live via SWR.
 *
 * Design follows the HistoricalContextCard pattern: plain section with
 * surface background and Dusty Ink CSS variables — no shadcn Card wrapper.
 */
export function FundamentalsCard() {
  const { data, error, isLoading } = useFundamentals();

  // While loading, show a skeleton sized to match the card
  if (isLoading) {
    return <FundamentalsCardSkeleton />;
  }

  // On error (network failure, API down), silently hide the card rather than
  // showing a broken state next to real forecast content.
  if (error || !data) {
    return null;
  }

  const { label: combinedLabel, color: combinedColor } = formatShiftPp(
    data.combined_shift_pp,
  );
  const { label: fundamentalsLabel, color: fundamentalsColor } = formatShiftPp(
    data.shift_pp,
  );

  const weightPct = Math.round(data.weight * 100);

  // Build the indicator rows from the snapshot + contributions
  const indicators: Array<{
    label: string;
    value: string;
    contribution: ReturnType<typeof formatContribution>;
  }> = [
    {
      label: "Presidential Approval",
      value:
        data.snapshot.approval_net_oct !== null
          ? `${data.snapshot.approval_net_oct > 0 ? "+" : ""}${data.snapshot.approval_net_oct.toFixed(0)} net`
          : "N/A",
      contribution: formatContribution(data.approval_contribution_pp),
    },
    {
      label: "GDP Growth (Q2)",
      value:
        data.snapshot.gdp_q2_growth_pct !== null
          ? `${data.snapshot.gdp_q2_growth_pct.toFixed(1)}%`
          : "N/A",
      contribution: formatContribution(data.gdp_contribution_pp),
    },
    {
      label: "Unemployment (Oct)",
      value:
        data.snapshot.unemployment_oct !== null
          ? `${data.snapshot.unemployment_oct.toFixed(1)}%`
          : "N/A",
      contribution: formatContribution(data.unemployment_contribution_pp),
    },
    {
      label: "CPI Inflation (YoY)",
      value:
        data.snapshot.cpi_yoy_oct !== null
          ? `${data.snapshot.cpi_yoy_oct.toFixed(1)}%`
          : "N/A",
      contribution: formatContribution(data.cpi_contribution_pp),
    },
  ];

  return (
    <section
      className="mb-8 rounded-md p-4 text-sm"
      aria-label="National Environment"
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      {/* Header row: title + combined shift */}
      <div className="flex flex-wrap items-baseline justify-between gap-3 mb-1">
        <h2
          className="font-serif text-lg"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          National Environment
        </h2>
        <span
          className="font-mono font-bold text-xl"
          style={{ color: combinedColor }}
          aria-label={`Combined forecast shift: ${combinedLabel}`}
        >
          {combinedLabel}
        </span>
      </div>

      {/* Subheader */}
      <p className="mb-4 text-xs" style={{ color: "var(--color-text-muted)" }}>
        Structural forecast from economic indicators and presidential approval
        (blended with generic ballot polls)
      </p>

      {/* Indicator breakdown table */}
      <dl className="space-y-2 mb-4">
        {indicators.map(({ label, value, contribution }) => (
          <div key={label} className="flex items-center justify-between gap-4">
            <dt style={{ color: "var(--color-text-muted)" }}>{label}</dt>
            <dd className="flex items-center gap-3 text-right">
              <span style={{ color: "var(--color-text)" }}>{value}</span>
              <span
                className="font-mono font-semibold w-14 text-right"
                style={{ color: contribution.color }}
                aria-label={`${label} contribution: ${contribution.text}`}
              >
                {contribution.text}
              </span>
            </dd>
          </div>
        ))}
      </dl>

      {/* Divider + blend note + uncertainty */}
      <div
        className="pt-3 space-y-1 border-t"
        style={{ borderColor: "var(--color-border)" }}
      >
        {/* Fundamentals-only shift vs combined */}
        <div className="flex items-center justify-between gap-4">
          <span style={{ color: "var(--color-text-muted)" }}>
            Fundamentals-only signal
          </span>
          <span
            className="font-mono font-semibold"
            style={{ color: fundamentalsColor }}
          >
            {fundamentalsLabel}
          </span>
        </div>

        {/* Blend weight note */}
        <p style={{ color: "var(--color-text-muted)" }}>
          Fundamentals weight:{" "}
          <span style={{ color: "var(--color-text)" }}>{weightPct}%</span>
          {" "}&mdash; blended with generic ballot polls
        </p>

        {/* Model uncertainty */}
        <p style={{ color: "var(--color-text-muted)" }}>
          Model uncertainty:{" "}
          <span style={{ color: "var(--color-text)" }}>
            &plusmn;{data.loo_rmse_pp.toFixed(1)}pp
          </span>
          {" "}(LOO RMSE from {data.n_training} midterm cycles)
        </p>
      </div>

      {/* State-level economic adjustment section */}
      {data.state_econ_enabled && data.state_econ.length > 0 && (
        <StateEconSection entries={data.state_econ} />
      )}
    </section>
  );
}


// ── State Economics sub-section ──────────────────────────────────────────

/**
 * Collapsible section showing per-state economic adjustments.
 *
 * Displays the top 5 and bottom 5 states by shift adjustment, derived
 * from BLS QCEW county employment data.  This gives users insight into
 * how regional economic conditions are modulating the national forecast.
 */
function StateEconSection({
  entries,
}: {
  entries: Array<{
    state_abbr: string | null;
    emp_growth_rel_pp: number;
    shift_adjustment_pp: number;
  }>;
}) {
  // Sort by shift adjustment (most positive = D-favorable economy)
  const sorted = [...entries]
    .filter((e) => e.state_abbr)
    .sort((a, b) => b.shift_adjustment_pp - a.shift_adjustment_pp);

  // Show top 3 and bottom 3 to keep the card compact
  const top = sorted.slice(0, 3);
  const bottom = sorted.slice(-3).reverse();

  return (
    <div
      className="pt-3 mt-3 border-t"
      style={{ borderColor: "var(--color-border)" }}
    >
      <p
        className="font-semibold text-xs mb-2"
        style={{ color: "var(--color-text-muted)" }}
      >
        State Economic Conditions (QCEW)
      </p>
      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
        <div>
          <p className="mb-1" style={{ color: "var(--color-text-muted)" }}>
            Fastest job growth
          </p>
          {top.map((e) => (
            <StateEconRow key={e.state_abbr} entry={e} />
          ))}
        </div>
        <div>
          <p className="mb-1" style={{ color: "var(--color-text-muted)" }}>
            Slowest job growth
          </p>
          {bottom.map((e) => (
            <StateEconRow key={e.state_abbr} entry={e} />
          ))}
        </div>
      </div>
      <p
        className="mt-2 text-xs"
        style={{ color: "var(--color-text-muted)" }}
      >
        Source: BLS QCEW 2021-2023 employment growth vs national average
      </p>
    </div>
  );
}


/** Single row in the state economics section. */
function StateEconRow({
  entry,
}: {
  entry: {
    state_abbr: string | null;
    emp_growth_rel_pp: number;
    shift_adjustment_pp: number;
  };
}) {
  const { text: adjText, color: adjColor } = formatContribution(
    entry.shift_adjustment_pp,
  );

  return (
    <div className="flex items-center justify-between gap-2">
      <span style={{ color: "var(--color-text)" }}>
        {entry.state_abbr ?? "??"}
      </span>
      <span className="font-mono" style={{ color: adjColor }}>
        {adjText}
      </span>
    </div>
  );
}
