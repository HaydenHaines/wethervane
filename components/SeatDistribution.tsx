import type { SeatDistribution as Distribution } from "../lib/site-types";
import { COLORS, percent } from "../lib/presentation";

export default function SeatDistribution({
  party,
  distribution,
  observed,
  baseline,
}: {
  party: "D" | "R";
  distribution: Distribution;
  observed: number;
  baseline?: Distribution;
}) {
  const support = [...distribution.pmf, ...(baseline?.pmf || [])];
  const occupied = [distribution.pmf, baseline?.pmf || []].flatMap((pmf) =>
    pmf.flatMap((p, seat) => (p > 0 ? [seat] : [])),
  );
  const lower = Math.max(0, Math.min(observed, ...occupied) - 1),
    upper = Math.min(100, Math.max(observed, ...occupied) + 1);
  const width = 420,
    height = 178,
    left = 34,
    right = 12,
    top = 30,
    bottom = 31;
  const count = upper - lower + 1,
    band = (width - left - right) / count;
  const max = Math.max(...support, 0.01),
    plotHeight = height - top - bottom;
  const x = (seat: number) => left + (seat - lower + 0.5) * band,
    y = (p: number) => height - bottom - (p / max) * plotHeight;
  const seats = Array.from({ length: count }, (_, i) => i + lower);
  const tickEvery = count > 26 ? 5 : count > 14 ? 2 : 1;
  const reference = baseline
    ? seats.map((seat) => `${x(seat)},${y(baseline.pmf[seat] || 0)}`).join(" ")
    : null;
  const name = party === "D" ? "Democratic" : "Republican";
  return (
    <figure className="seat-distribution">
      <figcaption>{name} caucus seats</figcaption>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${name} caucus seat probability distribution. Observed ${observed} seats. The selected model assigned ${percent(distribution.pmf[observed] || 0)} probability to that exact total. Shown range ${lower} to ${upper} seats.`}
      >
        <title>
          {name} caucus: predicted seat totals and the observed result
        </title>
        <line
          x1={left}
          x2={width - right}
          y1={height - bottom}
          y2={height - bottom}
          stroke="var(--line)"
        />
        <text
          x={left - 5}
          y={top + 4}
          textAnchor="end"
          className="distribution-axis"
        >
          {percent(max)}
        </text>
        <text
          x={left - 5}
          y={height - bottom + 3}
          textAnchor="end"
          className="distribution-axis"
        >
          0%
        </text>
        {seats.map((seat) => (
          <g key={seat}>
            <rect
              x={x(seat) - band * 0.4}
              y={y(distribution.pmf[seat] || 0)}
              width={Math.max(1, band * 0.8)}
              height={((distribution.pmf[seat] || 0) / max) * plotHeight}
              fill={COLORS[party]}
              opacity=".8"
            >
              <title>
                {seat} seats: {percent(distribution.pmf[seat] || 0)}
              </title>
            </rect>
            {(seat % tickEvery === 0 || seat === lower || seat === upper) && (
              <text
                x={x(seat)}
                y={height - bottom + 16}
                textAnchor="middle"
                className="distribution-axis"
              >
                {seat}
              </text>
            )}
          </g>
        ))}
        {reference && (
          <polyline
            points={reference}
            fill="none"
            stroke="var(--ink)"
            strokeWidth="1.5"
            strokeDasharray="4 3"
          />
        )}
        <line
          x1={x(observed)}
          x2={x(observed)}
          y1={top - 8}
          y2={height - bottom + 2}
          stroke="var(--ink)"
          strokeWidth="2"
        />
        <text
          x={Math.max(left + 34, Math.min(width - right - 34, x(observed)))}
          y={14}
          textAnchor="middle"
          className="distribution-observed"
        >
          Observed {observed}
        </text>
      </svg>
      <div className="distribution-legend">
        <span>
          <i style={{ background: COLORS[party] }} />
          Selected stack
        </span>
        <span>
          <i className="observed-line" />
          Observed result
        </span>
        {baseline && (
          <span>
            <i className="default-line" />
            Default stack
          </span>
        )}
      </div>
    </figure>
  );
}
