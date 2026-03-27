import Link from "next/link";

export default function AboutPage() {
  return (
    <div style={{ padding: "20px 16px" }}>
      {/* Title */}
      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: "22px",
          fontWeight: "700",
          margin: "0 0 8px",
        }}
      >
        About WetherVane
      </h1>
      <p
        style={{
          fontSize: "14px",
          color: "var(--color-text-muted)",
          margin: "0 0 20px",
          lineHeight: "1.6",
        }}
      >
        A structural model of US electoral behavior.
      </p>

      {/* Summary */}
      <p
        style={{
          fontSize: "14px",
          lineHeight: "1.7",
          color: "var(--color-text)",
          margin: "0 0 12px",
        }}
      >
        WetherVane discovers communities of voters who move together
        politically, estimates how those communities covary, and propagates
        polling signals across geography to produce county-level forecasts.
      </p>
      <p
        style={{
          fontSize: "14px",
          lineHeight: "1.7",
          color: "var(--color-text)",
          margin: "0 0 24px",
        }}
      >
        Unlike poll aggregators, WetherVane starts from the underlying
        structure — types discovered from 16 years of electoral shift patterns
        across 3,154 US counties. A poll in Florida updates predictions in
        Georgia, because both states contain counties that share type
        memberships.
      </p>

      {/* Full methodology link — prominent */}
      <Link
        href="/methodology"
        style={{
          display: "block",
          padding: "12px 16px",
          border: "1.5px solid var(--color-dem)",
          borderRadius: "6px",
          textDecoration: "none",
          marginBottom: "24px",
          background: "var(--color-surface)",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "15px",
            fontWeight: "700",
            color: "var(--color-dem)",
            marginBottom: "2px",
          }}
        >
          Read the full methodology →
        </div>
        <div
          style={{
            fontSize: "12px",
            color: "var(--color-text-muted)",
          }}
        >
          KMeans discovery, soft membership, covariance estimation, poll
          propagation, and model performance details
        </div>
      </Link>

      {/* Key metrics */}
      <div
        style={{
          borderTop: "1px solid var(--color-border)",
          paddingTop: "16px",
          marginBottom: "20px",
        }}
      >
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: "15px",
            fontWeight: "700",
            margin: "0 0 12px",
          }}
        >
          Model Metrics
        </h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "8px",
          }}
        >
          {[
            { label: "Counties", value: "3,154" },
            { label: "Fine types", value: "100" },
            { label: "Super-types", value: "5" },
            { label: "Ensemble LOO r", value: "0.711" },
            { label: "County RMSE", value: "7.3 pp" },
            { label: "Covariance val r", value: "0.915" },
          ].map((m) => (
            <div
              key={m.label}
              style={{
                padding: "8px 10px",
                border: "1px solid var(--color-border)",
                borderRadius: "4px",
                background: "var(--color-bg)",
              }}
            >
              <div
                style={{
                  fontSize: "11px",
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  color: "var(--color-text-muted)",
                  marginBottom: "2px",
                }}
              >
                {m.label}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-serif)",
                  fontSize: "16px",
                  fontWeight: "700",
                }}
              >
                {m.value}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Status + Credits */}
      <div
        style={{
          borderTop: "1px solid var(--color-border)",
          paddingTop: "16px",
        }}
      >
        <p style={{ fontSize: "13px", color: "var(--color-text-muted)", margin: "0 0 6px" }}>
          Targeting 2026 midterms. Currently tracking 18 races.
        </p>
        <p style={{ fontSize: "13px", color: "var(--color-text-muted)", margin: "0" }}>
          Built by Hayden Haines. Inspired by The Economist&apos;s 2020
          presidential model.
        </p>
      </div>
    </div>
  );
}
