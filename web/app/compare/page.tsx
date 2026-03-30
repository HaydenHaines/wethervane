import type { Metadata } from "next";
import Link from "next/link";
import { ComparisonTable } from "./ComparisonTable";

// ── Metadata ───────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "Forecaster Comparison | WetherVane",
  description:
    "See how WetherVane's 2026 election predictions compare to Cook Political Report, Sabato's Crystal Ball, and Inside Elections across every tracked Senate and Governor race.",
  openGraph: {
    title: "Forecaster Comparison | WetherVane",
    description:
      "How WetherVane's structural model stacks up against major election forecasters for the 2026 midterms.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary",
    title: "Forecaster Comparison | WetherVane",
    description:
      "WetherVane vs. Cook, Sabato, Inside Elections — side-by-side ratings for every 2026 Senate and Governor race.",
  },
};

const JSON_LD = {
  "@context": "https://schema.org",
  "@type": "WebPage",
  name: "Forecaster Comparison — WetherVane vs. Cook, Sabato, Inside Elections",
  description:
    "Side-by-side comparison of WetherVane election forecasts against Cook Political Report, Sabato's Crystal Ball, and Inside Elections for all 2026 Senate and Governor races.",
  url: "https://wethervane.hhaines.duckdns.org/compare",
  isPartOf: {
    "@type": "WebSite",
    name: "WetherVane",
    url: "https://wethervane.hhaines.duckdns.org",
  },
};

// ── Data fetch ─────────────────────────────────────────────────────────────

const API_BASE = process.env.API_URL || "http://localhost:8002";

async function fetchComparisons() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/comparisons`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

// ── Page ───────────────────────────────────────────────────────────────────

export default async function ComparePage() {
  const data = await fetchComparisons();

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(JSON_LD) }}
      />
      <div className="max-w-5xl mx-auto py-8 px-4 pb-20">
        {/* Breadcrumb */}
        <nav aria-label="breadcrumb" className="text-xs mb-6" style={{ color: "var(--color-text-muted)" }}>
          <ol className="flex flex-wrap items-center gap-x-1 list-none p-0 m-0">
            <li>
              <Link href="/" style={{ color: "var(--forecast-safe-d)", textDecoration: "none" }}>
                Home
              </Link>
            </li>
            <li aria-hidden="true">/</li>
            <li aria-current="page">Forecaster Comparison</li>
          </ol>
        </nav>

        {/* Header */}
        <header className="mb-8">
          <h1
            className="text-3xl font-bold mb-3"
            style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
          >
            Forecaster Comparison
          </h1>
          <p className="text-sm leading-relaxed max-w-2xl" style={{ color: "var(--color-text-muted)" }}>
            How does WetherVane stack up against the major election forecasters?
            Below is a side-by-side view of ratings from{" "}
            <a
              href="https://www.cookpolitical.com/"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "var(--forecast-safe-d)" }}
            >
              Cook Political Report
            </a>
            ,{" "}
            <a
              href="https://centerforpolitics.org/crystalball/"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "var(--forecast-safe-d)" }}
            >
              Sabato&apos;s Crystal Ball
            </a>
            , and{" "}
            <a
              href="https://insideelections.com/"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "var(--forecast-safe-d)" }}
            >
              Inside Elections
            </a>{" "}
            for all 2026 Senate and Governor races. WetherVane&apos;s predictions
            are derived from a structural model of electoral communities — not
            from polls or pundit consensus.
          </p>

          {/* Rating scale explainer */}
          <div
            className="mt-4 text-xs rounded-md px-4 py-3"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            <strong style={{ color: "var(--color-text)" }}>Rating scale: </strong>
            Safe/Solid D &rarr; Likely D &rarr; Lean D &rarr; Toss-up/Tilt &rarr; Lean R &rarr; Likely R &rarr; Safe/Solid R.
            WetherVane thresholds: Toss-up &plusmn;3pp, Lean &plusmn;3&ndash;8pp, Likely &plusmn;8&ndash;15pp, Safe 15pp+.
            External forecasters use their own judgment-based criteria.
            {data?.last_updated && (
              <span className="ml-2">
                External ratings as of <strong>{data.last_updated}</strong>.
              </span>
            )}
          </div>
        </header>

        {/* Table */}
        {data ? (
          <ComparisonTable races={data.races} />
        ) : (
          <div
            className="text-center py-16 rounded-md"
            style={{
              background: "var(--color-surface)",
              border: "1px solid var(--color-border)",
              color: "var(--color-text-muted)",
            }}
          >
            <p>Could not load comparison data. Please try again later.</p>
          </div>
        )}

        {/* Disclaimer */}
        <footer
          className="mt-8 text-xs"
          style={{ color: "var(--color-text-subtle)" }}
        >
          External ratings are manually curated and may not reflect the most
          current ratings from each forecaster. WetherVane is not affiliated
          with Cook Political Report, Sabato&apos;s Crystal Ball, or Inside Elections.
          Links open in a new tab.
        </footer>
      </div>
    </>
  );
}
