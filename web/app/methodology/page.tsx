import type { Metadata } from "next";
import { MethodologyContent } from "@/components/MethodologyContent";

// ── Metadata ──────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: "Methodology | WetherVane",
  description:
    "How WetherVane discovers electoral communities from shift patterns, estimates type covariance, and propagates polling signals across geography to produce county-level 2026 forecasts.",
  openGraph: {
    title: "Methodology | WetherVane",
    description:
      "How WetherVane discovers electoral communities from shift patterns, estimates type covariance, and propagates polling signals across geography to produce county-level 2026 forecasts.",
    type: "article",
    siteName: "WetherVane",
    images: [
      {
        url: "/methodology/opengraph-image",
        width: 1200,
        height: 630,
        alt: "WetherVane Methodology — How we discover electoral communities",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Methodology | WetherVane",
    description:
      "How WetherVane's type-primary electoral model works: KMeans discovery, soft membership, covariance estimation, and poll propagation.",
  },
};

// ── Page Component ────────────────────────────────────────────────────────

export default function MethodologyPage() {
  return (
    <main id="main-content">
      <MethodologyContent />
    </main>
  );
}
