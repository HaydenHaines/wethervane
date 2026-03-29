/**
 * Full-screen map page — /explore/map
 *
 * Renders a full-viewport interactive map with:
 *  - Stained glass super-type coloring (default)
 *  - Forecast partisan-lean choropleth (via overlay toggle)
 *  - State click → fly-to + tract detail
 *  - Floating legend, zoom controls, overlay toggle
 *
 * deck.gl is loaded client-side only (no SSR) via next/dynamic.
 * MapProvider is instantiated here since this page is outside the
 * (map) route group that normally provides it.
 */

import type { Metadata } from "next";
import { MapPageClient } from "./MapPageClient";

export const metadata: Metadata = {
  title: "Electoral Map | WetherVane",
  description:
    "Interactive full-screen map of US electoral communities. Explore tract-level super-types or view partisan forecast by state.",
  openGraph: {
    title: "Electoral Map | WetherVane",
    description:
      "Interactive full-screen map of US electoral communities and 2026 forecast.",
    type: "website",
    siteName: "WetherVane",
  },
  twitter: {
    card: "summary_large_image",
    title: "Electoral Map | WetherVane",
    description: "Interactive full-screen electoral community map.",
  },
};

export default function MapPage() {
  return <MapPageClient />;
}
