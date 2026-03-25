import { MetadataRoute } from "next";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const base = "https://wethervane.hhaines.duckdns.org";

  // Static pages
  const staticPages = ["", "/forecast", "/about", "/compare", "/explore"].map(
    (path) => ({
      url: `${base}${path}`,
      lastModified: new Date(),
      changeFrequency: "weekly" as const,
      priority: path === "" ? 1.0 : 0.8,
    })
  );

  // Fetch all county FIPS from API
  const apiBase =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";
  try {
    const res = await fetch(`${apiBase}/api/v1/counties`);
    const counties = await res.json();

    const countyPages = counties.map((c: { county_fips: string }) => ({
      url: `${base}/county/${c.county_fips}`,
      lastModified: new Date(),
      changeFrequency: "monthly" as const,
      priority: 0.6,
    }));

    return [...staticPages, ...countyPages];
  } catch {
    // If API is unavailable at build time, return static pages only
    return staticPages;
  }
}
