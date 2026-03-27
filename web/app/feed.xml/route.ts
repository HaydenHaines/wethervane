// RSS 2.0 feed — regenerated hourly via revalidate.
// Server-side fetch uses the direct API URL (bypasses the /api/* client rewrite).

export const revalidate = 3600;

const SITE_URL = "https://wethervane.hhaines.duckdns.org";
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

// Human-readable race label from a URL slug.
// "2026-fl-governor" → "2026 FL Governor"
function slugToTitle(slug: string): string {
  const parts = slug.split("-");
  if (parts.length < 3) return slug;
  const [year, state, ...rest] = parts;
  return `${year} ${state.toUpperCase()} ${rest.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(" ")}`;
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function buildFeedXml(slugs: string[], pubDate: string): string {
  const items = slugs
    .map((slug) => {
      const title = slugToTitle(slug);
      const link = `${SITE_URL}/forecast/${slug}`;
      const description = escapeXml(
        `County-level electoral forecast for ${title}. View WetherVane's type-based model prediction.`,
      );
      return [
        "    <item>",
        `      <title>${escapeXml(title)}</title>`,
        `      <link>${link}</link>`,
        `      <guid isPermaLink="true">${link}</guid>`,
        `      <description>${description}</description>`,
        `      <pubDate>${pubDate}</pubDate>`,
        "    </item>",
      ].join("\n");
    })
    .join("\n");

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
    "  <channel>",
    "    <title>WetherVane — Electoral Forecast Updates</title>",
    `    <link>${SITE_URL}</link>`,
    "    <description>County-level electoral forecasts for 2026 midterms. Updated weekly with new polling data.</description>",
    "    <language>en-us</language>",
    `    <lastBuildDate>${pubDate}</lastBuildDate>`,
    `    <atom:link href="${SITE_URL}/feed.xml" rel="self" type="application/rss+xml" />`,
    items,
    "  </channel>",
    "</rss>",
  ].join("\n");
}

export async function GET(): Promise<Response> {
  const pubDate = new Date().toUTCString();

  let slugs: string[] = [];
  try {
    const res = await fetch(`${API_BASE}/api/v1/forecast/race-slugs`, {
      next: { revalidate },
    });
    if (res.ok) {
      slugs = await res.json();
    }
  } catch {
    // API unavailable — return an empty but valid feed rather than 500.
  }

  const xml = buildFeedXml(slugs, pubDate);

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8",
      "Cache-Control": "public, max-age=3600, stale-while-revalidate=86400",
    },
  });
}
