import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Page Not Found — WetherVane",
  description: "The page you requested could not be found.",
};

const NAV_LINKS = [
  { label: "Home", href: "/", description: "National stained-glass map" },
  { label: "Forecast", href: "/forecast", description: "2026 race predictions" },
  { label: "Types", href: "/types", description: "Electoral community types" },
  { label: "Methodology", href: "/methodology", description: "How the model works" },
];

export default function NotFound() {
  return (
    <main
      style={{
        maxWidth: 600,
        margin: "0 auto",
        padding: "80px 24px 100px",
        textAlign: "center",
      }}
    >
      <p
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: 96,
          fontWeight: 700,
          lineHeight: 1,
          margin: "0 0 16px",
          color: "var(--color-text-muted)",
          opacity: 0.25,
        }}
      >
        404
      </p>

      <h1
        style={{
          fontFamily: "var(--font-serif)",
          fontSize: 32,
          margin: "0 0 12px",
          lineHeight: 1.2,
        }}
      >
        Page Not Found
      </h1>

      <p
        style={{
          fontSize: 16,
          color: "var(--color-text-muted)",
          lineHeight: 1.6,
          margin: "0 0 48px",
        }}
      >
        The page you requested doesn&apos;t exist or may have moved.
        <br />
        Try one of the sections below.
      </p>

      <nav aria-label="Site sections">
        <ul
          style={{
            listStyle: "none",
            margin: 0,
            padding: 0,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 12,
            textAlign: "left",
          }}
        >
          {NAV_LINKS.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                style={{
                  display: "block",
                  padding: "14px 16px",
                  borderRadius: 6,
                  border: "1px solid var(--color-border)",
                  background: "var(--color-surface)",
                  textDecoration: "none",
                  transition: "border-color 0.15s",
                }}
              >
                <span
                  style={{
                    display: "block",
                    fontWeight: 600,
                    fontSize: 14,
                    color: "var(--color-dem)",
                    marginBottom: 4,
                  }}
                >
                  {link.label}
                </span>
                <span
                  style={{
                    display: "block",
                    fontSize: 12,
                    color: "var(--color-text-muted)",
                  }}
                >
                  {link.description}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </main>
  );
}
