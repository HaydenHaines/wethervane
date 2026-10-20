import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ballotmancy — The Senate, in perspective",
  description:
    "Explore a dated research preview of the 2026 Senate: state races, model comparisons, and chamber scenarios.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
