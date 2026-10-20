import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ballotmancy — The races, in perspective",
  description:
    "Explore Senate races, model comparisons, and historical performance. A district-by-district House perspective is in preparation.",
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
