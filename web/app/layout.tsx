import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WetherVane — 2026 Electoral Forecast",
  description: "Community-based electoral forecasting for the 2026 midterms",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
