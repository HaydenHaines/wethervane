import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ballotmancy — The races, in perspective",
  description:
    "Explore Senate and House races, compare forecast models, and see how they performed in past elections.",
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
