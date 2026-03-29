import type { Metadata } from "next";
import { redirect } from "next/navigation";

export const metadata: Metadata = {
  title: "Forecast — WetherVane",
  description: "2026 Senate and Governor race forecasts",
};

export default function ForecastPage() {
  redirect("/forecast/senate");
}
