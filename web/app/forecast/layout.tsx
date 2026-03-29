import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Forecast — WetherVane",
  description: "2026 Senate and Governor race forecasts",
};

export default function ForecastLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col lg:flex-row h-[calc(100vh-3rem)]">
      {/* Map pane — placeholder; map refactor comes in Task 28 */}
      <div className="hidden lg:block lg:w-1/2 bg-[rgb(var(--color-bg))] border-r border-[rgb(var(--color-border))]">
        <div className="flex items-center justify-center h-full text-[rgb(var(--color-text-muted))]">
          Map loads here
        </div>
      </div>
      {/* Panel pane */}
      <div className="flex-1 overflow-y-auto p-4 lg:p-6">
        {children}
      </div>
    </div>
  );
}
