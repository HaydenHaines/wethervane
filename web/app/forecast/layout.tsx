"use client";

import dynamic from "next/dynamic";
import { MapProvider } from "@/components/MapContext";

// MapShell uses deck.gl which requires browser APIs — must be client-only.
const MapShell = dynamic(() => import("@/components/map/MapShell"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full text-[rgb(var(--color-text-muted))]">
      Loading map…
    </div>
  ),
});

export default function ForecastLayout({ children }: { children: React.ReactNode }) {
  return (
    <MapProvider>
      <div className="flex flex-col lg:flex-row h-[calc(100vh-3rem)]">
        {/* Map pane — desktop only; hidden on mobile */}
        <div className="hidden lg:block lg:w-1/2 bg-[rgb(var(--color-bg))] border-r border-[rgb(var(--color-border))]">
          <MapShell />
        </div>
        {/* Panel pane */}
        <div className="flex-1 overflow-y-auto p-4 lg:p-6">
          {children}
        </div>
      </div>
    </MapProvider>
  );
}
