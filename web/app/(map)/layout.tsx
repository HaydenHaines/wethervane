"use client";

import { MapProvider } from "@/components/MapContext";
import { TabBar } from "@/components/TabBar";
import dynamic from "next/dynamic";

const MapShell = dynamic(() => import("@/components/MapShell"), { ssr: false });

export default function MapLayout({ children }: { children: React.ReactNode }) {
  return (
    <MapProvider>
      <div className="app-shell" style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
      }}>
        <div className="map-pane" style={{ flex: 1, position: "relative", minWidth: 0 }}>
          <MapShell />
        </div>
        <div className="panel-pane" style={{
          width: "var(--color-panel-width)",
          display: "flex",
          flexDirection: "column",
          borderLeft: "1px solid var(--color-border)",
          background: "var(--color-surface)",
          overflow: "hidden",
        }}>
          <TabBar />
          <div className="panel-scroll" style={{ flex: 1, overflow: "auto" }}>
            {children}
          </div>
        </div>
      </div>
    </MapProvider>
  );
}
