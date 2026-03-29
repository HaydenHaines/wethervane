"use client";

/**
 * MapControls — zoom +/- buttons, reset view, and back-to-national button.
 *
 * Rendered as absolutely-positioned overlays on the map.
 * The parent MapShell passes callbacks for each action.
 */

interface ButtonProps {
  onClick: () => void;
  title: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}

function ControlButton({ onClick, title, children, style }: ButtonProps) {
  return (
    <button
      onClick={onClick}
      title={title}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 32,
        height: 32,
        borderRadius: 4,
        border: "1px solid var(--color-border, #e0ddd8)",
        background: "var(--color-surface, #fafaf8)",
        color: "var(--color-text, #3a3632)",
        fontSize: 16,
        cursor: "pointer",
        fontFamily: "var(--font-sans)",
        lineHeight: 1,
        ...style,
      }}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------

interface MapControlsProps {
  /** Called when user clicks zoom in (+). */
  onZoomIn: () => void;
  /** Called when user clicks zoom out (-). */
  onZoomOut: () => void;
  /** Called when user clicks reset view. */
  onResetView: () => void;
  /** When non-null, show "Back to national" button with this state abbreviation. */
  zoomedState: string | null;
  /** Called when user clicks "Back to national". */
  onBackToNational: () => void;
  /** Whether tracts are currently loading (shows a loading indicator). */
  loadingTracts?: boolean;
}

export function MapControls({
  onZoomIn,
  onZoomOut,
  onResetView,
  zoomedState,
  onBackToNational,
  loadingTracts,
}: MapControlsProps) {
  return (
    <>
      {/* Back to national — top left, only when zoomed */}
      {zoomedState && (
        <button
          onClick={onBackToNational}
          style={{
            position: "absolute",
            top: 12,
            left: 12,
            zIndex: 10,
            padding: "6px 12px",
            borderRadius: 4,
            border: "1px solid var(--color-border, #e0ddd8)",
            background: "var(--color-surface, #fafaf8)",
            color: "var(--color-text, #3a3632)",
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "var(--font-sans)",
          }}
        >
          &larr; Back to national
        </button>
      )}

      {/* Loading indicator */}
      {loadingTracts && (
        <div
          style={{
            position: "absolute",
            top: 12,
            left: zoomedState ? 160 : 12,
            zIndex: 10,
            padding: "4px 10px",
            borderRadius: 4,
            background: "rgba(20, 24, 32, 0.75)",
            color: "#f0f4f8",
            fontSize: 11,
            fontFamily: "var(--font-sans)",
          }}
        >
          Loading tracts...
        </div>
      )}

      {/* Zoom controls — bottom right */}
      <div
        style={{
          position: "absolute",
          bottom: 24,
          right: 16,
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          gap: 4,
        }}
      >
        <ControlButton onClick={onZoomIn} title="Zoom in">
          +
        </ControlButton>
        <ControlButton onClick={onZoomOut} title="Zoom out">
          −
        </ControlButton>
        <ControlButton onClick={onResetView} title="Reset view" style={{ fontSize: 11 }}>
          ↺
        </ControlButton>
      </div>
    </>
  );
}
