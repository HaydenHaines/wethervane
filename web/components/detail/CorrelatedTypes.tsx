/**
 * CorrelatedTypes — cards for the N most electorally similar types.
 *
 * When `correlatedTypes` is provided (from the Ledoit-Wolf covariance matrix),
 * it renders those with a correlation coefficient badge. Falls back to the
 * super-type sibling approach when covariance data is unavailable.
 *
 * All data is passed as props from the SSR page — no client-side SWR needed
 * for the initial render. The `useCorrelatedTypes` hook exists for client
 * components that want live updates.
 */

"use client";

import Link from "next/link";
import { MarginDisplay } from "@/components/shared/MarginDisplay";
import { getSuperTypeColor, rgbToHex } from "@/lib/config/palette";
import type { TypeSummary, SuperTypeSummary, CorrelatedTypeData } from "@/lib/types";

/** Maximum number of sibling types to show when falling back to super-type logic. */
const MAX_SIBLING_FALLBACK = 4;

// ── Shared card ──────────────────────────────────────────────────────────────

interface CardShellProps {
  typeId: number;
  superTypeId: number;
  displayName: string;
  nCounties: number;
  meanPredDemShare: number | null;
  superTypeName: string;
  /** Optional correlation label, e.g. "r = 0.72". Only shown when using covariance data. */
  correlationLabel?: string;
}

function TypeCard({
  typeId,
  superTypeId,
  displayName,
  nCounties,
  meanPredDemShare,
  superTypeName,
  correlationLabel,
}: CardShellProps) {
  const accentHex = rgbToHex(getSuperTypeColor(superTypeId));

  return (
    <Link
      href={`/type/${typeId}`}
      style={{ textDecoration: "none", display: "block" }}
    >
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 6,
          padding: "12px 14px",
          background: "var(--color-surface, var(--color-bg))",
          transition: "border-color 0.15s",
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLDivElement).style.borderColor = accentHex;
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLDivElement).style.borderColor =
            "var(--color-border)";
        }}
      >
        {/* Super-type badge */}
        <span
          style={{
            display: "inline-block",
            padding: "2px 8px",
            borderRadius: 3,
            fontSize: 11,
            fontWeight: 600,
            background: accentHex + "22",
            border: `1px solid ${accentHex}`,
            color: accentHex,
            marginBottom: 6,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            maxWidth: "100%",
          }}
        >
          {superTypeName}
        </span>

        {/* Type name + lean */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 8,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div
              style={{
                fontSize: 11,
                color: "var(--color-text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                marginBottom: 2,
              }}
            >
              Type {typeId}
            </div>
            <div
              style={{
                fontSize: 13,
                fontWeight: 700,
                fontFamily: "var(--font-serif)",
                color: "var(--color-text)",
                lineHeight: 1.3,
                wordBreak: "break-word",
              }}
            >
              {displayName}
            </div>
          </div>
          <div style={{ flexShrink: 0, paddingTop: 16 }}>
            <MarginDisplay demShare={meanPredDemShare} size="sm" />
          </div>
        </div>

        {/* County count + correlation */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginTop: 6,
          }}
        >
          <span style={{ fontSize: 12, color: "var(--color-text-muted)" }}>
            {nCounties} {nCounties === 1 ? "county" : "counties"}
          </span>
          {correlationLabel && (
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                color: "var(--color-text-muted)",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {correlationLabel}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

// ── Component ────────────────────────────────────────────────────────────────

interface CorrelatedTypesProps {
  /** All electoral type summaries — used for the super-type fallback. */
  allTypes: TypeSummary[];
  /** All super-type summaries — used for display names. */
  superTypes: SuperTypeSummary[];
  currentTypeId: number;
  superTypeId: number;
  /**
   * Real covariance-based correlated types from the API.
   * When present these take precedence over the super-type fallback.
   */
  correlatedTypes?: CorrelatedTypeData[];
}

export function CorrelatedTypes({
  allTypes,
  superTypes,
  currentTypeId,
  superTypeId,
  correlatedTypes,
}: CorrelatedTypesProps) {
  const superTypeMap = new Map(superTypes.map((st) => [st.super_type_id, st]));

  // ── Covariance-based path ─────────────────────────────────────────────────
  if (correlatedTypes && correlatedTypes.length > 0) {
    return (
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 12,
        }}
      >
        {correlatedTypes.map((ct) => {
          const stName =
            superTypeMap.get(ct.super_type_id)?.display_name ??
            `Super-Type ${ct.super_type_id}`;
          return (
            <TypeCard
              key={ct.type_id}
              typeId={ct.type_id}
              superTypeId={ct.super_type_id}
              displayName={ct.display_name}
              nCounties={ct.n_counties}
              meanPredDemShare={ct.mean_pred_dem_share}
              superTypeName={stName}
              correlationLabel={`r\u202f=\u202f${ct.correlation.toFixed(2)}`}
            />
          );
        })}
      </div>
    );
  }

  // ── Super-type fallback ───────────────────────────────────────────────────
  const siblings = allTypes
    .filter(
      (t) => t.super_type_id === superTypeId && t.type_id !== currentTypeId,
    )
    .slice(0, MAX_SIBLING_FALLBACK);

  if (siblings.length === 0) {
    return (
      <p style={{ fontSize: 14, color: "var(--color-text-muted)" }}>
        No similar types found.
      </p>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
        gap: 12,
      }}
    >
      {siblings.map((t) => {
        const stName =
          superTypeMap.get(t.super_type_id)?.display_name ??
          `Super-Type ${t.super_type_id}`;
        return (
          <TypeCard
            key={t.type_id}
            typeId={t.type_id}
            superTypeId={t.super_type_id}
            displayName={t.display_name}
            nCounties={t.n_counties}
            meanPredDemShare={t.mean_pred_dem_share}
            superTypeName={stName}
          />
        );
      })}
    </div>
  );
}
