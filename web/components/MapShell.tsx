/**
 * Re-export shim — the canonical MapShell now lives at
 * `@/components/map/MapShell`.
 *
 * This file is kept so existing imports in `app/(map)/layout.tsx` continue
 * to resolve without changes. Prefer importing from the canonical path in
 * new code.
 */

export { default } from "@/components/map/MapShell";
export type { SuperTypeInfo, TractContext } from "@/components/map/MapShell";

// Legacy named exports retained for any consumers that used them directly.
export { PALETTE, getColorForSuperType } from "@/components/map/map-palette";
