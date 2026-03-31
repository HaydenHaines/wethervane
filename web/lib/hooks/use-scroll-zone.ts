"use client";

import { useState, useEffect, RefObject } from "react";

/**
 * Tracks which narrative zone is currently most visible in the viewport.
 *
 * Each ref passed in should correspond to a <section data-zone="..."> element.
 * Returns the data-zone value of the section that most recently crossed the
 * threshold (0.3 visibility), or null if none has been seen yet.
 *
 * Implementation note: We use a single IntersectionObserver with threshold 0.3
 * and keep track of all currently-intersecting zones. The "active" zone is the
 * one with the highest intersectionRatio among those currently visible, breaking
 * ties by DOM order (first wins). This avoids the jitter of re-entrant observer
 * calls during fast scrolls.
 */
export function useScrollZone(
  refs: RefObject<HTMLElement | null>[],
): string | null {
  const [activeZone, setActiveZone] = useState<string | null>(null);

  useEffect(() => {
    if (refs.length === 0) return;

    // Map from element to its current intersection ratio
    const ratioMap = new Map<Element, number>();

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          ratioMap.set(entry.target, entry.intersectionRatio);
        }

        // Find the element with the highest ratio; among ties, pick the one
        // that appears earliest in `refs` (DOM order).
        let bestEl: Element | null = null;
        let bestRatio = 0;

        for (const ref of refs) {
          const el = ref.current;
          if (!el) continue;
          const ratio = ratioMap.get(el) ?? 0;
          if (ratio > bestRatio) {
            bestRatio = ratio;
            bestEl = el;
          }
        }

        if (bestEl && bestRatio > 0) {
          const zone = (bestEl as HTMLElement).dataset.zone ?? null;
          setActiveZone(zone);
        }
      },
      { threshold: [0, 0.1, 0.3, 0.5, 0.75, 1.0] },
    );

    for (const ref of refs) {
      if (ref.current) observer.observe(ref.current);
    }

    return () => observer.disconnect();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refs.length]);

  return activeZone;
}
