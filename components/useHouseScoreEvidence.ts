"use client";

import { useEffect, useState } from "react";
import type { HouseComparisonResult } from "../lib/house-comparison";
import { HOUSE_SCORE_EVIDENCE } from "../lib/house-score-evidence-data";
import { verifyHouseScoreEvidence } from "../lib/house-score-evidence";

type Evidence = Awaited<ReturnType<typeof verifyHouseScoreEvidence>>;
type State = Evidence | { status: "loading" };

export default function useHouseScoreEvidence(publication?: HouseComparisonResult): State | null {
  const [loaded, setLoaded] = useState<{ publication: HouseComparisonResult; state: Evidence } | null>(null);
  useEffect(() => {
    if (!publication) return;
    let active = true;
    if (!HOUSE_SCORE_EVIDENCE) {
      setLoaded({ publication, state: { status: "unavailable", reason: "The reviewed vote-share tests are unavailable." } });
      return;
    }
    verifyHouseScoreEvidence(HOUSE_SCORE_EVIDENCE, publication)
      .then(state => { if (active) setLoaded({ publication, state }); })
      .catch(() => {
        if (active) setLoaded({ publication, state: { status: "unavailable", reason: "The vote-share tests could not be matched to this exact saved result." } });
      });
    return () => { active = false; };
  }, [publication]);
  // Bind to the complete object, so changed content cannot retain prior evidence
  // merely by claiming the same file fingerprint.
  return !publication ? null : loaded?.publication === publication ? loaded.state : { status: "loading" };
}
