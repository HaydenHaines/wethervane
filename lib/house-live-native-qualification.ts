/** Native export conventions from export_live.py; byte authentication is separate. */
import { houseLivePin, type HouseLivePin } from "./house-live-result.ts";

export interface HouseLivePublicationQualification {
  schema: "HouseLivePublicationQualificationV1";
  status: "reviewed";
  numerical_review: HouseLivePin;
  seat_quantile_convention: "inverse_cdf" | "linear";
  share_quantile_convention: "linear";
  limitations: string[];
}

export function parseHouseLivePublicationQualification(value: unknown): HouseLivePublicationQualification | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const keys = ["schema", "status", "numerical_review", "seat_quantile_convention", "share_quantile_convention", "limitations"];
  if (Object.keys(raw).length !== keys.length || !keys.every(key => Object.hasOwn(raw, key)) ||
    raw.schema !== "HouseLivePublicationQualificationV1" || raw.status !== "reviewed" ||
    !houseLivePin(raw.numerical_review) || !["inverse_cdf", "linear"].includes(String(raw.seat_quantile_convention)) ||
    raw.share_quantile_convention !== "linear" || !Array.isArray(raw.limitations) || !raw.limitations.length ||
    !raw.limitations.every(note => typeof note === "string" && note.length > 0 && note.length <= 4000 &&
      note === note.trim() && !/[\u0000-\u001f\u007f]/.test(note))) return null;
  return { schema: "HouseLivePublicationQualificationV1", status: "reviewed",
    numerical_review: { path: raw.numerical_review.path, sha256: raw.numerical_review.sha256 },
    seat_quantile_convention: raw.seat_quantile_convention as "inverse_cdf" | "linear",
    share_quantile_convention: "linear", limitations: [...raw.limitations] };
}
