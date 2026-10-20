import data from "../data/house-score-evidence.json";
import { parseHouseScoreEvidenceManifest } from "./house-score-evidence";

export const HOUSE_SCORE_EVIDENCE = parseHouseScoreEvidenceManifest(data);
