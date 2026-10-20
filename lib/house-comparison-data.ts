import data from "../data/house-comparisons.json";
import { parseHouseComparisonManifest } from "./house-comparison";

/** Exact reviewed comparison groups, independent of result publication/readiness. */
export const HOUSE_COMPARISONS = parseHouseComparisonManifest(data);
