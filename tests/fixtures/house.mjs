/** Synthetic, test-only data. These probabilities and rectangles are not forecasts or legal maps. */
import { createHash } from "node:crypto";

const STATE_SEATS_2022 = {
  AL: 7,
  AK: 1,
  AZ: 9,
  AR: 4,
  CA: 52,
  CO: 8,
  CT: 5,
  DE: 1,
  FL: 28,
  GA: 14,
  HI: 2,
  ID: 2,
  IL: 17,
  IN: 9,
  IA: 4,
  KS: 4,
  KY: 6,
  LA: 6,
  ME: 2,
  MD: 8,
  MA: 9,
  MI: 13,
  MN: 8,
  MS: 4,
  MO: 8,
  MT: 2,
  NE: 3,
  NV: 4,
  NH: 2,
  NJ: 12,
  NM: 3,
  NY: 26,
  NC: 14,
  ND: 1,
  OH: 15,
  OK: 5,
  OR: 6,
  PA: 17,
  RI: 2,
  SC: 7,
  SD: 1,
  TN: 9,
  TX: 38,
  UT: 4,
  VT: 1,
  VA: 11,
  WA: 10,
  WV: 2,
  WI: 8,
  WY: 1,
};
const STATE_SEATS_2018 = {
  ...STATE_SEATS_2022,
  CA: 53,
  CO: 7,
  FL: 27,
  IL: 18,
  MI: 14,
  MT: 1,
  NY: 27,
  NC: 13,
  OH: 16,
  OR: 5,
  PA: 18,
  TX: 36,
  WV: 3,
};
const CENTERS = {
  AL: [-86.7, 32.8],
  AK: [-152, 64],
  AZ: [-111.5, 34.2],
  AR: [-92.4, 34.8],
  CA: [-119.5, 36.5],
  CO: [-105.5, 39],
  CT: [-72.7, 41.6],
  DE: [-75.5, 39],
  FL: [-82, 28],
  GA: [-83.4, 32.7],
  HI: [-157.5, 20.8],
  ID: [-114.5, 44.2],
  IL: [-89.2, 40],
  IN: [-86.2, 40],
  IA: [-93.5, 42],
  KS: [-98, 38.5],
  KY: [-85, 37.5],
  LA: [-91.9, 31],
  ME: [-69, 45],
  MD: [-76.8, 39],
  MA: [-71.8, 42.3],
  MI: [-85, 44],
  MN: [-94.5, 46],
  MS: [-89.7, 32.7],
  MO: [-92.4, 38.5],
  MT: [-110.7, 47],
  NE: [-99.7, 41.5],
  NV: [-116.7, 39],
  NH: [-71.7, 43.8],
  NJ: [-74.5, 40],
  NM: [-106, 34.5],
  NY: [-75.5, 43],
  NC: [-79.5, 35.5],
  ND: [-100.5, 47.5],
  OH: [-82.8, 40.3],
  OK: [-97.5, 35.5],
  OR: [-120.5, 44],
  PA: [-77.7, 41],
  RI: [-71.5, 41.7],
  SC: [-80.9, 33.8],
  SD: [-100, 44.5],
  TN: [-86, 35.8],
  TX: [-99, 31],
  UT: [-111.5, 39.5],
  VT: [-72.7, 44],
  VA: [-78.7, 37.5],
  WA: [-120.7, 47.3],
  WV: [-80.7, 38.7],
  WI: [-89.7, 44.5],
  WY: [-107.5, 43],
};

export const fixtureSha256 = (value) =>
  createHash("sha256")
    .update(typeof value === "string" ? value : JSON.stringify(value))
    .digest("hex");
export function districtKeysForYear(year = 2022) {
  const counts = year < 2022 ? STATE_SEATS_2018 : STATE_SEATS_2022;
  return Object.entries(counts).flatMap(([state, count]) =>
    Array.from(
      { length: count },
      (_, index) =>
        `${state}-${count === 1 ? "00" : String(index + 1).padStart(2, "0")}`,
    ),
  );
}
export const HOUSE_DISTRICT_KEYS = districtKeysForYear();
const PRIORITY = [
  "CA-01",
  "CA-02",
  "TX-01",
  "TX-02",
  "NY-01",
  "NY-02",
  "FL-01",
  "FL-02",
  "OH-01",
  "CO-01",
  "NC-01",
  "MI-01",
  "AZ-01",
  "GA-01",
  "AK-00",
  "HI-01",
];

export function makeHouseResult({
  eligibleCount = 16,
  year = 2022,
  fullChamber = false,
} = {}) {
  if (
    !Number.isInteger(eligibleCount) ||
    eligibleCount < 1 ||
    eligibleCount > 435
  )
    throw Error("Fixture eligibleCount must be between1 and435.");
  const keys = districtKeysForYear(year);
  const ranked = [
    ...PRIORITY.filter((key) => keys.includes(key)),
    ...keys.filter((key) => !PRIORITY.includes(key)),
  ];
  const eligible = new Set(ranked.slice(0, eligibleCount));
  const mapVintage = `house_election_map_${year}`;
  const configurationId = `synthetic-house-configuration-${year}`;
  const coverage = keys.map((districtKey) => ({
    raceId: `${year}:${districtKey}:house:regular`,
    districtKey,
    eligible: eligible.has(districtKey),
    exclusionReason: eligible.has(districtKey)
      ? null
      : "Excluded by synthetic test fixture",
  }));
  const races = coverage
    .filter((row) => row.eligible)
    .map((row, index) => {
      const p = [0.97, 0.8, 0.58, 0.42, 0.2, 0.03][index % 6];
      const outcomes = [
        { party: "D", p },
        { party: "R", p: 1 - p },
      ].map(({ party, p }) => {
        const name = `Synthetic ${party === "D" ? "Democrat" : "Republican"} ${row.districtKey}`;
        return {
          id: party,
          name,
          party,
          kind: "candidate",
          members: [
            {
              id: `fixture-${year}-${row.districtKey}-${party}`,
              name,
              party,
              sourceUrls: [
                `https://example.invalid/test-only/${year}/${row.districtKey}/${party}`,
              ],
            },
          ],
          winProbability: p,
        };
      });
      const observed = outcomes[index % 2];
      return {
        id: row.raceId,
        state: row.districtKey.slice(0, 2),
        seatId: `US-HOUSE-${row.districtKey}`,
        electionLabel: "Regular election",
        district: { cycle: year, mapVintage, districtKey: row.districtKey },
        outcomes,
        notes: [
          "Synthetic test-only categorical probabilities; not an electoral forecast.",
        ],
        observed: {
          outcomeId: observed.id,
          winnerName: observed.name,
          winnerParty: observed.party,
          sourceUrls: [
            `https://example.invalid/test-only/${year}/${row.districtKey}/observed`,
          ],
        },
      };
    });
  return {
    schemaVersion: 1,
    reviewStatus: "reviewed",
    run: {
      id: `synthetic-house-${year}-${eligibleCount}`,
      configurationId,
      configurationSha256: fixtureSha256(configurationId),
      label: `Synthetic House test fixture (${eligibleCount} races)`,
      year,
      historical: true,
      experimental: true,
      informationMode: "retrospective_proxy_strictly_earlier_fitted_labels",
      evidenceLabel:
        "Synthetic fixture for software tests only; no scientific qualification.",
    },
    metadata: {
      chamber: "house",
      chamber_size: 435,
      majority_threshold: 218,
      scope: fullChamber ? "full_chamber" : "eligible_race_subset",
      full_chamber: fullChamber,
      cycle: year,
      mapVintage,
      configurationId,
    },
    races,
    coverage,
    limitations: [
      "Synthetic fixture: values and observed outcomes are generated solely for automated tests.",
      "No joint chamber or tipping-point artifact is supplied.",
    ],
  };
}

export function makeHouseGeography(
  result,
  {
    allDistricts = true,
    admission = "verified",
    mapVintage = result.metadata.mapVintage,
    cycle = result.metadata.cycle,
  } = {},
) {
  const keys = allDistricts
    ? result.coverage.map((row) => row.districtKey)
    : result.races.map((race) => race.district.districtKey);
  const counts = result.coverage.reduce((sum, row) => {
    const state = row.districtKey.slice(0, 2);
    sum[state] = (sum[state] || 0) + 1;
    return sum;
  }, {});
  const geojson = {
    type: "FeatureCollection",
    features: keys.map((districtKey) => {
      const state = districtKey.slice(0, 2),
        number = Number(districtKey.slice(3)),
        index = number === 0 ? 0 : number - 1;
      const [centerLon, centerLat] = CENTERS[state],
        columns = Math.ceil(Math.sqrt(counts[state]));
      const lon = centerLon + ((index % columns) - (columns - 1) / 2) * 0.13,
        lat =
          centerLat + (Math.floor(index / columns) - (columns - 1) / 2) * 0.13;
      return {
        type: "Feature",
        properties: { districtKey },
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [lon, lat],
              [lon, lat + 0.1],
              [lon + 0.1, lat + 0.1],
              [lon + 0.1, lat],
              [lon, lat],
            ],
          ],
        },
      };
    }),
  };
  return {
    metadata: {
      cycle,
      mapVintage,
      admission,
      sourceSha256: fixtureSha256(geojson),
    },
    geojson,
  };
}

export function makeHousePublication({
  eligibleCount = 16,
  year = 2022,
  fullChamber = false,
  withGeography = true,
} = {}) {
  const result = makeHouseResult({ eligibleCount, year, fullChamber });
  return {
    schemaVersion: 1,
    status: "available",
    result,
    resultSha256: fixtureSha256(result),
    geography: withGeography ? makeHouseGeography(result) : null,
  };
}
