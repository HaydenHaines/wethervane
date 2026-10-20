/** Invented UI fixtures only. No values describe an actual electorate or forecast. */
const fingerprint = "a".repeat(64);
const none = { kind: "none" };
const denominator = (quantityId, label, interpretation) => ({
  kind: "quantity",
  quantityId,
  label,
  interpretation,
});
const predictive = (id, quantityType, unit, mean, denominatorValue = none) => ({
  id,
  label: `Illustrative ${id}`,
  quantityType,
  unit,
  denominator: denominatorValue,
  kind: "backend_weighted_predictive",
  mean,
  intervals: [
    {
      level: 0.8,
      lower: mean * 0.8,
      upper: Math.min(
        quantityType === "candidate_share" ? 1 : Infinity,
        mean * 1.2,
      ),
    },
  ],
  weightedSummaryId: "invented-weighted-summary",
});
export function partialTexasFixture() {
  const v = {
    kind: "count_view_preparation",
    provenance: {
      kind: "invented_fixture",
      label: "Invented partial Texas example — not an election forecast",
    },
    lifecycle: "ready",
    bindings: {
      requestSha256: fingerprint,
      configurationSha256: fingerprint,
      lawSha256: fingerprint,
      dataBundleSha256: fingerprint,
      admissionSha256: fingerprint,
      outputContractId: "invented-count-output",
      informationCutoff: "2026-10-20T00:00:00Z",
      targetId: "invented-count-target",
      componentVersions: [{ id: "invented-provider", version: "fixture-v1" }],
      jointArtifact: { id: "invented-joint-artifact", sha256: fingerprint },
      weightedSummary: {
        id: "invented-weighted-summary",
        sha256: fingerprint,
        jointArtifactSha256: fingerprint,
        weighting: "original_parent_mass_times_normalized_conditional_weight",
      },
      axes: {
        eventIds: ["TX-main"],
        contestIds: ["TX-senate"],
        candidateIds: ["named-D", "named-R"],
      },
    },
    audit: {
      status: "passed",
      sourceAdmission: "admitted",
      sameRowAccounting: "passed",
      qualification:
        "Invented audit declarations for UI tests; not verified scientific evidence.",
      parents: [
        {
          id: "invented-parent-1",
          originalMass: 0.7,
          representedMass: 0.7,
          status: "passed",
        },
        {
          id: "invented-parent-2",
          originalMass: 0.3,
          representedMass: 0.3,
          status: "passed",
        },
      ],
    },
    scope: {
      label: "Illustrative Texas pilot only",
      requiredEventIds: ["TX-main", "XX-main"],
      requiredContestIds: ["TX-senate", "XX-senate"],
      populationCells: {
        count: 24,
        label: "24 joint demographic cells",
        dimensions: [
          "Age:18–34/35–64/65+",
          "Hispanic/non-Hispanic White/non-Hispanic Black/other non-Hispanic",
          "Below BA/BA or higher",
        ],
        limitations:
          "No sex or gender dimension; citizen exposure is not exact legal eligibility.",
      },
      communityProfiles: {
        count: 12,
        label: "12 religion × attendance profiles",
        dimensions: ["Four religion groups", "Three attendance groups"],
        limitations:
          "Measured proxy, not a comprehensive definition of community.",
      },
    },
    quantities: [
      predictive("adult-residents", "adult_residents", "people", 130),
      predictive("citizen-exposure", "citizen_adult_exposure", "people", 90),
      predictive("participants", "event_participants", "people", 100),
      predictive(
        "valid-candidate-votes",
        "countable_candidate_total",
        "votes",
        95,
      ),
      predictive("D-count", "candidate_count", "votes", 60),
      predictive("R-count", "candidate_count", "votes", 35),
      predictive(
        "D-share",
        "candidate_share",
        "ratio",
        0.62,
        denominator(
          "valid-candidate-votes",
          "All countable candidate votes",
          "Named-candidate votes only; blank and invalid ballots excluded.",
        ),
      ),
      predictive(
        "R-share",
        "candidate_share",
        "ratio",
        0.38,
        denominator(
          "valid-candidate-votes",
          "All countable candidate votes",
          "Named-candidate votes only; blank and invalid ballots excluded.",
        ),
      ),
      predictive("blank-count", "noncandidate_count", "ballots", 3),
      predictive("invalid-count", "noncandidate_count", "ballots", 2),
      predictive(
        "turnout",
        "turnout_ratio",
        "ratio",
        1.15,
        denominator(
          "citizen-exposure",
          "Citizen-adult exposure",
          "Realized participants divided by exposure, not a verified eligibility fraction.",
        ),
      ),
      {
        id: "F1a-report",
        label: "Illustrative administrative participation report",
        quantityType: "administrative_participation",
        unit: "people",
        denominator: none,
        kind: "observed_administrative",
        value: 102,
        sourceId: "invented-administrative-source",
        sourceAdmission: "admitted",
        sourceQualification:
          "Invented administrative-source review declaration for UI tests, not a verified real report.",
        sourceSha256: fingerprint,
        referenceAt: "2026-11-03",
        publicationAt: "2026-12-01",
      },
    ],
    events: [
      {
        id: "TX-main",
        state: "TX",
        date: "2026-11-03",
        participantQuantityId: "participants",
        citizenExposureQuantityId: "citizen-exposure",
        turnoutQuantityId: "turnout",
        administrativeReportQuantityId: "F1a-report",
        contestIds: ["TX-senate"],
      },
    ],
    contests: [
      {
        id: "TX-senate",
        eventId: "TX-main",
        applicability: "present",
        rule: {
          id: "invented-plurality-rule",
          family: "single_mark_plurality",
          admission: "admitted",
          qualification: "Invented rule admission only.",
        },
        rosterComplete: true,
        candidates: [
          {
            id: "named-D",
            name: "Invented Alice Example",
            party: "D",
            identitySourceSha256: fingerprint,
            countQuantityId: "D-count",
            shareQuantityId: "D-share",
          },
          {
            id: "named-R",
            name: "Invented Robin Example",
            party: "R",
            identitySourceSha256: fingerprint,
            countQuantityId: "R-count",
            shareQuantityId: "R-share",
          },
        ],
        noncandidateCategories: [
          {
            id: "blank",
            label: "Blank/undervote",
            kind: "undervote",
            admitted: true,
            countQuantityId: "blank-count",
          },
          {
            id: "invalid",
            label: "Invalid ballot mark",
            kind: "invalid",
            admitted: true,
            countQuantityId: "invalid-count",
          },
        ],
        outcomes: {
          kind: "backend_weighted_predictive",
          weightedSummaryId: "invented-weighted-summary",
          candidateProbabilities: { "named-D": 0.65, "named-R": 0.3 },
          tiedMaximumProbability: 0.03,
          noValidCandidateVotesProbability: 0.02,
          derivation: "same_joint_integer_count_rows",
        },
      },
    ],
  };
  return v;
}
export const makeCountViewFixture = partialTexasFixture;
export function missingNationalCoverageFixture() {
  const v = partialTexasFixture();
  v.scope.label = "Invented missing national coverage";
  return v;
}
export function failedParentFixture() {
  const v = partialTexasFixture();
  v.audit.parents[1].status = "failed";
  v.audit.parents[1].representedMass = 0;
  return v;
}
export function unsupportedRuleFixture() {
  const v = partialTexasFixture();
  v.contests[0].rule.family = "unsupported";
  v.contests[0].rule.admission = "held";
  v.contests[0].outcomes = {
    kind: "unavailable",
    reason: "Ranked-choice tabulation is not admitted in this fixture.",
  };
  return v;
}
export function unresolvedRaceTieFixture() {
  const v = partialTexasFixture();
  v.contests[0].outcomes.candidateProbabilities = {
    "named-D": 0,
    "named-R": 0,
  };
  v.contests[0].outcomes.tiedMaximumProbability = 1;
  v.contests[0].outcomes.noValidCandidateVotesProbability = 0;
  return v;
}
export function absentContestFixture() {
  const v = partialTexasFixture();
  v.contests[0].applicability = "absent";
  v.contests[0].candidates = [];
  v.contests[0].noncandidateCategories = [];
  v.contests[0].outcomes = {
    kind: "unavailable",
    reason: "No contest applies to this event.",
  };
  v.bindings.axes.candidateIds = [];
  return v;
}
export function chamberTieScenarioFixture() {
  const v = partialTexasFixture();
  v.provenance.label = "Invented 50–50 chamber/VP scenario — not a forecast";
  v.scope.requiredEventIds = ["TX-main"];
  v.scope.requiredContestIds = ["TX-senate"];
  const fixed = [
    ...Array.from({ length: 50 }, (_, i) => `D-fixed-${i}`),
    ...Array.from({ length: 49 }, (_, i) => `R-fixed-${i}`),
  ];
  v.chamber = {
    chamberSize: 100,
    fixedSeatIds: fixed,
    requiredFixedSeatIds: [...fixed],
    contestIds: ["TX-senate"],
    scenario: {
      id: "invented-50-50-R-VP",
      label: "Illustrative 50–50 organizing scenario",
      organizingDescription:
        "In this invented scenario the parties have 50 seats each; the Republican vice president breaks a chamber tie. This is distinct from tied candidate vote totals.",
      vicePresidentialTieBreakParty: "R",
      sourceSha256: fingerprint,
    },
    weightedSummaryId: "invented-weighted-summary",
    derivation: "same_joint_winner_rows_and_fixed_ledger",
    control: {
      democratic: 0,
      republican: 1,
      unresolvedRace: 0,
      unresolvedOrganizing: 0,
    },
  };
  v.contests[0].outcomes.candidateProbabilities = {
    "named-D": 0,
    "named-R": 1,
  };
  v.contests[0].outcomes.tiedMaximumProbability = 0;
  v.contests[0].outcomes.noValidCandidateVotesProbability = 0;
  for (const [id, mean] of [
    ["D-count", 0],
    ["R-count", 100],
    ["valid-candidate-votes", 100],
    ["blank-count", 0],
    ["invalid-count", 0],
    ["D-share", 0],
    ["R-share", 1],
  ]) {
    const q = v.quantities.find((q) => q.id === id);
    q.mean = mean;
    q.intervals = [{ level: 0.8, lower: mean, upper: mean }];
  }
  return v;
}
