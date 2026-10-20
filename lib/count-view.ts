/** Internal presentation preparation only. Not a backend API or an artifact parser.
 * Every accepted view remains an invented fixture, never a production forecast.
 * Audit declarations are required context, not proof of a joint law from summaries.
 */
export type CountQuantityType =
  | "adult_residents"
  | "citizen_adult_exposure"
  | "event_participants"
  | "candidate_count"
  | "countable_candidate_total"
  | "candidate_share"
  | "noncandidate_count"
  | "turnout_ratio"
  | "standardized_propensity"
  | "administrative_participation";
export interface CountInterval {
  level: number;
  lower: number;
  upper: number;
}
export interface CountQuantityBase {
  id: string;
  label: string;
  quantityType: CountQuantityType;
  unit: "people" | "votes" | "ballots" | "responses" | "ratio";
  denominator:
    | { kind: "none" }
    | {
        kind: "quantity";
        quantityId: string;
        label: string;
        interpretation: string;
      };
}
export type CountQuantity = CountQuantityBase &
  (
    | { kind: "unavailable"; reason: string }
    | {
        kind: "observed_administrative";
        value: number;
        sourceId: string;
        sourceAdmission: "admitted" | "held";
        sourceQualification: string;
        sourceSha256: string;
        referenceAt: string;
        publicationAt: string;
      }
    | {
        kind: "backend_weighted_predictive";
        mean: number;
        intervals: CountInterval[];
        weightedSummaryId: string;
      }
  );
export interface CountEventView {
  id: string;
  state: string;
  date: string;
  participantQuantityId: string;
  citizenExposureQuantityId: string;
  turnoutQuantityId?: string;
  administrativeReportQuantityId?: string;
  contestIds: string[];
}
export interface CountContestView {
  id: string;
  eventId: string;
  applicability: "present" | "absent";
  rule: {
    id: string;
    family: "single_mark_plurality" | "unsupported";
    admission: "admitted" | "held";
    qualification: string;
  };
  rosterComplete: boolean;
  candidates: {
    id: string;
    name: string;
    party: string | null;
    identitySourceSha256: string;
    countQuantityId: string;
    shareQuantityId: string;
  }[];
  noncandidateCategories: {
    id: string;
    label: string;
    kind:
      "blank" | "undervote" | "invalid" | "nonresponse" | "lawful_nonperson";
    admitted: boolean;
    countQuantityId: string;
  }[];
  outcomes:
    | { kind: "unavailable"; reason: string }
    | {
        kind: "backend_weighted_predictive";
        weightedSummaryId: string;
        candidateProbabilities: Record<string, number>;
        tiedMaximumProbability: number;
        noValidCandidateVotesProbability: number;
        /** The backend must apply the rule to exact integer rows, never rounded shares. */
        derivation: "same_joint_integer_count_rows";
      };
}
export interface CountChamberView {
  chamberSize: number;
  fixedSeatIds: string[];
  requiredFixedSeatIds: string[];
  contestIds: string[];
  scenario: {
    id: string;
    label: string;
    organizingDescription: string;
    vicePresidentialTieBreakParty: string | null;
    sourceSha256: string;
  };
  weightedSummaryId: string;
  derivation: "same_joint_winner_rows_and_fixed_ledger";
  control: {
    democratic: number;
    republican: number;
    unresolvedRace: number;
    unresolvedOrganizing: number;
  };
}
export interface CountViewDraft {
  kind: "count_view_preparation";
  provenance: { kind: "invented_fixture"; label: string };
  lifecycle: "ready" | "requires_computation" | "incompatible" | "failed";
  bindings: {
    requestSha256: string;
    configurationSha256: string;
    lawSha256: string;
    dataBundleSha256: string;
    admissionSha256: string;
    outputContractId: string;
    informationCutoff: string;
    targetId: string;
    componentVersions: { id: string; version: string }[];
    jointArtifact: { id: string; sha256: string };
    weightedSummary: {
      id: string;
      sha256: string;
      jointArtifactSha256: string;
      weighting: "original_parent_mass_times_normalized_conditional_weight";
    };
    axes: { eventIds: string[]; contestIds: string[]; candidateIds: string[] };
  };
  audit: {
    status: "passed" | "failed" | "pending";
    sourceAdmission: "admitted" | "held";
    parents: {
      id: string;
      originalMass: number;
      representedMass: number;
      status: "passed" | "failed" | "pending";
    }[];
    qualification: string;
    sameRowAccounting: "passed" | "failed" | "pending";
  };
  scope: {
    label: string;
    requiredEventIds: string[];
    requiredContestIds: string[];
    populationCells: {
      count: number;
      label: string;
      dimensions: string[];
      limitations: string;
    };
    communityProfiles: {
      count: number;
      label: string;
      dimensions: string[];
      limitations: string;
    };
  };
  quantities: CountQuantity[];
  events: CountEventView[];
  contests: CountContestView[];
  chamber?: CountChamberView;
}
export interface CountViewReadiness {
  productionActivated: false;
  valid: boolean;
  status: "unavailable" | "partial" | "complete";
  canShowAdministrativeReports: boolean;
  canShowPopulation: boolean;
  canShowParticipation: boolean;
  canShowRaces: boolean;
  canShowControl: boolean;
  displayableEventIds: string[];
  displayableContestIds: string[];
  winnerContestIds: string[];
  missingEventIds: string[];
  missingContestIds: string[];
  quantityGates: Record<string, { display: boolean; reason: string | null }>;
  contestGates: Record<
    string,
    { counts: boolean; outcomes: boolean; reason: string | null }
  >;
  problems: { code: string; message: string }[];
}

/** Validates an internal typed draft. This deliberately does not admit raw research files. */
export function countViewReadiness(view: CountViewDraft): CountViewReadiness {
  const problems: CountViewReadiness["problems"] = [];
  let valid = true;
  const issue = (code: string, message: string, invalid = false) => {
    problems.push({ code, message });
    if (invalid) valid = false;
  };
  const label = (value: string) =>
    typeof value === "string" && value.trim().length > 0;
  const hash = (value: string) =>
    typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
  const nonnegative = (value: number) => Number.isFinite(value) && value >= 0;
  const probability = (value: number) => nonnegative(value) && value <= 1;
  const close = (a: number, b: number) =>
    Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(a), Math.abs(b));
  const unique = (ids: string[]) =>
    ids.every(label) && new Set(ids).size === ids.length;
  const same = (a: string[], b: string[]) =>
    unique(a) &&
    unique(b) &&
    a.length === b.length &&
    a.every((id) => b.includes(id));
  if (
    view.kind !== "count_view_preparation" ||
    view.provenance.kind !== "invented_fixture" ||
    !label(view.provenance.label)
  )
    issue(
      "fixture",
      "This preparation view requires explicit invented-fixture provenance.",
      true,
    );
  const b = view.bindings;
  for (const digest of [
    b.requestSha256,
    b.configurationSha256,
    b.lawSha256,
    b.dataBundleSha256,
    b.admissionSha256,
    b.jointArtifact.sha256,
    b.weightedSummary.sha256,
  ])
    if (!hash(digest))
      issue(
        "binding",
        "A required immutable identity or source fingerprint is missing.",
        true,
      );
  if (
    ![
      b.outputContractId,
      b.targetId,
      b.jointArtifact.id,
      b.weightedSummary.id,
    ].every(label) ||
    !Number.isFinite(Date.parse(b.informationCutoff)) ||
    !unique(b.componentVersions.map((c) => c.id)) ||
    !b.componentVersions.length ||
    !b.componentVersions.every((c) => label(c.version))
  )
    issue(
      "binding",
      "Target, cutoff, output contract and component versions must be explicit.",
      true,
    );
  if (
    b.weightedSummary.jointArtifactSha256 !== b.jointArtifact.sha256 ||
    b.weightedSummary.weighting !==
      "original_parent_mass_times_normalized_conditional_weight"
  )
    issue(
      "weighting",
      "Summaries must bind the original joint artifact and declared parent-mass weighting.",
      true,
    );
  for (const scope of [
    view.scope.populationCells,
    view.scope.communityProfiles,
  ])
    if (
      !Number.isSafeInteger(scope.count) ||
      scope.count <= 0 ||
      !label(scope.label) ||
      !scope.dimensions.length ||
      !scope.dimensions.every(label) ||
      !label(scope.limitations)
    )
      issue(
        "scope",
        "Population and community support must be named and qualified.",
        true,
      );
  if (
    !label(view.scope.label) ||
    !unique(view.scope.requiredEventIds) ||
    !unique(view.scope.requiredContestIds)
  )
    issue(
      "scope",
      "Coverage requires explicit unique event and contest identities.",
      true,
    );
  if (!unique(view.quantities.map((q) => q.id)))
    issue("quantity", "Quantity IDs must be unique.", true);
  const quantities = new Map(view.quantities.map((q) => [q.id, q]));
  for (const q of view.quantities) {
    const ratio = [
      "candidate_share",
      "turnout_ratio",
      "standardized_propensity",
    ].includes(q.quantityType);
    const unitValid = ratio
      ? q.unit === "ratio"
      : q.quantityType === "noncandidate_count"
        ? ["ballots", "responses", "votes"].includes(q.unit)
        : ["candidate_count", "countable_candidate_total"].includes(
              q.quantityType,
            )
          ? q.unit === "votes"
          : q.unit === "people";
    if (!label(q.label) || !unitValid)
      issue(
        "unit",
        "Every quantity needs its own explicit meaningful unit; noncandidate categories are not candidate votes.",
        true,
      );
    if (ratio && q.denominator.kind === "none")
      issue(
        "denominator",
        "Shares, turnout ratios and propensities require explicit denominators.",
        true,
      );
    if (
      q.denominator.kind === "quantity" &&
      (!quantities.has(q.denominator.quantityId) ||
        !label(q.denominator.label) ||
        !label(q.denominator.interpretation))
    )
      issue(
        "denominator",
        "A named denominator must identify an available quantity and its meaning.",
        true,
      );
    if (
      q.kind === "backend_weighted_predictive" &&
      q.denominator.kind === "quantity" &&
      quantities.get(q.denominator.quantityId)?.kind === "unavailable"
    )
      issue(
        "denominator",
        "A predictive ratio cannot borrow an unavailable denominator summary.",
        true,
      );
    if (
      q.quantityType === "candidate_share" &&
      (q.denominator.kind !== "quantity" ||
        quantities.get(q.denominator.quantityId)?.quantityType !==
          "countable_candidate_total")
    )
      issue(
        "denominator",
        "All-candidate shares require the complete countable-candidate denominator; a named-pair poll share cannot substitute.",
        true,
      );
    if (q.kind === "unavailable") {
      if (!label(q.reason))
        issue(
          "missing",
          "Unavailable is distinct from zero and needs a reason.",
          true,
        );
      continue;
    }
    const values =
      q.kind === "observed_administrative"
        ? [q.value]
        : [q.mean, ...q.intervals.flatMap((i) => [i.lower, i.upper])];
    if (
      !values.every(nonnegative) ||
      (["candidate_share", "standardized_propensity"].includes(
        q.quantityType,
      ) &&
        !values.every(probability))
    )
      issue(
        "quantity",
        "A quantity contains invalid values for its meaning.",
        true,
      );
    if (q.kind === "observed_administrative") {
      if (
        q.quantityType !== "administrative_participation" ||
        !label(q.sourceId) ||
        !["admitted", "held"].includes(q.sourceAdmission) ||
        !label(q.sourceQualification) ||
        !hash(q.sourceSha256) ||
        !Number.isFinite(Date.parse(q.referenceAt)) ||
        !Number.isFinite(Date.parse(q.publicationAt))
      )
        issue(
          "administrative",
          "Administrative participation is an observed report with separate source and dates, not predicted participants.",
          true,
        );
      if (q.sourceAdmission === "held")
        issue(
          "administrative_source_held",
          `Administrative report ${q.id} is held for its own source review: ${q.sourceQualification}`,
        );
    } else {
      if (
        q.weightedSummaryId !== b.weightedSummary.id ||
        !q.intervals.length ||
        new Set(q.intervals.map((i) => i.level)).size !== q.intervals.length ||
        !q.intervals.every(
          (i) => i.level > 0 && i.level < 1 && i.lower <= i.upper,
        )
      )
        issue(
          "summary",
          "Predictive quantities need the bound backend-weighted summary and explicit interval levels.",
          true,
        );
    }
  }
  const predictive = (id: string | undefined, type: CountQuantityType) => {
    const q = id ? quantities.get(id) : undefined;
    return q?.kind === "backend_weighted_predictive" && q.quantityType === type
      ? q
      : undefined;
  };
  const eventIds = view.events.map((e) => e.id),
    contestIds = view.contests.map((c) => c.id);
  const candidateIds = view.contests.flatMap((c) =>
    c.candidates.map((candidate) => candidate.id),
  );
  if (
    !same(eventIds, b.axes.eventIds) ||
    !same(contestIds, b.axes.contestIds) ||
    !same(candidateIds, b.axes.candidateIds)
  )
    issue(
      "axes",
      "All event, contest and named-candidate axes must match the immutable summary binding exactly.",
      true,
    );
  if (
    eventIds.some((id) => !view.scope.requiredEventIds.includes(id)) ||
    contestIds.some((id) => !view.scope.requiredContestIds.includes(id))
  )
    issue(
      "unexpected_scope",
      "Event or contest IDs outside the declared required scope cannot supply coverage or complete national accounting.",
      true,
    );
  const events = new Map(view.events.map((e) => [e.id, e]));
  for (const event of view.events) {
    if (
      !label(event.state) ||
      !Number.isFinite(Date.parse(event.date)) ||
      !same(
        event.contestIds,
        view.contests.filter((c) => c.eventId === event.id).map((c) => c.id),
      )
    )
      issue(
        "event",
        "Each physical event needs its own date and exact contest membership.",
        true,
      );
    if (
      quantities.get(event.participantQuantityId)?.quantityType !==
        "event_participants" ||
      quantities.get(event.citizenExposureQuantityId)?.quantityType !==
        "citizen_adult_exposure"
    )
      issue(
        "event",
        "Event participants and citizen exposure must be separate named quantities.",
        true,
      );
    if (
      event.administrativeReportQuantityId &&
      quantities.get(event.administrativeReportQuantityId)?.kind !==
        "observed_administrative"
    )
      issue(
        "administrative",
        "An administrative report cannot be substituted with predicted participation.",
        true,
      );
    if (event.turnoutQuantityId) {
      const q = quantities.get(event.turnoutQuantityId);
      if (
        q?.quantityType !== "turnout_ratio" ||
        q.denominator.kind !== "quantity" ||
        q.denominator.quantityId !== event.citizenExposureQuantityId
      )
        issue(
          "turnout",
          "Realized turnout ratios must name the citizen exposure denominator; ratios above one are permitted.",
          true,
        );
    }
  }
  const displayableContestIds: string[] = [],
    winnerContestIds: string[] = [];
  for (const contest of view.contests) {
    const event = events.get(contest.eventId);
    if (!event || !label(contest.rule.id) || !label(contest.rule.qualification))
      issue(
        "contest",
        "A contest needs a physical event and admitted dated rule identity.",
        true,
      );
    if (contest.applicability === "absent") {
      if (
        contest.candidates.length ||
        contest.noncandidateCategories.length ||
        contest.outcomes.kind !== "unavailable"
      )
        issue(
          "absence",
          "An absent contest has no candidate counts or undervote outcomes.",
          true,
        );
      continue;
    }
    const categories = contest.noncandidateCategories;
    if (
      !unique(categories.map((c) => c.id)) ||
      categories.some((c) => !label(c.label))
    )
      issue(
        "category",
        "Noncandidate categories need distinct named meanings.",
        true,
      );
    for (const category of categories) {
      const q = quantities.get(category.countQuantityId);
      const units =
        category.kind === "nonresponse"
          ? ["responses"]
          : category.kind === "lawful_nonperson"
            ? ["votes", "ballots", "responses"]
            : ["ballots"];
      if (q?.quantityType !== "noncandidate_count" || !units.includes(q.unit))
        issue(
          "category_unit",
          "A noncandidate category needs the unit matching its declared meaning within the admitted event; blank and invalid ballots are not candidate votes.",
          true,
        );
    }
    for (const candidate of contest.candidates)
      if (
        !label(candidate.name) ||
        candidate.name.trim().toLowerCase() === "other" ||
        (candidate.party !== null && !label(candidate.party)) ||
        !hash(candidate.identitySourceSha256) ||
        quantities.get(candidate.countQuantityId)?.quantityType !==
          "candidate_count" ||
        quantities.get(candidate.shareQuantityId)?.quantityType !==
          "candidate_share"
      )
        issue(
          "candidate",
          "Every candidate needs a complete named identity; pooled 'Other' is not a person.",
          true,
        );
    const supported =
      contest.rule.family === "single_mark_plurality" &&
      contest.rule.admission === "admitted" &&
      contest.rosterComplete &&
      contest.candidates.length > 0 &&
      categories.every((c) => c.admitted);
    if (!supported) {
      issue(
        "unsupported_rule_or_roster",
        `Contest ${contest.id} is unavailable pending complete roster/category and rule admission.`,
      );
      if (contest.outcomes.kind !== "unavailable")
        issue(
          "unsupported_winner",
          "Unsupported rules or incomplete rosters cannot supply winner probabilities.",
          true,
        );
      continue;
    }
    const counts = [
      ...contest.candidates.map((c) =>
        predictive(c.countQuantityId, "candidate_count"),
      ),
      ...categories.map((c) =>
        predictive(c.countQuantityId, "noncandidate_count"),
      ),
    ];
    const participants = predictive(
      event?.participantQuantityId,
      "event_participants",
    );
    if (counts.every(Boolean) && participants) {
      if (
        !close(
          counts.reduce((sum, q) => sum + q!.mean, 0),
          participants.mean,
        )
      )
        issue(
          "accounting",
          "Applicable single-mark candidate and admitted noncandidate counts must reconcile to the same event participant mean.",
          true,
        );
      else displayableContestIds.push(contest.id);
    }
    if (contest.outcomes.kind === "backend_weighted_predictive") {
      const o = contest.outcomes;
      if (
        !same(
          Object.keys(o.candidateProbabilities),
          contest.candidates.map((c) => c.id),
        ) ||
        !Object.values(o.candidateProbabilities).every(probability) ||
        !probability(o.tiedMaximumProbability) ||
        !probability(o.noValidCandidateVotesProbability) ||
        !close(
          Object.values(o.candidateProbabilities).reduce((a, c) => a + c, 0) +
            o.tiedMaximumProbability +
            o.noValidCandidateVotesProbability,
          1,
        ) ||
        o.weightedSummaryId !== b.weightedSummary.id ||
        o.derivation !== "same_joint_integer_count_rows"
      )
        issue(
          "outcomes",
          "Candidate wins, tied maximum and no valid votes must stay separate and sum to one, bound to the same count rows.",
          true,
        );
      else if (displayableContestIds.includes(contest.id))
        winnerContestIds.push(contest.id);
    }
  }
  const parents = view.audit.parents;
  const parentAudit =
    parents.length > 0 &&
    unique(parents.map((p) => p.id)) &&
    parents.every(
      (p) =>
        probability(p.originalMass) &&
        probability(p.representedMass) &&
        p.status === "passed" &&
        close(p.originalMass, p.representedMass),
    ) &&
    close(
      parents.reduce((sum, p) => sum + p.originalMass, 0),
      1,
    );
  if (!parentAudit)
    issue(
      "parent_mass",
      "A failed or missing parent cannot be dropped or have its mass redistributed. Summary declarations alone do not verify the underlying weights.",
    );
  if (
    view.audit.status !== "passed" ||
    view.audit.sourceAdmission !== "admitted" ||
    view.audit.sameRowAccounting !== "passed" ||
    !label(view.audit.qualification)
  )
    issue(
      "audit",
      "Predictive display requires declared source admission and passed numerical/parent audit context.",
    );
  const predictiveReady =
    valid &&
    view.lifecycle === "ready" &&
    view.audit.status === "passed" &&
    view.audit.sourceAdmission === "admitted" &&
    view.audit.sameRowAccounting === "passed" &&
    parentAudit;
  const missingEventIds = view.scope.requiredEventIds.filter(
    (id) => !eventIds.includes(id),
  );
  const missingContestIds = view.scope.requiredContestIds.filter(
    (id) => !contestIds.includes(id),
  );
  if (missingEventIds.length || missingContestIds.length)
    issue(
      "coverage",
      "Missing national event or contest coverage cannot be filled from a pilot or race subtotal.",
    );
  let canShowControl = false;
  const chamber = view.chamber;
  if (chamber) {
    const ledger =
      same(chamber.fixedSeatIds, chamber.requiredFixedSeatIds) &&
      same(chamber.contestIds, view.scope.requiredContestIds) &&
      Number.isSafeInteger(chamber.chamberSize) &&
      chamber.chamberSize > 0 &&
      chamber.fixedSeatIds.length + chamber.contestIds.length ===
        chamber.chamberSize;
    const values = Object.values(chamber.control);
    const bound =
      chamber.weightedSummaryId === b.weightedSummary.id &&
      chamber.derivation === "same_joint_winner_rows_and_fixed_ledger" &&
      label(chamber.scenario.id) &&
      label(chamber.scenario.label) &&
      label(chamber.scenario.organizingDescription) &&
      hash(chamber.scenario.sourceSha256);
    if (
      !values.every(probability) ||
      !close(
        values.reduce((a, c) => a + c, 0),
        1,
      ) ||
      !bound
    ) {
      issue(
        "chamber",
        "Control must bind joint winner rows, a named organizing/VP scenario and separate race/organizing uncertainty.",
        true,
      );
    }
    canShowControl =
      predictiveReady &&
      valid &&
      ledger &&
      !missingEventIds.length &&
      !missingContestIds.length &&
      chamber.contestIds.every((id) => winnerContestIds.includes(id));
    if (!canShowControl)
      issue(
        "chamber_coverage",
        "Control is unavailable without complete qualified event/contest support and a complete fixed-seat ledger.",
      );
  }
  const usable = predictiveReady && valid;
  const displayableEventIds = usable
    ? view.events
        .filter(
          (e) =>
            predictive(e.participantQuantityId, "event_participants") &&
            predictive(e.citizenExposureQuantityId, "citizen_adult_exposure"),
        )
        .map((e) => e.id)
    : [];
  const canShowPopulation =
    usable &&
    view.quantities.some(
      (q) =>
        q.kind === "backend_weighted_predictive" &&
        ["adult_residents", "citizen_adult_exposure"].includes(q.quantityType),
    );
  const quantityGates: CountViewReadiness["quantityGates"] = {};
  for (const q of view.quantities) {
    const administrativeValid =
      q.kind === "observed_administrative" &&
      q.quantityType === "administrative_participation" &&
      q.unit === "people" &&
      label(q.label) &&
      nonnegative(q.value) &&
      label(q.sourceId) &&
      q.sourceAdmission === "admitted" &&
      label(q.sourceQualification) &&
      hash(q.sourceSha256) &&
      Number.isFinite(Date.parse(q.referenceAt)) &&
      Number.isFinite(Date.parse(q.publicationAt));
    const contestQuantity = [
      "candidate_count",
      "candidate_share",
      "countable_candidate_total",
      "noncandidate_count",
    ].includes(q.quantityType);
    const coveredByContest =
      !contestQuantity ||
      view.contests.some(
        (c) =>
          displayableContestIds.includes(c.id) &&
          (c.candidates.some(
            (candidate) =>
              candidate.countQuantityId === q.id ||
              candidate.shareQuantityId === q.id ||
              (quantities.get(candidate.shareQuantityId)?.denominator.kind ===
                "quantity" &&
                (
                  quantities.get(candidate.shareQuantityId)!.denominator as {
                    quantityId: string;
                  }
                ).quantityId === q.id),
          ) ||
            c.noncandidateCategories.some(
              (category) => category.countQuantityId === q.id,
            )),
      );
    const display =
      administrativeValid ||
      (usable && coveredByContest && q.kind === "backend_weighted_predictive");
    quantityGates[q.id] = {
      display,
      reason: display
        ? null
        : q.kind === "unavailable"
          ? q.reason
          : q.kind === "observed_administrative"
            ? `Administrative report is held pending its own source review and dated attribution: ${q.sourceQualification || "No source qualification supplied."}`
            : "Predictive display is held pending complete binding, source and joint-parent/accounting audit declarations.",
    };
  }
  const contestGates: CountViewReadiness["contestGates"] = {};
  for (const c of view.contests) {
    const counts = usable && displayableContestIds.includes(c.id),
      outcomes = usable && winnerContestIds.includes(c.id);
    contestGates[c.id] = {
      counts,
      outcomes,
      reason:
        counts && outcomes
          ? null
          : c.applicability === "absent"
            ? "No contest applies to this event; this is not an undervote."
            : "Contest display requires admitted complete support, same-row counts and a bound rule outcome summary.",
    };
  }
  return {
    productionActivated: false,
    valid,
    status: !usable
      ? "unavailable"
      : missingEventIds.length || missingContestIds.length || problems.length
        ? "partial"
        : "complete",
    canShowAdministrativeReports: view.quantities.some(
      (q) =>
        q.kind === "observed_administrative" && quantityGates[q.id]?.display,
    ),
    canShowPopulation,
    canShowParticipation: displayableEventIds.length > 0,
    canShowRaces: usable && displayableContestIds.length > 0,
    canShowControl: usable && canShowControl,
    displayableEventIds,
    displayableContestIds: usable ? displayableContestIds : [],
    winnerContestIds: usable ? winnerContestIds : [],
    missingEventIds,
    missingContestIds,
    quantityGates,
    contestGates,
    problems,
  };
}
