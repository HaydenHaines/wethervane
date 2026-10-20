import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  deriveRibbonLayout,
  ribbonMinimumWidth,
  ribbonSeatLabel,
} from "../lib/ribbon-layout.ts";

const read = (name) =>
  JSON.parse(
    readFileSync(
      new URL("../public/data/explorer/" + name + ".json", import.meta.url),
      "utf8",
    ),
  );
const pair = (race) => ({
  id: race.id,
  state: race.state,
  seatId: race.seatId,
  candidates: race.outcomes.map((o) => ({
    id: o.id,
    name: o.name,
    party: o.party,
  })),
  firstShareMean: race.outcomes[0].share.mean,
  firstWinProbability: race.outcomes[0].winProbability,
  secondWinProbability: race.outcomes[1].winProbability,
});

for (const id of read("registry")
  .runs.filter((run) => !run.historical)
  .map((run) => run.id)) {
  test(
    id +
      ": all saved scenario markers agree with independently derived layouts",
    () => {
      const bundle = read(id),
        races = bundle.races.map(pair);
      for (const [name, scenario] of Object.entries(bundle.tipping.scenarios)) {
        const actual = deriveRibbonLayout([...races].reverse(), scenario);
        assert.deepEqual(
          actual.raceOrder,
          bundle.tipping.raceOrder,
          name + " order",
        );
        for (const party of ["D", "R"]) {
          assert.equal(
            actual.caucuses[party].projectedTippingRaceId,
            scenario[party].projectedTippingRaceId,
            name + party,
          );
          assert.ok(
            Math.abs(
              actual.caucuses[party].projectedMarginPp -
                scenario[party].projectedMarginPp,
            ) < 1e-10,
          );
          assert.equal(
            actual.caucuses[party].requiredContestedWins,
            scenario[party].requiredContestedWins,
          );
          assert.equal(actual.caucuses[party].status, scenario[party].status);
        }
      }
    },
  );
}

function race(index, share = 0.9 - index * 0.01) {
  return {
    id: `2034:seat-${String(index).padStart(2, "0")}`,
    seatId: `seat-${index}`,
    state: "MN",
    candidates: [
      { id: `first-${index}`, party: "D", name: `First ${index}` },
      { id: `second-${index}`, party: "R", name: `Second ${index}` },
    ],
    firstShareMean: share,
    firstWinProbability: 0.6,
    secondWinProbability: 0.4,
  };
}
function metadata(races, threshold = 3, retained = 1) {
  return {
    D: {
      thresholdSeats: threshold,
      retainedSeats: retained,
      guaranteedRaceIds: [],
      eligibleCandidateByRace: Object.fromEntries(
        races.map((r) => [r.id, r.candidates[0].id]),
      ),
    },
    R: {
      thresholdSeats: threshold,
      retainedSeats: retained,
      guaranteedRaceIds: [],
      eligibleCandidateByRace: Object.fromEntries(
        races.map((r) => [r.id, r.candidates[1].id]),
      ),
    },
  };
}

for (const count of [5, 12, 40])
  test(
    count +
      " shuffled races determine their own order, thresholds and readable width",
    () => {
      const races = Array.from({ length: count }, (_, i) =>
        race(i, 0.95 - (i * 0.9) / count),
      );
      const scenario = metadata(races, Math.ceil(count / 2) + 2, 2),
        before = JSON.stringify({ races, scenario });
      const shuffled = [
        ...races.filter((_, i) => i % 2),
        ...races.filter((_, i) => !(i % 2)),
      ].reverse();
      const layout = deriveRibbonLayout(shuffled, scenario),
        needed = Math.ceil(count / 2);
      assert.deepEqual(
        layout.raceOrder,
        races.map((r) => r.id),
      );
      assert.equal(
        layout.caucuses.D.projectedTippingRaceId,
        races[needed - 1].id,
      );
      assert.equal(
        layout.caucuses.R.projectedTippingRaceId,
        races[count - needed].id,
      );
      assert.equal(layout.raceOrder.length, count);
      assert.equal(
        ribbonMinimumWidth(count),
        98 + count * 22 + (count + 1) * 3,
      );
      assert.equal(
        JSON.stringify({ races, scenario }),
        before,
        "layout must not mutate source races/memberships",
      );
    },
  );

test("regular and special contests in one state remain distinct, with stable ID ties", () => {
  const regular = { ...race(0, 0.5), id: "2034:MN:senate:regular" },
    special = { ...race(1, 0.5), id: "2034:MN:senate:special" };
  const races = [special, regular, ...[2, 3, 4].map((i) => race(i, 0.4))];
  const scenario = metadata(races, 2, 0),
    layout = deriveRibbonLayout(races, scenario);
  assert.deepEqual(layout.raceOrder.slice(0, 2), [regular.id, special.id]);
  assert.equal(layout.caucuses.D.projectedTippingRaceId, special.id);
  assert.equal(ribbonSeatLabel(regular), "’34");
  assert.equal(ribbonSeatLabel(special), "SP");
  assert.equal(
    ribbonSeatLabel({ id: "unversioned-race", seatId: "regular" }),
    "REG",
  );
});

test("retained seats and guaranteed contests change the threshold without removing independent tiles", () => {
  const races = Array.from({ length: 5 }, (_, i) => race(i, 0.8 - i * 0.1));
  races[0].candidates[0].party = "I";
  const scenario = metadata(races, 4, 1);
  scenario.R.guaranteedRaceIds = [races[0].id];
  const initial = deriveRibbonLayout(races, scenario);
  assert.equal(initial.caucuses.R.requiredContestedWins, 2);
  assert.equal(initial.caucuses.R.projectedTippingRaceId, races[3].id);
  assert.equal(initial.raceOrder.length, 5);
  assert.equal(races[0].candidates[0].party, "I");
  scenario.D.retainedSeats = 2;
  assert.equal(
    deriveRibbonLayout(races, scenario).caucuses.D.projectedTippingRaceId,
    races[1].id,
  );
});

test("already-reached and unreachable thresholds have no invented marker or margin", () => {
  const races = Array.from({ length: 5 }, (_, i) => race(i)),
    scenario = metadata(races);
  scenario.D.retainedSeats = 4;
  scenario.R.thresholdSeats = 20;
  const layout = deriveRibbonLayout(races, scenario);
  assert.equal(layout.caucuses.D.status, "already_reached");
  assert.equal(layout.caucuses.R.status, "unreachable");
  for (const party of ["D", "R"]) {
    assert.equal(layout.caucuses[party].projectedTippingRaceId, null);
    assert.equal(layout.caucuses[party].projectedMarginPp, null);
    assert.ok(layout.caucuses[party].statusReason);
  }
});

test("projected layout never reads or estimates joint-draw frequencies or saved marker/order fields", () => {
  const races = Array.from({ length: 5 }, (_, i) => race(i)),
    scenario = metadata(races);
  for (const party of ["D", "R"]) {
    Object.defineProperty(scenario[party], "frequencyByRace", {
      get() {
        throw Error("Must not inspect draw frequencies");
      },
    });
    Object.defineProperty(scenario[party], "projectedTippingRaceId", {
      get() {
        throw Error("Must derive its own marker");
      },
    });
  }
  assert.doesNotThrow(() => deriveRibbonLayout(races, scenario));
  assert.equal(
    deriveRibbonLayout(races, scenario).caucuses.D.projectedTippingRaceId,
    races[1].id,
  );
});

test("invalid shares, duplicate races and unknown member candidates fail explicitly", () => {
  const races = [race(0), race(1)],
    scenario = metadata(races);
  assert.throws(
    () => deriveRibbonLayout([races[0], races[0]], scenario),
    /unique/,
  );
  assert.throws(
    () => deriveRibbonLayout([{ ...races[0], firstShareMean: NaN }], scenario),
    /valid mean/,
  );
  scenario.D.eligibleCandidateByRace[races[0].id] = "not-a-candidate";
  assert.throws(() => deriveRibbonLayout(races, scenario), /Unknown eligible/);
});
