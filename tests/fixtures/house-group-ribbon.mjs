import { modeledWinnerFixture } from "./house-winner.mjs";

function district(index) { return `A${String.fromCharCode(65 + Math.floor(index / 99))}-${String(index % 99).padStart(2, "0")}`; }
/** Invented probabilities and geography codes; never public forecast inputs. */
export function groupRibbonFixtures(count = 16) {
  const distributions = [{ D: .95, R: .04, O: .01 }, { D: .65, R: .3, O: .05 }, { D: .4, R: .4, O: .2 }, { D: .1, R: .85, O: .05 }, { D: .1, R: .1, O: .8 }, { D: 1 / 3, R: 1 / 3, O: 1 / 3 }];
  return Array.from({ length: count }, (_, index) => {
    const fixture = modeledWinnerFixture(); fixture.district = district(index);
    fixture.prediction.probabilities = Object.entries(distributions[index % distributions.length]).map(([groupId, probability]) => ({ groupId, probability }));
    fixture.truth = { status: "unavailable", reason: "No recorded outcome is supplied for this invented district." };
    if (index % 7 === 6) fixture.prediction = { status: "unavailable", reason: "No valid group prediction is supplied in this example." };
    return fixture;
  });
}
