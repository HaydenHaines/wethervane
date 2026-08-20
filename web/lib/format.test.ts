import assert from "node:assert/strict";

import { formatMargin, formatPartisanMargin } from "./format";

assert.equal(formatMargin(0.5), "EVEN");
assert.equal(formatMargin(0.504), "EVEN");
assert.equal(formatMargin(0.532), "D+3.2");
assert.equal(formatMargin(0.468), "R+3.2");

assert.equal(formatPartisanMargin(0.004), "EVEN");
assert.equal(formatPartisanMargin(0.005), "D+0.5");
assert.equal(formatPartisanMargin(-0.032), "R+3.2");
