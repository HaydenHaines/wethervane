# Independent review: invented fixed-profile candidate allocation oracle

I checked the oracle against the pinned 2018 provider source and reran its seven standard-library tests. The fixed-input algebra is correct: softmax candidate choice, response split, noncandidate simplex allocation, and common event scaling. The independent tiny probe reproduces the 7/3-to-4 ratio-of-sums example and the same-odds invariant.

The main applicability limit is that the actual provider response is cell-specific, while this oracle accepts one scalar response for all profiles. An invented counterexample gives an aggregate D/R ratio of 43/7 under responses 0.9 and 0.1, versus 7/3 for the oracle's common response. That is within the declared toy scope; it prevents direct use as a numerical emulation of the selected graph until the response weighting is represented.

The example shows composition redistribution. At fixed event intensity and response, total named-category mass is unchanged as minor candidates are added. It cannot by itself explain a total candidate-mark excess; fitted event scale, intensity, response, observation structure, and their interactions are outside this calculation.

Use “profile intensity weights” and “expected category marks” when describing this example. Do not call it physical turnout or candidate-specific calibration. No Arizona outcomes/raw rows, model imports, fits, forecasts or owner jobs were used. Exact pins and the reproduced probe are in source-pins.json and luna_probe.*. My first supplementary invariant fixture had the wrong D/R odds for its second profile; I corrected the invented fixture before the passing run and made no subject changes.
